"""LongMemEval-S (Wu et al., ICLR 2025): long-term memory of a chat assistant.

Each question comes with a history of about 47 earlier chat sessions
(~490 messages, ~120k tokens) between a user and an assistant, over weeks or
months, and is asked "as of" a later date. The answer hangs on a few details
in that history: one session, several, a changed fact, a date, a preference,
or nothing at all (abstention: the right answer is "I don't know").

This is the benchmark where sora's own harness does the work. The history is
fed through sora's real memory pipeline, as if it had lived those sessions:
every exchange saved, consolidation (the small model) turning each stretch of
chat into facts and a dated episode, every 6 exchanges and at each session's
end. Facts and episodes are stamped with the session's date. Then the
question is an ordinary turn, as of its own date: the gate decides whether to
recall, recall searches facts and episodes, the model answers.

Grading follows LongMemEval's published judge prompts, one per question type
(temporal answers may be off by one; knowledge updates must use the latest
value; preferences are judged against a rubric; abstentions are right only
if sora says it doesn't know). MIT licensed; kept in ~/.sora/benchmarks.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from loop.llm import ask_json, schema
from memory.consolidation import EVERY, consolidate
from ops.bench.runner import Item, download

URL = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"

VERDICT = schema({"correct": {"type": "boolean"}, "reason": {"type": "string"}})

# Adapted from LongMemEval's evaluation prompts (src/evaluation/evaluate_qa.py).
GENERIC = ("I will give you a question, a correct answer, and a response from a model. Answer correct=true if the "
           "response contains the correct answer, or is equivalent to it, or contains all the intermediate steps to "
           "get it. If the response only contains a subset of the information the answer requires, correct=false.")
RULES = {
    "temporal-reasoning": GENERIC + (" Do not penalize off-by-one errors in the number of days, weeks or months: "
                                     "predicting 19 days when the answer is 18 is still correct."),
    "knowledge-update": GENERIC + (" If the response mentions earlier information along with an updated answer, it "
                                   "is correct as long as the updated answer is the required one."),
    "single-session-preference": ("I will give you a question, a rubric describing the desired personalized "
                                  "response, and a response from a model. correct=true if the response satisfies the "
                                  "rubric. It need not reflect every point; it is correct as long as it recalls and "
                                  "uses the user's personal information correctly."),
    "abstention": ("I will give you an unanswerable question, an explanation, and a response from a model. "
                   "correct=true if the model correctly identifies the question as unanswerable: it may say the "
                   "information is incomplete, or that some other information is given but not what was asked."),
}
GRADE = """{rule}

Question: {question}
{label}: {answer}
Model response: {response}

Give correct (true/false) and a one-sentence reason."""


def parse_date(text: str) -> datetime:
    """'2023/05/30 (Tue) 23:40' -> datetime."""
    day, _, rest = text.partition(" (")
    return datetime.strptime(f"{day} {rest.partition(') ')[2]}", "%Y/%m/%d %H:%M")


def kind_of(row: dict) -> str:
    return "abstention" if row["question_id"].endswith("_abs") else row["question_type"]


def exchanges(session: list[dict]) -> list[tuple[str, str]]:
    """(user, assistant) pairs; consecutive messages from one side are joined."""
    pairs, user, reply = [], [], []
    for turn in session:
        if turn["role"] == "user":
            if reply:
                pairs.append(("\n\n".join(user), "\n\n".join(reply)))
                user, reply = [], []
            user.append(turn["content"])
        else:
            reply.append(turn["content"])
    if user or reply:
        pairs.append(("\n\n".join(user) or "(continued)", "\n\n".join(reply)))
    return pairs


def ingest(agent, row: dict, notify=None) -> dict:
    """Feed a question's history through sora's memory, oldest session first.
    Returns counts (sessions, exchanges, consolidations)."""
    store, sessions = agent.store, sorted(zip(row["haystack_dates"], row["haystack_sessions"]),
                                          key=lambda d: parse_date(d[0]))
    counts = {"sessions": 0, "exchanges": 0, "consolidations": 0}
    every = agent.settings.consolidate_every or EVERY
    for date, session in sessions:
        at = parse_date(date).strftime("%Y-%m-%d %H:%M:%S")
        pairs = exchanges(session)
        sid = store.ensure_session(None, pairs[0][0] if pairs else "session")
        for user, reply in pairs:
            store.add_exchange(sid, user, reply, created_at=at)
            counts["exchanges"] += 1
            if consolidate(store, agent.small, agent.small_id, sid, notify, every=every, at=at):
                counts["consolidations"] += 1
        if consolidate(store, agent.small, agent.small_id, sid, notify, every=every, force=True, at=at):
            counts["consolidations"] += 1
        counts["sessions"] += 1
    return counts


class LongMemEval:
    name = "longmemeval"
    group_by = "type"
    strata_key = "type"
    parallel = 5                                   # questions at once, each in its own home
    settings = {"browser": False}                  # no web pages in this test: skip starting a browser
    closed_book = False

    def __init__(self, grader, grader_model: str):
        self.grader, self.grader_model = grader, grader_model
        self.rows: dict[str, dict] = {}

    def load(self, folder: Path) -> list[Item]:
        data = json.loads(download(URL, folder / "longmemeval_s_cleaned.json").read_text(encoding="utf-8"))
        self.rows = {r["question_id"]: r for r in data}
        return [Item(r["question_id"], r["question"], str(r["answer"]), {"type": kind_of(r)}) for r in data]

    def case(self, item: Item) -> dict:
        row = self.rows[item.id]
        case = {"message": row["question"], "now": parse_date(row["question_date"])}
        if not self.closed_book:
            case["prepare"] = lambda agent, tracer, run_id, _row=row: self._prepare(agent, tracer, run_id, _row)
        return case

    @staticmethod
    def _prepare(agent, tracer, run_id: str, row: dict) -> float:
        """Ingestion as its own traced turn, so its cost is counted and visible."""
        turn = tracer.start(f"(memory: ingest {len(row['haystack_sessions'])} past sessions)", None, "eval",
                            agent.settings.small_model, run_id)
        counts = ingest(agent, row, turn.notify)
        done = turn.finish(f"ingested {counts['sessions']} sessions, {counts['exchanges']} exchanges, "
                           f"{counts['consolidations']} consolidations, {len(agent.store.list_facts())} facts", "ok")
        return done.get("cost_usd") or 0.0

    def grade(self, item: Item, turn: dict) -> dict:
        kind = item.meta["type"]
        prompt = GRADE.format(rule=RULES.get(kind, GENERIC), question=item.question,
                              label={"single-session-preference": "Rubric", "abstention": "Explanation"}
                              .get(kind, "Correct answer"),
                              answer=item.answer, response=(turn.get("reply") or "(no reply)")[:6000])
        verdict = ask_json(self.grader, self.grader_model, prompt, VERDICT, name="grader", max_tokens=1024)
        ok = bool(verdict["correct"])
        return {"checks": {"correct": ok},
                "judge": {"score": 1.0 if ok else 0.0, "grade": 5 if ok else 1,
                          "reason": f"{kind}: {verdict['reason']}", "extracted": (turn.get("reply") or "")[:80]}}
