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
# v2 (2026-11-17), after a blind check of 60 v1 verdicts (ops/judge_check.py): v1
# never passed a wrong answer, but 13 of its 24 FAILs were correct replies. Three
# causes, all from the judge not seeing the chat history: it called real recalled
# details "fabricated"; it failed right answers that came with caveats or follow-up
# questions; and on abstentions it failed "no record of X" when the reply also
# mentioned the nearest thing it did know.
V2_UNSEEN = (" You do not see the user's chat history. Details, dates, quotes or session numbers in the response "
             "that the reference does not mention may well be real: never mark the response wrong for being "
             "unsupported, invented or fabricated, only if what it says contradicts the reference.")
V2_COMMIT = (" Caveats, hedges, follow-up questions and second thoughts are fine when the response still gives the "
             "correct answer as its answer (its main or best answer). It is wrong if its final answer differs from "
             "the correct one, or if it leaves the choice open between the correct answer and a wrong one.")
V2_RECALL = (" The question asks what the assistant said earlier: a response that correctly recalls it is correct, "
             "even if it now adds a correction, a safety caveat or doubts about it.")
V2_ABSTENTION = ("I will give you an unanswerable question, an explanation of why it can't be answered, and a "
                 "response from a model. correct=true if the response makes clear that it has no record of the "
                 "specific thing asked about (or that the question's premise doesn't match what it knows). It is "
                 "fine, and still correct, to also mention the closest related thing it does know or to ask which "
                 "one the user meant. correct=false only if the response answers as though the thing asked about "
                 "exists, i.e. treats a related item as the answer without flagging the mismatch."
                 " You do not see the user's chat history, so do not judge whether the related details are real.")
JUDGES = {
    "lme-v1": RULES,
    "lme-v2": {"temporal-reasoning": RULES["temporal-reasoning"] + V2_UNSEEN + V2_COMMIT,
               "knowledge-update": RULES["knowledge-update"] + V2_UNSEEN + V2_COMMIT,
               "single-session-assistant": GENERIC + V2_UNSEEN + V2_COMMIT + V2_RECALL,
               "single-session-preference": RULES["single-session-preference"] + V2_UNSEEN,   # a rubric: no single answer
               "abstention": V2_ABSTENTION, "": GENERIC + V2_UNSEEN + V2_COMMIT},
}
JUDGE = "lme-v1"          # the paper's protocol; see JUDGES for the validated alternative

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


# The paper's "full-context" reader: every past session, dated, in the prompt, and no
# memory system at all. It is the fair opponent for sora's memory: the same model
# reading everything, instead of recalling a few facts. Adapted from LongMemEval's
# reader prompt (src/generation/run_generation.py).
FULL_CONTEXT = """I will give you several past chats between you and a user, each with its date. \
Answer the question at the end from them.

History chats:

{history}

Current date: {date}
Question: {question}"""


def history_text(row: dict) -> str:
    """All of a question's past sessions, oldest first, as dated transcripts."""
    sessions = sorted(zip(row["haystack_dates"], row["haystack_sessions"]), key=lambda d: parse_date(d[0]))
    parts = []
    for n, (date, session) in enumerate(sessions, 1):
        lines = "\n".join(f"{t['role']}: {t['content']}" for t in session)
        parts.append(f"### Session {n} ({date})\n{lines}")
    return "\n\n".join(parts)


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
    full_context = False                           # the history goes into the prompt; no memory system

    def __init__(self, grader, grader_model: str, judge: str = JUDGE):
        if judge not in JUDGES:
            raise ValueError(f"unknown judge {judge!r}; one of {', '.join(JUDGES)}")
        self.grader, self.grader_model, self.judge = grader, grader_model, judge
        self.rows: dict[str, dict] = {}

    def load(self, folder: Path) -> list[Item]:
        data = json.loads(download(URL, folder / "longmemeval_s_cleaned.json").read_text(encoding="utf-8"))
        self.rows = {r["question_id"]: r for r in data}
        return [Item(r["question_id"], r["question"], str(r["answer"]), {"type": kind_of(r)}) for r in data]

    def case(self, item: Item) -> dict:
        row = self.rows[item.id]
        case = {"message": row["question"], "now": parse_date(row["question_date"])}
        if self.full_context:
            case["message"] = FULL_CONTEXT.format(history=history_text(row), date=row["question_date"],
                                                  question=row["question"])
            return case
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

    @staticmethod
    def reference_label(item: Item) -> str:
        """What the reference is: an answer, a rubric (preferences) or why no answer exists (abstentions)."""
        return {"single-session-preference": "Rubric", "abstention": "Explanation"}.get(
            item.meta.get("type", ""), "Correct answer")

    def grade(self, item: Item, turn: dict) -> dict:
        kind = item.meta["type"]
        rules = JUDGES[self.judge]
        prompt = GRADE.format(rule=rules.get(kind, rules.get("", GENERIC)), question=item.question,
                              label=self.reference_label(item),
                              answer=item.answer, response=(turn.get("reply") or "(no reply)")[:6000])
        verdict = ask_json(self.grader, self.grader_model, prompt, VERDICT, name="grader", max_tokens=1024)
        ok = bool(verdict["correct"])
        return {"checks": {"correct": ok},
                "judge": {"score": 1.0 if ok else 0.0, "grade": 5 if ok else 1,
                          "reason": f"{kind}: {verdict['reason']}", "extracted": (turn.get("reply") or "")[:80],
                          "version": self.judge}}
