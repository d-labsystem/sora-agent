"""The retrieval gate: one cheap decision before each turn.

Searching memory on every message has two costs. It adds tokens, and worse,
it adds *irrelevant* context ("hi" pulls in random facts), which can distract
the main model. So a small model looks at the new message, plus the last
exchange, and answers three questions:

    retrieve    does answering this need anything from long-term memory?
    query       if so, what words should we search for? (rewritten, so
                "what about her birthday?" becomes "Priya birthday")
    correction  is the user telling us the previous reply was wrong?

`correction` is the hook for self-learning: memory/learning.py reflects on
turns the user corrected. Asking here costs nothing extra, since the gate
already reads the last exchange.

The gate fails open. If the call errors, we search with the raw message and
assume no correction: a wasted search is cheaper than a forgotten fact.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from loop.llm import ask_json, schema
from memory.store import MemoryStore

log = logging.getLogger(__name__)

GATE_FORMAT = schema({
    "retrieve": {"type": "boolean"},
    "query": {"type": "string"},
    "correction": {"type": "boolean"},
    "complex": {"type": "boolean"},
    "scope": {"type": "string", "enum": ["one", "all"]},
    "queries": {"type": "array", "items": {"type": "string"}},
    "reason": {"type": "string"},
})

# For "all" questions (how many, total, list every...), recall searches with each of the
# gate's extra phrasings and keeps more of what it finds. LongMemEval-S, 2026-11-16: "how
# many instruments do I own?" found 3 of 4, because the four were mentioned in passing in
# five chats ("my Korg B1", "my old Pearl Export") and "musical instruments" matched only
# some of them, while recall kept the top 5 facts and 4 excerpts.
WIDE = {"facts": 12, "episodes": 5, "excerpts": 10}
MAX_QUERIES = 4

# A message about the user ("my", "I", "me") always searches, whatever the gate says.
# LongMemEval-S, 2026-11-17: with Qwen3.5-9B as the gate model, it skipped retrieval on 11
# of 100 questions ("Can you suggest some activities I can do in the evening?" was "a
# general request"; one said "requires memory retrieval" and still answered false), and
# 9 of those 11 were answered wrong. Haiku skipped none. This rule catches 10 of the 11
# (not "which trip did the narrator take first?"). A search is a few local lookups; a
# skipped one is usually a wrong answer.
PERSONAL = re.compile(r"\b(i|i'm|i've|i'd|i'll|me|my|mine|myself)\b", re.IGNORECASE)

PROMPT = """You decide whether a personal assistant should search its long-term memory
before answering, and whether the user is correcting the assistant.

Memory holds facts about the user (people, preferences, plans, projects) and
dated summaries of past conversations.

retrieve: true if the answer could depend on anything the user said in an
earlier conversation. False for greetings, general knowledge, or requests fully
answered by the message itself.
query: if retrieve is true, a few search keywords (names, topics), with
pronouns resolved using the previous exchange. Empty otherwise.
correction: true only if the new message says or implies the previous reply
was wrong, unhelpful, or ignored an instruction.
complex: true if doing what the user asks takes several distinct steps, such as
researching or comparing several things, then combining them, or finding
information and then saving, scheduling or writing something with it. False
for a single question, chat, or one simple action.
scope: "all" if the answer needs every relevant mention from past conversations, such as
how many, how much in total, list all, every time, which ones; "one" otherwise.
queries: only when scope is "all", 2-4 more specific searches for the kinds of things that
would be mentioned (for "how many instruments do I own": "guitar", "piano", "drum set",
"violin"). Empty otherwise.
reason: one short sentence.

{previous}New message:
{message}"""


@dataclass
class GateResult:
    retrieve: bool
    query: str
    correction: bool
    reason: str
    complex: bool = False          # several steps: the harness asks for a plan first
    scope: str = "one"             # "all": the answer needs every mention (counts, totals, lists)
    queries: list[str] = field(default_factory=list)   # extra searches for "all" questions
    excerpts: list[str] = field(default_factory=list)   # words actually said in past conversations
    facts: list[str] = field(default_factory=list)
    episodes: list[str] = field(default_factory=list)
    failed: bool = False
    forced: bool = False           # the gate said no, but the message is about the user


def _previous(last: dict | None) -> str:
    if not last:
        return ""
    return (f"Previous exchange:\nUser: {last['user'][:1000]}\n"
            f"Assistant: {last['reply'][:1000]}\n\n")


def decide(client, model: str, message: str, last: dict | None, notify=None) -> GateResult:
    """Ask the small model. On any failure, search with the message itself."""
    try:
        answer = ask_json(client, model,
                          PROMPT.format(previous=_previous(last), message=message),
                          GATE_FORMAT, notify, name="retrieval gate", max_tokens=1024)
        query = answer["query"].strip() or message
        scope = "all" if answer.get("scope") == "all" else "one"
        extra = [q.strip() for q in answer.get("queries") or [] if q.strip()] if scope == "all" else []
        retrieve = bool(answer["retrieve"])
        forced = not retrieve and bool(PERSONAL.search(message))
        return GateResult(retrieve or forced, query,
                          bool(answer["correction"]) and last is not None,
                          answer["reason"], bool(answer.get("complex", False)),
                          scope=scope, queries=extra[:MAX_QUERIES], forced=forced)
    except Exception as exc:
        log.warning("retrieval gate failed, searching anyway: %s", exc)
        return GateResult(True, message, False, f"gate failed: {exc}", failed=True)


def recall(store: MemoryStore, client, model: str, session_id: str, message: str,
           notify=None, top_facts: int = 5, top_episodes: int = 3, top_excerpts: int = 4) -> GateResult:
    """Run the gate, then search memory if it said so. Emits a "gate" span."""
    result = decide(client, model, message, store.last_exchange(session_id), notify)
    if result.retrieve and result.scope == "all":
        queries = [result.query, *result.queries]
        result.facts = _merged([store.search_facts(q, WIDE["facts"], dated=True) for q in queries], WIDE["facts"])
        result.episodes = _merged([store.search_episodes(q, WIDE["episodes"]) for q in queries], WIDE["episodes"])
        result.excerpts = _merged([store.search_messages(q, WIDE["excerpts"], exclude_session=session_id)
                                   for q in queries], WIDE["excerpts"])
    elif result.retrieve:
        result.facts = store.search_facts(result.query, top_facts, dated=True)
        result.episodes = store.search_episodes(result.query, top_episodes)
        # Summaries lose details, and consolidation keeps facts about the user, not what sora
        # said; excerpts of the conversations themselves fill both gaps (this session's own
        # messages are already in the history window).
        result.excerpts = store.search_messages(result.query, top_excerpts, exclude_session=session_id)
    if notify:
        notify("gate", name="memory recall", retrieve=result.retrieve, query=result.query,
               correction=result.correction, complex=result.complex, scope=result.scope, queries=result.queries,
               reason=result.reason, failed=result.failed, forced=result.forced,
               facts=result.facts, episodes=result.episodes, excerpts=result.excerpts)
    return result


def _merged(lists: list[list[str]], limit: int) -> list[str]:
    """Round-robin over each search's results, best first, without repeats, so
    every phrasing contributes its top hits before any adds its tail."""
    out, seen = [], set()
    for rank in range(max((len(x) for x in lists), default=0)):
        for found in lists:
            if rank < len(found) and found[rank] not in seen:
                seen.add(found[rank])
                out.append(found[rank])
    return out[:limit]
