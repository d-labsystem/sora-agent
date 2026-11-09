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
from dataclasses import dataclass, field

from loop.llm import ask_json, schema
from memory.store import MemoryStore

log = logging.getLogger(__name__)

GATE_FORMAT = schema({
    "retrieve": {"type": "boolean"},
    "query": {"type": "string"},
    "correction": {"type": "boolean"},
    "reason": {"type": "string"},
})

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
reason: one short sentence.

{previous}New message:
{message}"""


@dataclass
class GateResult:
    retrieve: bool
    query: str
    correction: bool
    reason: str
    facts: list[str] = field(default_factory=list)
    episodes: list[str] = field(default_factory=list)
    failed: bool = False


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
        return GateResult(bool(answer["retrieve"]), query,
                          bool(answer["correction"]) and last is not None,
                          answer["reason"])
    except Exception as exc:
        log.warning("retrieval gate failed, searching anyway: %s", exc)
        return GateResult(True, message, False, f"gate failed: {exc}", failed=True)


def recall(store: MemoryStore, client, model: str, session_id: str, message: str,
           notify=None, top_facts: int = 5, top_episodes: int = 3) -> GateResult:
    """Run the gate, then search memory if it said so. Emits a "gate" span."""
    result = decide(client, model, message, store.last_exchange(session_id), notify)
    if result.retrieve:
        result.facts = store.search_facts(result.query, top_facts)
        result.episodes = store.search_episodes(result.query, top_episodes)
    if notify:
        notify("gate", name="memory recall", retrieve=result.retrieve, query=result.query,
               correction=result.correction, reason=result.reason, failed=result.failed,
               facts=result.facts, episodes=result.episodes)
    return result
