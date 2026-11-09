"""Consolidation: turning raw chat into long-term memory.

Raw messages are too long to search usefully, and they fall out of the
history window after 10 exchanges. So every few exchanges, the small model
reads the messages it hasn't processed yet and writes two things:

    facts     short, standalone statements that stay true
              ("Alex's sister Priya lives in Pune")
    episode   one dated sentence about what happened in this stretch
              ("Planned a weekend trip to Pune with Priya")

This mirrors how people sleep on a day: the details fade, the gist and the
durable facts stay. Neuroscience calls it memory consolidation, hence the name.

It is loss-safe. Messages are marked consolidated only after the facts and
episode are saved, so a failed call just means the same messages are tried
again next time. It runs after the reply is sent, so the user never waits.
"""

from __future__ import annotations

import logging

from loop.llm import ask_json, schema
from memory.store import MemoryStore

log = logging.getLogger(__name__)

EVERY = 6               # exchanges, i.e. 12 messages
MAX_CHARS = 2000        # per message, so one huge paste can't blow the budget
KNOWN_FACTS = 40        # most recent facts shown to avoid repeats

CONSOLIDATION_FORMAT = schema({
    "facts": {"type": "array", "items": {"type": "string"}},
    "episode": {"type": "string"},
})

PROMPT = """You maintain the long-term memory of a personal assistant.
Read the conversation excerpt and return:

facts: durable, standalone facts about the user and their world (people,
preferences, plans with dates, projects, decisions). Each must make sense on
its own, without the conversation: use names, not pronouns. Skip small talk,
one-off requests, things the assistant said that the user did not confirm,
and anything already in the known facts. Return [] if nothing is worth keeping.
episode: one sentence on what happened or was decided in this excerpt.

Known facts:
{known}

Conversation:
{conversation}"""


def _transcript(rows) -> str:
    return "\n".join(f"{r['role'].title()}: {r['content'][:MAX_CHARS]}" for r in rows)


def consolidate(store: MemoryStore, client, model: str, session_id: str,
                notify=None, every: int = EVERY, force: bool = False) -> dict | None:
    """Consolidate this session if enough unprocessed exchanges have built up
    (or always, with force=True). Returns what was saved, or None if nothing ran
    or the call failed. Never raises: memory upkeep must not break a turn."""
    rows = store.unconsolidated(session_id)
    if not rows or (not force and len(rows) < every * 2):
        return None
    known = [f["content"] for f in store.list_facts()[:KNOWN_FACTS]]
    prompt = PROMPT.format(known="\n".join(f"- {k}" for k in known) or "(none)",
                           conversation=_transcript(rows))
    try:
        answer = ask_json(client, model, prompt, CONSOLIDATION_FORMAT, notify,
                          name="consolidation")
    except Exception as exc:
        log.warning("consolidation failed, will retry later: %s", exc)
        if notify:
            notify("memory", name="consolidation failed", error=str(exc), pending=len(rows))
        return None

    added = [f.strip() for f in answer["facts"] if f.strip() and store.add_fact(f.strip(), "consolidation")]
    episode = answer["episode"].strip()
    if episode:
        store.add_episode(episode, session_id)
    store.mark_consolidated(session_id, rows[-1]["id"])
    result = {"messages": len(rows), "facts": added, "episode": episode}
    if notify:
        notify("memory", name="consolidation", **result)
    return result
