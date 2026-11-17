"""Consolidation: turning raw chat into long-term memory.

Raw messages are too long to search usefully, and they fall out of the
history window after 10 exchanges. So every few exchanges, the small model
reads the messages it hasn't processed yet and writes two things:

    facts     short, standalone statements that stay true
              ("Alex's sister Priya lives in Pune")
    episode   one dated sentence about what happened in this stretch
              ("Planned a weekend trip to Pune with Priya")

The writer sees the known facts related to what the user said (searched), then the
newest, so it can skip what's already known. It used to see only the 40 newest. LongMemEval-S,
2026-11-17: "how many pre-1920 coins do I have?" needed "37 coins" (said a week earlier) next
to "just added a 1915-S quarter", and with a big memory the 40 newest needn't include it.

Memory only grows: an update is stored as a new, dated fact next to the old one, and
recall shows them oldest first (harness/app.py), so the newest reads as current. Making a
9B writer *replace* facts was tried and dropped (2026-11-17): asked inside this prompt, it
never did; asked as a separate yes/no per fact, it said yes to nearly anything ("two cameras
and a nickel added" made the coin count 39) and overwrote true facts; limited to close
matches and made additive, it was safe but changed nothing in the cases it was built for.

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
KNOWN_FACTS = 40        # facts shown to the writer: related ones first, then the newest
RELATED = 5             # related facts found per user message

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

Known facts (with the day each was learned):
{known}

Conversation:
{conversation}"""


def _transcript(rows) -> str:
    return "\n".join(f"{r['role'].title()}: {r['content'][:MAX_CHARS]}" for r in rows)


def known_facts(store: MemoryStore, rows) -> list[dict]:
    """What the writer sees, so it doesn't repeat them: facts related to what the user said
    here, then the newest, up to KNOWN_FACTS (it used to be only the 40 newest)."""
    said = [r["content"][:MAX_CHARS] for r in rows if r["role"] == "user"]
    known = store.related_facts(said, RELATED)[:KNOWN_FACTS]
    seen = {str(k["id"]) for k in known}
    for f in store.list_facts():
        if len(known) >= KNOWN_FACTS:
            break
        if str(f["id"]) not in seen:
            known.append({"id": f["id"], "text": f["content"], "created_at": f["created_at"]})
    return known


def consolidate(store: MemoryStore, client, model: str, session_id: str,
                notify=None, every: int = EVERY, force: bool = False, at: str | None = None) -> dict | None:
    """Consolidate this session if enough unprocessed exchanges have built up
    (or always, with force=True). Returns what was saved, or None if nothing ran
    or the call failed. Never raises: memory upkeep must not break a turn.
    `at` dates the new facts and episode (when the conversation happened, if
    not now: importing old chats)."""
    rows = store.unconsolidated(session_id)
    if not rows or (not force and len(rows) < every * 2):
        return None
    known = known_facts(store, rows)
    prompt = PROMPT.format(known="\n".join(f"- ({k['created_at'][:10]}) {k['text']}" for k in known) or "(none)",
                           conversation=_transcript(rows))
    try:
        answer = ask_json(client, model, prompt, CONSOLIDATION_FORMAT, notify,
                          name="consolidation")
    except Exception as exc:
        log.warning("consolidation failed, will retry later: %s", exc)
        if notify:
            notify("memory", name="consolidation failed", error=str(exc), pending=len(rows))
        return None

    added = [f.strip() for f in answer["facts"] if f.strip() and store.add_fact(f.strip(), "consolidation", at)]
    episode = answer["episode"].strip()
    if episode:
        store.add_episode(episode, session_id, at)
    store.mark_consolidated(session_id, rows[-1]["id"])
    result = {"messages": len(rows), "facts": added, "episode": episode}
    if notify:
        notify("memory", name="consolidation", **result)
    return result
