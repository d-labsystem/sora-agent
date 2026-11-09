"""Self-learning: the agent writes down lessons from its own mistakes.

No weights change. The model stays the same; what improves is the context
we give it. This is "learning in context", the same idea as a junior engineer
keeping a notebook of gotchas. The loop has four steps:

    1. detect    did this turn go wrong? (signals below)
    2. reflect   the small model reads the turn and writes 0-3 lessons
    3. store     lessons go into an FTS table, tagged with tool and source turn
    4. reuse     next turns get relevant lessons in the system prompt, and
                 tool-specific lessons appended to that tool's description

Signals, from cheapest to most trustworthy:

    tool_error        a tool returned is_error (wrong args, bad path, ...)
    iteration_limit   the loop ran out of steps: it was going in circles
    user_correction   the gate saw "no, that's wrong" about the last reply
    thumbs_down       the user pressed 👎 in the dashboard

Turns with no signal are not reflected on: reflection costs money, and
"everything went fine" lessons are noise.
"""

from __future__ import annotations

import logging
from pathlib import Path

from loop.llm import ask_json, schema
from memory.store import connect, fts_query

log = logging.getLogger(__name__)

SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS lessons USING fts5(
    situation, lesson, tool UNINDEXED, signal UNINDEXED, source_turn UNINDEXED,
    uses UNINDEXED, created_at UNINDEXED, tokenize='porter');
"""

SIGNALS = ("tool_error", "iteration_limit", "user_correction", "thumbs_down")
MAX_LESSONS = 3
NOTES_PER_TOOL = 3
TRACE_CHARS = 1500

REFLECTION_FORMAT = schema({
    "lessons": {"type": "array", "items": schema({
        "situation": {"type": "string"},
        "lesson": {"type": "string"},
        "tool": {"type": "string"},
    })["schema"]},
})

PROMPT = """You review a turn where a personal assistant went wrong, and write lessons
that would prevent the same mistake next time.

What went wrong: {signals}
{note}
Each lesson has:
situation: when it applies, in words a future request would contain.
lesson: one concrete instruction, e.g. "Pass dates to calendar__create as
YYYY-MM-DD" (not vague advice like "be careful").
tool: the exact tool name if the lesson is about using one tool, else "".

Write at most {max} lessons, only for mistakes you can see in the turn.
Do not repeat a known lesson. Return [] if the cause is unclear or external
(an outage, a site that was down).

Known lessons:
{known}

{previous}User message:
{message}

Tool calls:
{calls}

Assistant reply:
{reply}"""


def detect(tool_errors: int = 0, limit_reached: bool = False,
           correction: bool = False, thumbs_down: bool = False) -> list[str]:
    """Which learning signals fired for a turn. Empty means: don't reflect."""
    fired = [tool_errors > 0, limit_reached, correction, thumbs_down]
    return [name for name, on in zip(SIGNALS, fired) if on]


def _calls(tool_calls: list[dict]) -> str:
    lines = []
    for c in tool_calls:
        status = "ERROR" if c.get("is_error") else "ok"
        lines.append(f"- {c['tool']}({c.get('args')}) -> {status}: {str(c.get('output'))[:TRACE_CHARS]}")
    return "\n".join(lines) or "(none)"


class LessonStore:
    def __init__(self, path: Path):
        self.conn = connect(path, SCHEMA)

    def add(self, situation: str, lesson: str, tool: str = "", signal: str = "",
            source_turn: str | None = None) -> bool:
        lesson = " ".join(lesson.split())
        if not lesson or self.conn.execute(
                "SELECT 1 FROM lessons WHERE lower(lesson) = lower(?)", (lesson,)).fetchone():
            return False
        self.conn.execute("INSERT INTO lessons VALUES (?, ?, ?, ?, ?, 0, datetime('now'))",
                          (" ".join(situation.split()), lesson, tool.strip(), signal, source_turn))
        self.conn.commit()
        return True

    def relevant(self, query: str, top_k: int = 3) -> list[str]:
        """General lessons whose situation or text matches the request.
        Counting uses tells us later which lessons actually get applied."""
        match = fts_query(query)
        if not match:
            return []
        rows = self.conn.execute(
            "SELECT rowid, situation, lesson FROM lessons WHERE lessons MATCH ? AND tool = '' "
            "ORDER BY rank LIMIT ?", (match, top_k)).fetchall()
        self._used([r["rowid"] for r in rows])
        return [f"When {r['situation']}: {r['lesson']}" for r in rows]

    def tool_notes(self, tool_names: list[str]) -> dict[str, list[str]]:
        """Newest lessons per tool, to append to that tool's description."""
        notes: dict[str, list[str]] = {}
        for name in tool_names:
            rows = self.conn.execute(
                "SELECT rowid, lesson FROM lessons WHERE tool = ? ORDER BY rowid DESC LIMIT ?",
                (name, NOTES_PER_TOOL)).fetchall()
            if rows:
                notes[name] = [r["lesson"] for r in rows]
        return notes

    def _used(self, ids: list[int]) -> None:
        self.conn.executemany("UPDATE lessons SET uses = uses + 1 WHERE rowid = ?",
                              [(i,) for i in ids])
        self.conn.commit()

    def list(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT rowid AS id, situation, lesson, tool, signal, source_turn, uses, created_at "
            "FROM lessons ORDER BY rowid DESC")
        return [dict(r) for r in rows]

    def delete(self, lesson_id: int) -> None:
        self.conn.execute("DELETE FROM lessons WHERE rowid = ?", (lesson_id,))
        self.conn.commit()


def reflect(lessons: LessonStore, client, model: str, signals: list[str], message: str,
            reply: str, tool_calls: list[dict], turn_id: str | None = None,
            previous: dict | None = None, note: str = "", notify=None) -> list[dict]:
    """Ask the small model what to learn from a turn that went wrong.
    Returns the lessons actually saved. Never raises."""
    if not signals:
        return []
    known = [l["lesson"] for l in lessons.list()[:30]]
    prompt = PROMPT.format(
        signals=", ".join(signals),
        note=f"User feedback: {note}\n" if note else "",
        max=MAX_LESSONS,
        known="\n".join(f"- {k}" for k in known) or "(none)",
        previous=(f"Previous exchange (the user corrected this reply):\nUser: {previous['user'][:TRACE_CHARS]}\n"
                  f"Assistant: {previous['reply'][:TRACE_CHARS]}\n\n") if previous else "",
        message=message[:TRACE_CHARS], calls=_calls(tool_calls), reply=reply[:TRACE_CHARS])
    try:
        answer = ask_json(client, model, prompt, REFLECTION_FORMAT, notify, name="reflection")
    except Exception as exc:
        log.warning("reflection failed: %s", exc)
        if notify:
            notify("learning", name="reflection failed", signals=signals, error=str(exc))
        return []
    saved = [l for l in answer["lessons"][:MAX_LESSONS]
             if lessons.add(l["situation"], l["lesson"], l["tool"], ",".join(signals), turn_id)]
    if notify:
        notify("learning", name="reflection", signals=signals, lessons=saved)
    return saved
