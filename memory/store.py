"""Long-term state: one SQLite file the agent reads and writes.

    sessions   separate conversations, like chats in a sidebar
    messages   every exchange, tagged with its session and turn
    facts      durable things about the user ("Alex prefers mornings")
    episodes   dated summaries of what happened in past conversations

Facts and episodes are FTS5 tables: SQLite's built-in full-text search. It
matches words (with "meetings" also finding "meeting"), not meanings. That is
the deliberate starting point: zero setup, fast, and you can read every row.
Swapping in embeddings later changes only search_facts and search_episodes.
"""

from __future__ import annotations

import re
import sqlite3
import uuid
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')));
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    turn_id TEXT,
    consolidated INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')));
CREATE INDEX IF NOT EXISTS messages_by_session ON messages(session_id, id);
CREATE VIRTUAL TABLE IF NOT EXISTS facts USING fts5(
    content, source UNINDEXED, created_at UNINDEXED, tokenize='porter');
CREATE VIRTUAL TABLE IF NOT EXISTS episodes USING fts5(
    summary, session_id UNINDEXED, created_at UNINDEXED, tokenize='porter');
"""

TITLE_CHARS = 60


def connect(path: Path, schema: str) -> sqlite3.Connection:
    """One connection shared by the agent's thread. WAL lets the dashboard read
    on its own connections while a turn is writing."""
    conn = sqlite3.connect(path, check_same_thread=False, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(schema)
    return conn


def fts_query(text: str) -> str | None:
    """User text as a safe FTS5 query: each word quoted, joined by OR.

    Without the quotes, a message like 'Alex" OR NEAR(' is parsed as query
    syntax and raises. OR (not AND) because recall matters more than precision
    here: the gate already decided this turn needs memory."""
    words = re.findall(r"\w+", text)
    return " OR ".join(f'"{w}"' for w in words) or None


class MemoryStore:
    def __init__(self, path: Path):
        self.path = path
        self.conn = connect(path, SCHEMA)

    # ---- sessions

    def ensure_session(self, session_id: str | None, first_message: str = "") -> str:
        """The session to use: the one asked for if it exists, else a new one
        titled after the message that started it."""
        if session_id and self.conn.execute(
                "SELECT 1 FROM sessions WHERE id = ?", (session_id,)).fetchone():
            return session_id
        new_id = session_id or uuid.uuid4().hex[:12]
        title = " ".join(first_message.split())[:TITLE_CHARS] or "New conversation"
        self.conn.execute("INSERT INTO sessions (id, title) VALUES (?, ?)", (new_id, title))
        self.conn.commit()
        return new_id

    def list_sessions(self, limit: int = 50) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id, title, created_at, updated_at FROM sessions "
            "ORDER BY updated_at DESC, rowid DESC LIMIT ?", (limit,))
        return [dict(r) for r in rows]

    # ---- messages

    def add_exchange(self, session_id: str, user_message: str, reply: str,
                     turn_id: str | None = None) -> None:
        self.conn.executemany(
            "INSERT INTO messages (session_id, role, content, turn_id) VALUES (?, ?, ?, ?)",
            [(session_id, "user", user_message, turn_id),
             (session_id, "assistant", reply or "(no reply)", turn_id)])
        self.conn.execute("UPDATE sessions SET updated_at = datetime('now') WHERE id = ?",
                          (session_id,))
        self.conn.commit()

    def messages(self, session_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT role, content, turn_id, created_at FROM messages "
            "WHERE session_id = ? ORDER BY id", (session_id,))
        return [dict(r) for r in rows]

    def history(self, session_id: str, turns: int) -> list[dict]:
        """The last `turns` exchanges, as messages the model reads.

        A bounded window keeps cost and latency flat however long a
        conversation runs. Older context comes back through facts and
        episodes, only when the gate decides it is needed."""
        rows = self.conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? "
            "ORDER BY id DESC LIMIT ?", (session_id, turns * 2)).fetchall()
        history = [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]
        while history and history[0]["role"] != "user":
            history.pop(0)   # the API expects a conversation to start with the user
        return history

    def last_exchange(self, session_id: str) -> dict | None:
        rows = self.conn.execute(
            "SELECT role, content, turn_id FROM messages WHERE session_id = ? "
            "ORDER BY id DESC LIMIT 2", (session_id,)).fetchall()
        if len(rows) < 2:
            return None
        reply, user = rows
        return {"user": user["content"], "reply": reply["content"], "turn_id": reply["turn_id"]}

    def unconsolidated(self, session_id: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT id, role, content FROM messages "
            "WHERE session_id = ? AND consolidated = 0 ORDER BY id", (session_id,)).fetchall()

    def mark_consolidated(self, session_id: str, up_to_id: int) -> None:
        self.conn.execute("UPDATE messages SET consolidated = 1 WHERE session_id = ? AND id <= ?",
                          (session_id, up_to_id))
        self.conn.commit()

    # ---- facts: what is durably true

    def add_fact(self, content: str, source: str = "tool") -> bool:
        """Store a fact. False when the same text is already stored."""
        content = " ".join(content.split())
        if not content or self.conn.execute(
                "SELECT 1 FROM facts WHERE content = ?", (content,)).fetchone():
            return False
        self.conn.execute("INSERT INTO facts VALUES (?, ?, datetime('now'))", (content, source))
        self.conn.commit()
        return True

    def search_facts(self, query: str, top_k: int = 5) -> list[str]:
        match = fts_query(query)
        if not match:
            return []
        rows = self.conn.execute(
            "SELECT content FROM facts WHERE facts MATCH ? ORDER BY rank LIMIT ?", (match, top_k))
        return [r["content"] for r in rows]

    def list_facts(self) -> list[dict]:
        rows = self.conn.execute("SELECT rowid AS id, content, source, created_at FROM facts "
                                 "ORDER BY rowid DESC")
        return [dict(r) for r in rows]

    def forget(self, text: str) -> int:
        """Delete facts containing `text` (case-insensitive). Returns how many."""
        needle = " ".join(text.split()).lower()
        if not needle:
            return 0
        ids = [r["rowid"] for r in self.conn.execute("SELECT rowid, content FROM facts")
               if needle in r["content"].lower()]
        self.conn.executemany("DELETE FROM facts WHERE rowid = ?", [(i,) for i in ids])
        self.conn.commit()
        return len(ids)

    def delete_fact(self, fact_id: int) -> None:
        self.conn.execute("DELETE FROM facts WHERE rowid = ?", (fact_id,))
        self.conn.commit()

    # ---- episodes: what happened, and when

    def add_episode(self, summary: str, session_id: str) -> None:
        summary = " ".join(summary.split())
        if summary:
            self.conn.execute("INSERT INTO episodes VALUES (?, ?, datetime('now'))",
                              (summary, session_id))
            self.conn.commit()

    def search_episodes(self, query: str, top_k: int = 3) -> list[str]:
        match = fts_query(query)
        if not match:
            return []
        rows = self.conn.execute(
            "SELECT summary, created_at FROM episodes WHERE episodes MATCH ? "
            "ORDER BY rank LIMIT ?", (match, top_k))
        # The date travels with the summary, so the model can say "last Tuesday".
        return [f"({r['created_at'][:10]}) {r['summary']}" for r in rows]

    def list_episodes(self) -> list[dict]:
        rows = self.conn.execute("SELECT rowid AS id, summary, session_id, created_at "
                                 "FROM episodes ORDER BY rowid DESC")
        return [dict(r) for r in rows]

    def delete_episode(self, episode_id: int) -> None:
        self.conn.execute("DELETE FROM episodes WHERE rowid = ?", (episode_id,))
        self.conn.commit()
