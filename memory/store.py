"""Long-term state: one SQLite file the agent reads and writes.

    sessions   separate conversations, like chats in a sidebar
    messages   every exchange, tagged with its session and turn
    facts      durable things about the user ("Alex prefers mornings")
    episodes   dated summaries of what happened in past conversations

Facts and episodes are searched by meaning, and can live elsewhere:
MemoryStore hands them to a semantic backend (memory/backends/). The default
keeps them in this same SQLite file; pgvector and LangGraph's store put them
in Postgres. Sessions and messages always stay here: they are read by
session id, in order, and gain nothing from vectors.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from pathlib import Path

from memory.backends.base import (DUPLICATE, FILLER, MIN_SIMILARITY, RELATED,  # noqa: F401 (re-exported)
                                  WORD_BONUS, MemoryBackend, content_words)
from memory.backends.sqlite import SQLiteMemory, fts_query  # noqa: F401

log = logging.getLogger(__name__)

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
CREATE TABLE IF NOT EXISTS vectors (
    kind TEXT NOT NULL,                -- fact | episode
    row INTEGER NOT NULL,              -- rowid in facts or episodes
    model TEXT NOT NULL,               -- vectors from different models don't compare
    vector BLOB NOT NULL,              -- float32, length-normalised
    PRIMARY KEY (kind, row));
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


class MemoryStore:
    def __init__(self, path: Path, embedder=None, min_similarity: float = MIN_SIMILARITY,
                 backend: MemoryBackend | None = None):
        self.path = path
        self.conn = connect(path, SCHEMA)
        # Added after the first release: older memory.db files get the column here.
        have = {r[1] for r in self.conn.execute("PRAGMA table_info(messages)")}
        for column in ("actions", "visuals"):
            if column not in have:
                self.conn.execute(f"ALTER TABLE messages ADD COLUMN {column} TEXT")
        self.conn.commit()
        # Facts and episodes: SQLite in this same file unless another backend
        # (pgvector, langgraph) is given. See memory/backends/.
        self.semantic: MemoryBackend = backend or SQLiteMemory(self.conn, embedder, min_similarity)

    @property
    def embedder(self):
        return self.semantic.embedder

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
                     turn_id: str | None = None, actions: str | None = None,
                     visuals: list[dict] | None = None) -> None:
        """`actions` is a short note of the tools the reply used. It is kept
        apart from the reply, so the chat shows clean text, and added back
        when the model reads history (see history())."""
        self.conn.executemany(
            "INSERT INTO messages (session_id, role, content, turn_id, actions, visuals) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [(session_id, "user", user_message, turn_id, None, None),
             (session_id, "assistant", reply or "(no reply)", turn_id, actions,
              json.dumps(visuals) if visuals else None)])
        self.conn.execute("UPDATE sessions SET updated_at = datetime('now') WHERE id = ?",
                          (session_id,))
        self.conn.commit()

    def messages(self, session_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT role, content, turn_id, actions, visuals, created_at FROM messages "
            "WHERE session_id = ? ORDER BY id", (session_id,))
        return [{**dict(r), "visuals": json.loads(r["visuals"]) if r["visuals"] else []} for r in rows]

    def history(self, session_id: str, turns: int) -> list[dict]:
        """The last `turns` exchanges, as messages the model reads.

        A bounded window keeps cost and latency flat however long a
        conversation runs. Older context comes back through facts and
        episodes, only when the gate decides it is needed.

        Each assistant message carries a note of the tools it used. Without
        it, the model sees "I saved that" with no evidence, concludes it
        didn't, and saves (or apologises) again: seen in a live session."""
        rows = self.conn.execute(
            "SELECT role, content, actions FROM messages WHERE session_id = ? "
            "ORDER BY id DESC LIMIT ?", (session_id, turns * 2)).fetchall()
        history = [{"role": r["role"],
                    "content": r["content"] + (f"\n\n{r['actions']}" if r["actions"] else "")}
                   for r in reversed(rows)]
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

    # ---- facts and episodes: the semantic backend does the work

    def closest_fact(self, content: str):
        return self.semantic.closest_fact(content)

    def add_fact(self, content: str, source: str = "tool", created_at: str | None = None) -> bool:
        return self.semantic.add_fact(content, source, created_at)

    def search_facts(self, query: str, top_k: int = 5) -> list[str]:
        return self.semantic.search_facts(query, top_k)

    def list_facts(self) -> list[dict]:
        return self.semantic.list_facts()

    def forget(self, text: str) -> int:
        return self.semantic.forget(text)

    def delete_fact(self, fact_id) -> None:
        self.semantic.delete_fact(fact_id)

    def add_episode(self, summary: str, session_id: str, created_at: str | None = None) -> None:
        self.semantic.add_episode(summary, session_id, created_at)

    def search_episodes(self, query: str, top_k: int = 3) -> list[str]:
        return self.semantic.search_episodes(query, top_k)

    def list_episodes(self) -> list[dict]:
        return self.semantic.list_episodes()

    def delete_episode(self, episode_id) -> None:
        self.semantic.delete_episode(episode_id)

    def close(self) -> None:
        if self.semantic is not None and not isinstance(self.semantic, SQLiteMemory):
            self.semantic.close()
