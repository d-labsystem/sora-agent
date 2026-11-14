"""Long-term state: one SQLite file the agent reads and writes.

    sessions   separate conversations, like chats in a sidebar
    messages   every exchange, tagged with its session and turn
    facts      durable things about the user ("Alex prefers mornings")
    episodes   dated summaries of what happened in past conversations

Facts and episodes are FTS5 tables: SQLite's built-in full-text search. It
matches words (with "meetings" also finding "meeting"), not meanings.

With an Embedder, search is hybrid. Every fact and episode also gets a
vector (memory/embeddings.py, stored in the `vectors` table), and results
are ranked by meaning: cosine similarity to the question, plus a small bonus
when the text also contains one of the question's real words, which keeps
exact names, IDs and rare terms safe. On a test of paraphrased questions,
word search found the right fact first 21% of the time; this found it 93%.

Meaning search always returns *something*, so results must clear a minimum
similarity (or contain a real query word) to count; otherwise every question
would drag loosely related facts into the prompt.

Without an Embedder (or if the model can't load), search falls back to words.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import uuid
from pathlib import Path

import numpy as np

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

# Words too common to say anything about which fact is meant. Used only for
# the exact-word bonus in hybrid search, not for word-only search.
FILLER = set("""a an the is are was were be been am i me my mine you your yours we our us he she it
its they them their what which who whom when where why how do does did can could should would
will shall to of in on at for with about from by as and or but if then so than that this these
those there here not no yes any some""".split())
MIN_SIMILARITY = 0.45   # below this, a meaning match is treated as unrelated
# Measured on pairs from a live session: rewordings of the same fact scored
# 0.957-0.989, updates (3 Dec -> 5 Dec, Pune -> Mumbai) 0.894-0.921, different
# facts up to 0.892. So >= 0.95 is a duplicate (not stored again), and
# 0.88-0.95 is "related": stored, with a warning so the model can decide
# whether it replaces the old one. Updates must never be silently dropped.
DUPLICATE = 0.95
RELATED = 0.88
WORD_BONUS = 0.05       # added when the text also contains a real query word

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


def content_words(text: str) -> list[str]:
    return [w for w in re.findall(r"\w+", text.lower()) if w not in FILLER]


class MemoryStore:
    def __init__(self, path: Path, embedder=None, min_similarity: float = MIN_SIMILARITY):
        self.path = path
        self.conn = connect(path, SCHEMA)
        # Added after the first release: older memory.db files get the column here.
        have = {r[1] for r in self.conn.execute("PRAGMA table_info(messages)")}
        for column in ("actions", "visuals"):
            if column not in have:
                self.conn.execute(f"ALTER TABLE messages ADD COLUMN {column} TEXT")
        self.conn.commit()
        self.embedder = embedder
        self.min_similarity = min_similarity

    # ---- vectors (only with an embedder)

    def _embed_rows(self, kind: str, rows: list[tuple[int, str]]) -> None:
        if not rows:
            return
        vectors = self.embedder.documents([text for _, text in rows])
        self.conn.executemany(
            "INSERT OR REPLACE INTO vectors (kind, row, model, vector) VALUES (?, ?, ?, ?)",
            [(kind, row, self.embedder.model_name, v.astype(np.float32).tobytes())
             for (row, _), v in zip(rows, vectors)])
        self.conn.commit()

    def _missing(self, kind: str) -> list[tuple[int, str]]:
        """Rows with no vector from the current model: added before embeddings
        were on, or embedded by a different model."""
        table, column = ("facts", "content") if kind == "fact" else ("episodes", "summary")
        return [(r[0], r[1]) for r in self.conn.execute(
            f"SELECT t.rowid, t.{column} FROM {table} t LEFT JOIN vectors v "
            f"ON v.kind = ? AND v.row = t.rowid AND v.model = ? WHERE v.row IS NULL",
            (kind, self.embedder.model_name))]

    def _drop_vectors(self, kind: str, rows: list[int]) -> None:
        self.conn.executemany("DELETE FROM vectors WHERE kind = ? AND row = ?",
                              [(kind, r) for r in rows])

    def _hybrid(self, kind: str, query: str, top_k: int) -> list[int] | None:
        """Row ids ranked by meaning plus a word bonus, or None to fall back
        to word search (no embedder, or the model failed to load)."""
        if self.embedder is None:
            return None
        try:
            self._embed_rows(kind, self._missing(kind))
            q = self.embedder.query(query)
        except Exception as exc:
            log.warning("embeddings unavailable, using word search: %s", exc)
            self.embedder = None
            return None
        rows = self.conn.execute("SELECT row, vector FROM vectors WHERE kind = ? AND model = ?",
                                 (kind, self.embedder.model_name)).fetchall()
        if not rows:
            return []
        ids = [r[0] for r in rows]
        matrix = np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float32).reshape(len(rows), -1)
        similarity = matrix @ q
        table = "facts" if kind == "fact" else "episodes"
        words = content_words(query)
        hits = set()
        if words:
            hits = {r[0] for r in self.conn.execute(
                f"SELECT rowid FROM {table} WHERE {table} MATCH ?",
                (" OR ".join(f'"{w}"' for w in words),))}
        scored = [(s + (WORD_BONUS if i in hits else 0.0), i)
                  for i, s in zip(ids, similarity.tolist())
                  if s >= self.min_similarity or i in hits]
        scored.sort(reverse=True)
        return [i for _, i in scored[:top_k]]

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

    # ---- facts: what is durably true

    def closest_fact(self, content: str) -> tuple[float, str] | None:
        """The most similar stored fact and its similarity, by meaning.
        None without an embedder or with no facts."""
        if self.embedder is None:
            return None
        try:
            self._embed_rows("fact", self._missing("fact"))
            vector = self.embedder.documents([content])[0]
        except Exception as exc:
            log.warning("couldn't compare facts by meaning: %s", exc)
            return None
        rows = self.conn.execute(
            "SELECT f.content, v.vector FROM vectors v JOIN facts f ON f.rowid = v.row "
            "WHERE v.kind = 'fact' AND v.model = ?", (self.embedder.model_name,)).fetchall()
        if not rows:
            return None
        matrix = np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float32).reshape(len(rows), -1)
        similarity = matrix @ vector
        best = int(similarity.argmax())
        return float(similarity[best]), rows[best][0]

    def add_fact(self, content: str, source: str = "tool") -> bool:
        """Store a fact. False when the same fact is already stored: the same
        text, or (with an embedder) a rewording of it (similarity >= DUPLICATE)."""
        content = " ".join(content.split())
        if not content or self.conn.execute(
                "SELECT 1 FROM facts WHERE content = ?", (content,)).fetchone():
            return False
        closest = self.closest_fact(content)
        if closest and closest[0] >= DUPLICATE:
            return False
        cur = self.conn.execute("INSERT INTO facts VALUES (?, ?, datetime('now'))", (content, source))
        self.conn.commit()
        if self.embedder is not None:
            try:
                self._embed_rows("fact", [(cur.lastrowid, content)])
            except Exception as exc:      # the fact is saved; its vector is added on next search
                log.warning("couldn't embed fact now: %s", exc)
        return True

    def search_facts(self, query: str, top_k: int = 5) -> list[str]:
        ranked = self._hybrid("fact", query, top_k)
        if ranked is not None:
            text = {r[0]: r[1] for r in self.conn.execute(
                f"SELECT rowid, content FROM facts WHERE rowid IN ({','.join('?' * len(ranked))})",
                ranked)} if ranked else {}
            return [text[i] for i in ranked if i in text]
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
        self._drop_vectors("fact", ids)
        self.conn.commit()
        return len(ids)

    def delete_fact(self, fact_id: int) -> None:
        self.conn.execute("DELETE FROM facts WHERE rowid = ?", (fact_id,))
        self._drop_vectors("fact", [fact_id])
        self.conn.commit()

    # ---- episodes: what happened, and when

    def add_episode(self, summary: str, session_id: str) -> None:
        summary = " ".join(summary.split())
        if summary:
            cur = self.conn.execute("INSERT INTO episodes VALUES (?, ?, datetime('now'))",
                                    (summary, session_id))
            self.conn.commit()
            if self.embedder is not None:
                try:
                    self._embed_rows("episode", [(cur.lastrowid, summary)])
                except Exception as exc:
                    log.warning("couldn't embed episode now: %s", exc)

    def search_episodes(self, query: str, top_k: int = 3) -> list[str]:
        ranked = self._hybrid("episode", query, top_k)
        if ranked is not None:
            found = {r[0]: r for r in self.conn.execute(
                f"SELECT rowid, summary, created_at FROM episodes WHERE rowid IN "
                f"({','.join('?' * len(ranked))})", ranked)} if ranked else {}
            rows = [found[i] for i in ranked if i in found]
        else:
            match = fts_query(query)
            if not match:
                return []
            rows = self.conn.execute(
                "SELECT summary, created_at FROM episodes WHERE episodes MATCH ? "
                "ORDER BY rank LIMIT ?", (match, top_k)).fetchall()
        # The date travels with the summary, so the model can say "last Tuesday".
        return [f"({r['created_at'][:10]}) {r['summary']}" for r in rows]

    def list_episodes(self) -> list[dict]:
        rows = self.conn.execute("SELECT rowid AS id, summary, session_id, created_at "
                                 "FROM episodes ORDER BY rowid DESC")
        return [dict(r) for r in rows]

    def delete_episode(self, episode_id: int) -> None:
        self.conn.execute("DELETE FROM episodes WHERE rowid = ?", (episode_id,))
        self._drop_vectors("episode", [episode_id])
        self.conn.commit()
