"""Facts and episodes in SQLite: the default backend, and the original code.

Facts and episodes are FTS5 tables: SQLite's built-in full-text search. It
matches words (with "meetings" also finding "meeting"), not meanings.

With an Embedder, every fact and episode also gets a vector (stored in the
`vectors` table) and search is hybrid (memory/backends/base.py). On a test of
paraphrased questions, word search found the right fact first 21% of the
time; hybrid found it 93%.

Cosine similarity is computed in numpy over every vector at once. That is
exact and fast for a personal memory (thousands of rows); it is the part
pgvector's HNSW index replaces when memory grows much bigger.
"""

from __future__ import annotations

import logging
import re
import sqlite3

import numpy as np

from memory.backends.base import Candidate, MemoryBackend, content_words, rank

log = logging.getLogger(__name__)

TABLES = {"fact": ("facts", "content"), "episode": ("episodes", "summary")}


def fts_query(text: str) -> str | None:
    """User text as a safe FTS5 query: each word quoted, joined by OR.

    Without the quotes, a message like 'Alex" OR NEAR(' is parsed as query
    syntax and raises. OR (not AND) because recall matters more than precision
    here: the gate already decided this turn needs memory."""
    words = re.findall(r"\w+", text)
    return " OR ".join(f'"{w}"' for w in words) or None


class SQLiteMemory(MemoryBackend):
    name = "sqlite"

    def __init__(self, conn: sqlite3.Connection, embedder=None, min_similarity: float = 0.45):
        super().__init__(embedder, min_similarity)
        self.conn = conn            # shared with MemoryStore, which created the tables

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
        table, column = TABLES[kind]
        return [(r[0], r[1]) for r in self.conn.execute(
            f"SELECT t.rowid, t.{column} FROM {table} t LEFT JOIN vectors v "
            f"ON v.kind = ? AND v.row = t.rowid AND v.model = ? WHERE v.row IS NULL",
            (kind, self.embedder.model_name))]

    def _drop_vectors(self, kind: str, rows: list[int]) -> None:
        self.conn.executemany("DELETE FROM vectors WHERE kind = ? AND row = ?",
                              [(kind, r) for r in rows])

    def _embed_new(self, kind: str, row: int, text: str) -> None:
        if self.embedder is not None:
            try:
                self._embed_rows(kind, [(row, text)])
            except Exception as exc:      # the row is saved; its vector is added on next search
                log.warning("couldn't embed %s now: %s", kind, exc)

    def _hybrid(self, kind: str, query: str, top_k: int) -> list[int] | None:
        """Row ids ranked by meaning plus a word bonus, or None to fall back
        to word search (no embedder, or the model failed to load)."""
        if self.embedder is None:
            return None
        try:
            self._embed_rows(kind, self._missing(kind))
            q = self.embedder.query(query)
        except Exception as exc:
            self._embedder_failed(exc)
            return None
        rows = self.conn.execute("SELECT row, vector FROM vectors WHERE kind = ? AND model = ?",
                                 (kind, self.embedder.model_name)).fetchall()
        if not rows:
            return []
        matrix = np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float32).reshape(len(rows), -1)
        similarity = (matrix @ q).tolist()
        table, _ = TABLES[kind]
        words = content_words(query)
        hits = set()
        if words:
            hits = {r[0] for r in self.conn.execute(
                f"SELECT rowid FROM {table} WHERE {table} MATCH ?",
                (" OR ".join(f'"{w}"' for w in words),))}
        return rank([Candidate(r[0], s, r[0] in hits) for r, s in zip(rows, similarity)],
                    self.min_similarity, top_k)

    def _search(self, kind: str, query: str, top_k: int) -> list[dict]:
        table, column = TABLES[kind]
        ranked = self._hybrid(kind, query, top_k)
        if ranked is None:
            match = fts_query(query)
            if not match:
                return []
            rows = self.conn.execute(
                f"SELECT rowid, {column}, created_at FROM {table} WHERE {table} MATCH ? "
                f"ORDER BY rank LIMIT ?", (match, top_k)).fetchall()
            return [{"id": r[0], "text": r[1], "created_at": r[2]} for r in rows]
        if not ranked:
            return []
        found = {r[0]: r for r in self.conn.execute(
            f"SELECT rowid, {column}, created_at FROM {table} WHERE rowid IN "
            f"({','.join('?' * len(ranked))})", ranked)}
        return [{"id": i, "text": found[i][1], "created_at": found[i][2]} for i in ranked if i in found]

    # ---- facts

    def _has_fact(self, content: str) -> bool:
        return self.conn.execute("SELECT 1 FROM facts WHERE content = ?", (content,)).fetchone() is not None

    def _fact_vectors(self) -> list[tuple[str, object]]:
        self._embed_rows("fact", self._missing("fact"))
        return [(r[0], np.frombuffer(r[1], dtype=np.float32)) for r in self.conn.execute(
            "SELECT f.content, v.vector FROM vectors v JOIN facts f ON f.rowid = v.row "
            "WHERE v.kind = 'fact' AND v.model = ?", (self.embedder.model_name,))]

    def _insert_fact(self, content: str, source: str, created_at: str | None = None) -> None:
        cur = self.conn.execute("INSERT INTO facts VALUES (?, ?, COALESCE(?, datetime('now')))",
                                (content, source, created_at))
        self.conn.commit()
        self._embed_new("fact", cur.lastrowid, content)

    def list_facts(self) -> list[dict]:
        rows = self.conn.execute("SELECT rowid AS id, content, source, created_at FROM facts "
                                 "ORDER BY rowid DESC")
        return [dict(r) for r in rows]

    def _delete_fact(self, fact_id) -> None:
        self.conn.execute("DELETE FROM facts WHERE rowid = ?", (int(fact_id),))
        self._drop_vectors("fact", [int(fact_id)])
        self.conn.commit()

    # ---- episodes

    def _insert_episode(self, summary: str, session_id: str, created_at: str | None = None) -> None:
        cur = self.conn.execute("INSERT INTO episodes VALUES (?, ?, COALESCE(?, datetime('now')))",
                                (summary, session_id, created_at))
        self.conn.commit()
        self._embed_new("episode", cur.lastrowid, summary)

    def list_episodes(self) -> list[dict]:
        rows = self.conn.execute("SELECT rowid AS id, summary, session_id, created_at "
                                 "FROM episodes ORDER BY rowid DESC")
        return [dict(r) for r in rows]

    def _delete_episode(self, episode_id) -> None:
        self.conn.execute("DELETE FROM episodes WHERE rowid = ?", (int(episode_id),))
        self._drop_vectors("episode", [int(episode_id)])
        self.conn.commit()

    def clear(self) -> None:
        self.conn.executescript("DELETE FROM facts; DELETE FROM episodes; DELETE FROM vectors;")
        self.conn.commit()
