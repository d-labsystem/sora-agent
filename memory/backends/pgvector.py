"""Facts and episodes in Postgres with pgvector.

Same rules as SQLite (memory/backends/base.py), different machinery:

    words      a generated tsvector column (Postgres full-text search, English
               stemming) with a GIN index; websearch_to_tsquery parses the
               question safely, so no user text is ever read as query syntax
    meaning    a vector(384) column with an HNSW index on cosine distance
               (`<=>`). HNSW is approximate: it walks a graph of neighbours
               instead of comparing the question with every row, which is what
               keeps search fast at a million rows, where numpy's
               compare-everything approach would not

Search asks Postgres for two candidate lists, the nearest rows by meaning and
the rows holding a real query word, then ranks them with the shared rank().

Every row carries a `scope`: one per sora home folder. Your ~/.sora and an
eval case's temporary home can share one database without seeing each
other's facts, and clear() removes one scope's rows only.

Needs: `uv sync --extra memory-pg` and Postgres with the vector extension
(`docker run -d --name sora-pg -p 5433:5432 -e POSTGRES_PASSWORD=sora pgvector/pgvector:pg17`).
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

from memory.backends.base import Candidate, MemoryBackend, content_words, rank

log = logging.getLogger(__name__)

DIMS = 384              # BAAI/bge-small-en-v1.5; the column type fixes it
CANDIDATES = 50         # nearest rows fetched per search, before ranking

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS sora_facts (
    id BIGSERIAL PRIMARY KEY,
    scope TEXT NOT NULL,
    content TEXT NOT NULL,
    source TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    model TEXT,
    embedding vector({DIMS}),
    words tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED);
CREATE TABLE IF NOT EXISTS sora_episodes (
    id BIGSERIAL PRIMARY KEY,
    scope TEXT NOT NULL,
    summary TEXT NOT NULL,
    session_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    model TEXT,
    embedding vector({DIMS}),
    words tsvector GENERATED ALWAYS AS (to_tsvector('english', summary)) STORED);
CREATE INDEX IF NOT EXISTS sora_facts_scope ON sora_facts (scope);
CREATE INDEX IF NOT EXISTS sora_episodes_scope ON sora_episodes (scope);
CREATE INDEX IF NOT EXISTS sora_facts_words ON sora_facts USING gin (words);
CREATE INDEX IF NOT EXISTS sora_episodes_words ON sora_episodes USING gin (words);
CREATE INDEX IF NOT EXISTS sora_facts_embedding ON sora_facts USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS sora_episodes_embedding ON sora_episodes USING hnsw (embedding vector_cosine_ops);
"""

TABLES = {"fact": ("sora_facts", "content"), "episode": ("sora_episodes", "summary")}


def as_vector(v) -> str:
    """pgvector's text form, '[0.1,0.2,...]', cast with ::vector in SQL. This
    avoids needing the pgvector Python package just to send a list."""
    return "[" + ",".join(f"{x:.7g}" for x in v) + "]"


def from_vector(text: str):
    import numpy as np
    return np.asarray([float(x) for x in text.strip("[]").split(",")], dtype=np.float32)


def stamp(value) -> str:
    """Timestamps as SQLite writes them: 'YYYY-MM-DD HH:MM:SS' in UTC."""
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    return str(value)


class PgVectorMemory(MemoryBackend):
    name = "pgvector"

    def __init__(self, url: str, scope: str, embedder=None, min_similarity: float = 0.45, conn=None):
        super().__init__(embedder, min_similarity)
        import psycopg                      # optional dependency: imported only when chosen
        from psycopg.rows import dict_row
        self.scope = scope
        self.conn = conn or psycopg.connect(url, autocommit=True, row_factory=dict_row)
        self.conn.execute(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def _sql(self, query: str, params=()) -> list[dict]:
        cur = self.conn.execute(query, params)
        return cur.fetchall() if cur.description else []

    # ---- vectors

    def _vector(self, text: str) -> str | None:
        vector = self.embedder.documents([text])[0]
        if len(vector) != DIMS:
            raise ValueError(f"{self.embedder.model_name} makes {len(vector)}-d vectors; "
                             f"the table holds {DIMS}")
        return as_vector(vector)

    def _embed_missing(self, kind: str) -> None:
        """Rows without a vector from the current model: added while
        embeddings were off, or copied in by `memory migrate`."""
        table, column = TABLES[kind]
        rows = self._sql(f"SELECT id, {column} AS text FROM {table} WHERE scope = %s "
                         f"AND (model IS DISTINCT FROM %s OR embedding IS NULL)",
                         (self.scope, self.embedder.model_name))
        if not rows:
            return
        vectors = self.embedder.documents([r["text"] for r in rows])
        with self.conn.transaction():
            for r, v in zip(rows, vectors):
                self.conn.execute(f"UPDATE {table} SET embedding = %s::vector, model = %s WHERE id = %s",
                                  (as_vector(v), self.embedder.model_name, r["id"]))

    def _embed_new(self, kind: str, row_id: int, text: str) -> None:
        if self.embedder is None:
            return
        try:
            table, _ = TABLES[kind]
            self.conn.execute(f"UPDATE {table} SET embedding = %s::vector, model = %s WHERE id = %s",
                              (self._vector(text), self.embedder.model_name, row_id))
        except Exception as exc:          # saved; the vector is added on the next search
            log.warning("couldn't embed %s now: %s", kind, exc)

    # ---- search

    def _search(self, kind: str, query: str, top_k: int) -> list[dict]:
        table, column = TABLES[kind]
        words = " or ".join(content_words(query))
        if self.embedder is not None:
            try:
                self._embed_missing(kind)
                q = as_vector(self.embedder.query(query))
            except Exception as exc:
                self._embedder_failed(exc)
        if self.embedder is None:          # words only, ranked by Postgres
            all_words = " or ".join(re.findall(r"\w+", query))
            if not all_words:
                return []
            rows = self._sql(
                f"SELECT id, {column} AS text, created_at FROM {table} "
                f"WHERE scope = %s AND words @@ websearch_to_tsquery('english', %s) "
                f"ORDER BY ts_rank(words, websearch_to_tsquery('english', %s)) DESC, id DESC LIMIT %s",
                (self.scope, all_words, all_words, top_k))
            return [{**r, "created_at": stamp(r["created_at"])} for r in rows]

        hit, hit_params = (("words @@ websearch_to_tsquery('english', %s)", (words,)) if words
                           else ("FALSE", ()))
        near = self._sql(
            f"SELECT id, {column} AS text, created_at, 1 - (embedding <=> %s::vector) AS similarity, "
            f"{hit} AS hit FROM {table} WHERE scope = %s AND model = %s "
            f"ORDER BY embedding <=> %s::vector LIMIT %s",
            (q, *hit_params, self.scope, self.embedder.model_name, q, CANDIDATES))
        by_word = self._sql(
            f"SELECT id, {column} AS text, created_at, 1 - (embedding <=> %s::vector) AS similarity, "
            f"TRUE AS hit FROM {table} WHERE scope = %s AND model = %s "
            f"AND words @@ websearch_to_tsquery('english', %s) LIMIT %s",
            (q, self.scope, self.embedder.model_name, words, CANDIDATES)) if words else []
        rows = {r["id"]: r for r in near + by_word}
        ranked = rank([Candidate(r["id"], float(r["similarity"]), bool(r["hit"])) for r in near + by_word],
                      self.min_similarity, top_k)
        return [{"id": i, "text": rows[i]["text"], "created_at": stamp(rows[i]["created_at"])}
                for i in ranked]

    # ---- facts

    def _has_fact(self, content: str) -> bool:
        return bool(self._sql("SELECT 1 FROM sora_facts WHERE scope = %s AND content = %s",
                              (self.scope, content)))

    def _fact_vectors(self) -> list[tuple[str, object]]:
        self._embed_missing("fact")
        return [(r["content"], from_vector(r["embedding"])) for r in self._sql(
            "SELECT content, embedding::text AS embedding FROM sora_facts "
            "WHERE scope = %s AND model = %s", (self.scope, self.embedder.model_name))]

    def closest_fact(self, content: str) -> tuple[float, str] | None:
        """Here the database finds the nearest fact (the HNSW index), instead
        of every vector coming back to Python as in the base class."""
        if self.embedder is None:
            return None
        try:
            self._embed_missing("fact")
            v = self._vector(content)
        except Exception as exc:
            log.warning("couldn't compare facts by meaning: %s", exc)
            return None
        rows = self._sql("SELECT content, 1 - (embedding <=> %s::vector) AS similarity FROM sora_facts "
                         "WHERE scope = %s AND model = %s ORDER BY embedding <=> %s::vector LIMIT 1",
                         (v, self.scope, self.embedder.model_name, v))
        return (float(rows[0]["similarity"]), rows[0]["content"]) if rows else None

    def _insert_fact(self, content: str, source: str, created_at: str | None = None) -> None:
        row = self._sql("INSERT INTO sora_facts (scope, content, source, created_at) "
                        "VALUES (%s, %s, %s, COALESCE(%s::timestamptz, now())) RETURNING id",
                        (self.scope, content, source, created_at and f"{created_at}+00"))[0]
        self._embed_new("fact", row["id"], content)

    def list_facts(self) -> list[dict]:
        return [{**r, "created_at": stamp(r["created_at"])} for r in self._sql(
            "SELECT id, content, source, created_at FROM sora_facts WHERE scope = %s ORDER BY id DESC",
            (self.scope,))]

    def _delete_fact(self, fact_id) -> None:
        self.conn.execute("DELETE FROM sora_facts WHERE scope = %s AND id = %s", (self.scope, int(fact_id)))

    # ---- episodes

    def _insert_episode(self, summary: str, session_id: str, created_at: str | None = None) -> None:
        row = self._sql("INSERT INTO sora_episodes (scope, summary, session_id, created_at) "
                        "VALUES (%s, %s, %s, COALESCE(%s::timestamptz, now())) RETURNING id",
                        (self.scope, summary, session_id, created_at and f"{created_at}+00"))[0]
        self._embed_new("episode", row["id"], summary)

    def list_episodes(self) -> list[dict]:
        return [{**r, "created_at": stamp(r["created_at"])} for r in self._sql(
            "SELECT id, summary, session_id, created_at FROM sora_episodes WHERE scope = %s "
            "ORDER BY id DESC", (self.scope,))]

    def _delete_episode(self, episode_id) -> None:
        self.conn.execute("DELETE FROM sora_episodes WHERE scope = %s AND id = %s",
                          (self.scope, int(episode_id)))

    def clear(self) -> None:
        with self.conn.transaction():
            self.conn.execute("DELETE FROM sora_facts WHERE scope = %s", (self.scope,))
            self.conn.execute("DELETE FROM sora_episodes WHERE scope = %s", (self.scope,))
