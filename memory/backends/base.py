"""Semantic memory: what every backend must do, and the rules they share.

Semantic memory is the part of memory searched by meaning:

    facts      durable things about the user ("Alex prefers mornings")
    episodes   dated summaries of what happened in past conversations

The conversation log (sessions, messages) is not here: it is read by session
id, in order, and stays in SQLite whatever backend holds the facts.

Where facts live is a choice (memory/backends/):

    sqlite     one file, FTS5 for words, vectors in a table, numpy for cosine
    pgvector   Postgres: tsvector for words, a vector(384) column, an HNSW index
    langgraph  LangGraph's BaseStore (Postgres + pgvector, or in memory), the
               store LangMem reads and writes

The rules are the same everywhere and live in this file, so the backends
behave the same by construction:

  * a fact is stored once: exact text, or a rewording (similarity >= DUPLICATE)
    counts as already stored
  * search is hybrid: cosine similarity plus WORD_BONUS when the text holds a
    real word of the question; a result must clear min_similarity or hold such
    a word, or it isn't returned
  * without an embedder, search falls back to words

A backend supplies storage primitives (insert, delete, list, candidates for a
query); MemoryBackend turns them into the methods the agent calls.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

log = logging.getLogger(__name__)

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


def content_words(text: str) -> list[str]:
    return [w for w in re.findall(r"\w+", text.lower()) if w not in FILLER]


def tidy(text: str) -> str:
    return " ".join(text.split())


@dataclass
class Candidate:
    """One row a backend found for a query: its id, its similarity to the
    question (None when found by words alone) and whether it holds a real
    query word."""
    id: object
    similarity: float | None
    word_hit: bool


def rank(candidates: list[Candidate], min_similarity: float, top_k: int) -> list:
    """The shared hybrid ranking: ids, best first."""
    merged: dict = {}
    for c in candidates:            # a row can be found twice: by meaning and by words
        m = merged.setdefault(c.id, Candidate(c.id, None, False))
        if c.similarity is not None:
            m.similarity = c.similarity if m.similarity is None else max(m.similarity, c.similarity)
        m.word_hit = m.word_hit or c.word_hit
    scored = []
    for c in merged.values():
        s = c.similarity if c.similarity is not None else 0.0
        if (c.similarity is not None and s >= min_similarity) or c.word_hit:
            scored.append((s + (WORD_BONUS if c.word_hit else 0.0), c.id))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [i for _, i in scored[:top_k]]


class MemoryBackend(ABC):
    """Facts and episodes. Subclasses implement the primitives below."""

    name = "base"

    def __init__(self, embedder=None, min_similarity: float = MIN_SIMILARITY):
        self.embedder = embedder
        self.min_similarity = min_similarity
        self.on_change: Callable[[], None] | None = None   # e.g. rewrite MEMORY.md

    # ---- primitives each backend implements

    @abstractmethod
    def _has_fact(self, content: str) -> bool: ...

    def _fact_vectors(self) -> list[tuple[str, object]]:
        """(content, vector) for every fact with a vector from this embedder.
        Used by closest_fact(); a backend can override closest_fact instead."""
        raise NotImplementedError

    @abstractmethod
    def _insert_fact(self, content: str, source: str, created_at: str | None = None) -> None: ...

    @abstractmethod
    def _search(self, kind: str, query: str, top_k: int) -> list[dict]:
        """Rows ranked for `query` (rank() with this backend's candidates, or
        word search without an embedder). Each row has id, text, created_at."""

    @abstractmethod
    def list_facts(self) -> list[dict]:
        """Newest first: id, content, source, created_at ('YYYY-MM-DD HH:MM:SS', UTC)."""

    @abstractmethod
    def _delete_fact(self, fact_id) -> None: ...

    @abstractmethod
    def _insert_episode(self, summary: str, session_id: str, created_at: str | None = None) -> None: ...

    @abstractmethod
    def list_episodes(self) -> list[dict]:
        """Newest first: id, summary, session_id, created_at."""

    @abstractmethod
    def _delete_episode(self, episode_id) -> None: ...

    @abstractmethod
    def clear(self) -> None:
        """Remove every fact and episode in this backend's scope (eval cleanup)."""

    def close(self) -> None:
        pass

    # ---- the methods the agent calls

    def _changed(self) -> None:
        if self.on_change:
            try:
                self.on_change()
            except Exception as exc:          # a mirror must never break a turn
                log.warning("memory change hook failed: %s", exc)

    def _embedder_failed(self, exc: Exception) -> None:
        log.warning("embeddings unavailable, using word search: %s", exc)
        self.embedder = None

    def closest_fact(self, content: str) -> tuple[float, str] | None:
        """The most similar stored fact and its similarity, by meaning.
        None without an embedder or with no facts."""
        if self.embedder is None:
            return None
        try:
            rows = self._fact_vectors()
            vector = self.embedder.documents([content])[0]
        except Exception as exc:
            log.warning("couldn't compare facts by meaning: %s", exc)
            return None
        if not rows:
            return None
        import numpy as np
        matrix = np.asarray([v for _, v in rows], dtype=np.float32)
        similarity = matrix @ vector
        best = int(similarity.argmax())
        return float(similarity[best]), rows[best][0]

    def add_fact(self, content: str, source: str = "tool", created_at: str | None = None) -> bool:
        """Store a fact. False when the same fact is already stored: the same
        text, or (with an embedder) a rewording of it (similarity >= DUPLICATE).
        `created_at` ('YYYY-MM-DD HH:MM:SS') dates it, e.g. when importing old chats; default now."""
        content = tidy(content)
        if not content or self._has_fact(content):
            return False
        closest = self.closest_fact(content)
        if closest and closest[0] >= DUPLICATE:
            return False
        self._insert_fact(content, source, created_at)
        self._changed()
        return True

    def search_facts(self, query: str, top_k: int = 5, dated: bool = False) -> list[str]:
        """`dated` prefixes each fact with the day it was learned: without it an update and the
        fact it replaced look like a contradiction (LongMemEval: 4 of 7 knowledge updates lost)."""
        return [f"({r['created_at'][:10]}) {r['text']}" if dated else r["text"]
                for r in self._search("fact", query, top_k)]

    def forget(self, text: str) -> int:
        """Delete facts containing `text` (case-insensitive). Returns how many."""
        needle = tidy(text).lower()
        if not needle:
            return 0
        ids = [f["id"] for f in self.list_facts() if needle in f["content"].lower()]
        for i in ids:
            self._delete_fact(i)
        if ids:
            self._changed()
        return len(ids)

    def delete_fact(self, fact_id) -> None:
        self._delete_fact(fact_id)
        self._changed()

    def add_episode(self, summary: str, session_id: str, created_at: str | None = None) -> None:
        summary = tidy(summary)
        if summary:
            self._insert_episode(summary, session_id, created_at)
            self._changed()

    def search_episodes(self, query: str, top_k: int = 3) -> list[str]:
        # The date travels with the summary, so the model can say "last Tuesday".
        return [f"({r['created_at'][:10]}) {r['text']}" for r in self._search("episode", query, top_k)]

    def delete_episode(self, episode_id) -> None:
        self._delete_episode(episode_id)
        self._changed()

    def import_from(self, other: "MemoryBackend") -> dict:
        """Copy another backend's facts and episodes here, oldest first, keeping
        their dates. Facts already here (same text) are skipped, so running it
        twice copies nothing new."""
        facts = episodes = 0
        for f in reversed(other.list_facts()):
            if not self._has_fact(f["content"]):
                self._insert_fact(f["content"], f["source"], f["created_at"])
                facts += 1
        have = {(e["summary"], e["session_id"]) for e in self.list_episodes()}
        for e in reversed(other.list_episodes()):
            if (e["summary"], e["session_id"]) not in have:
                self._insert_episode(e["summary"], e["session_id"], e["created_at"])
                episodes += 1
        self._changed()
        return {"facts": facts, "episodes": episodes}
