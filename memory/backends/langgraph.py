"""Facts and episodes in a LangGraph store: the store LangMem works on.

LangGraph's BaseStore is a key-value store of JSON documents grouped into
namespaces (tuples like ("sora", <scope>, "facts")), with optional vector
search over chosen fields. Two implementations matter here:

    PostgresStore   Postgres + pgvector (langgraph-checkpoint-postgres)
    InMemoryStore   a dict in the process: what the offline tests use

Why a second Postgres backend when pgvector.py exists? Because this is the
format LangMem reads and writes. A fact is stored the way LangMem stores one,
{"kind": "Memory", "content": {"content": "..."}}, so sora's own tools and
LangMem's manager (memory/langmem_consolidator.py) edit the same memories.

Differences from the other backends, kept small on purpose:
  * keys are UUID strings, not numbers
  * words are matched in Python over the scope's items (no full-text index),
    fine for a personal memory of thousands, not for millions
  * the store embeds documents and queries itself, through SoraEmbeddings,
    which wraps the same local bge-small model as the other backends
"""

from __future__ import annotations

import hashlib
import logging
import re
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

from memory.backends.base import Candidate, MemoryBackend, content_words, rank
from memory.backends.pgvector import stamp

log = logging.getLogger(__name__)

CANDIDATES = 50
ALL = 100_000           # "every item" for listing a scope
FIELDS = ["content.content", "summary"]     # what gets embedded: a fact's text, an episode's summary


def scope_key(scope: str) -> str:
    """Namespace labels can't contain '.', and a home path does."""
    return hashlib.sha1(scope.encode()).hexdigest()[:16]


def fact_text(value: dict) -> str:
    content = value.get("content")
    return content.get("content", "") if isinstance(content, dict) else str(content or "")


def stems(text: str) -> set[str]:
    """Lowercase words with a plural 's' dropped: enough for "meetings"
    to match "meeting", like the other backends' stemming."""
    return {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in re.findall(r"\w+", text.lower())}


try:
    from langchain_core.embeddings import Embeddings
except ImportError:                 # the extra isn't installed: only this backend needs it
    Embeddings = object


class SoraEmbeddings(Embeddings):
    """sora's Embedder as a LangChain Embeddings, which is what LangGraph's
    index expects. It has to be a subclass: LangGraph treats anything else
    as a plain function and calls it.

    bge embeds questions and documents differently. Duplicate detection
    compares a new fact with stored facts, document to document (the 0.95
    threshold was measured that way), so as_documents() makes queries use
    the document form for the length of a with-block."""

    def __init__(self, embedder):
        self.embedder = embedder
        self.documents_only = False

    @contextmanager
    def as_documents(self):
        self.documents_only = True
        try:
            yield
        finally:
            self.documents_only = False

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self.embedder.documents(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        if self.documents_only:
            return self.embedder.documents([text])[0].tolist()
        return self.embedder.query(text).tolist()

    async def aembed_documents(self, texts):
        return self.embed_documents(texts)

    async def aembed_query(self, text):
        return self.embed_query(text)


def index_config(embedder, dims: int = 384) -> tuple[dict, SoraEmbeddings]:
    embeddings = SoraEmbeddings(embedder)
    return {"dims": dims, "embed": embeddings, "fields": FIELDS}, embeddings


class LangGraphMemory(MemoryBackend):
    name = "langgraph"

    def __init__(self, store, scope: str, embedder=None, min_similarity: float = 0.45,
                 embeddings: SoraEmbeddings | None = None, owner=None):
        super().__init__(embedder, min_similarity)
        self.store = store
        self.embeddings = embeddings          # set when the store has a vector index
        self.owner = owner                    # e.g. the Postgres connection to close
        self.prefix = ("sora", scope_key(scope))
        self.facts_ns = (*self.prefix, "facts")
        self.episodes_ns = (*self.prefix, "episodes")

    def close(self) -> None:
        if self.owner is not None:
            self.owner.close()

    def _items(self, namespace) -> list:
        items = self.store.search(namespace, limit=ALL)
        return sorted(items, key=lambda it: (self._created(it), it.created_at),   # the store's time breaks ties
                      reverse=True)

    @staticmethod
    def _text(kind: str, value: dict) -> str:
        return fact_text(value) if kind == "fact" else value.get("summary", "")

    @staticmethod
    def _created(item) -> str:
        return item.value.get("created_at") or stamp(item.created_at)

    def _search(self, kind: str, query: str, top_k: int) -> list[dict]:
        namespace = self.facts_ns if kind == "fact" else self.episodes_ns
        items = {it.key: it for it in self._items(namespace)}
        if not items:
            return []
        wanted = {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in content_words(query)}
        hits = {k for k, it in items.items() if wanted & stems(self._text(kind, it.value))}
        if self.embeddings is not None and self.embedder is not None:
            try:
                near = self.store.search(namespace, query=query, limit=CANDIDATES)
            except Exception as exc:
                self._embedder_failed(exc)
            else:
                candidates = [Candidate(it.key, it.score, it.key in hits) for it in near
                              if it.score is not None]
                candidates += [Candidate(k, None, True) for k in hits]
                ranked = rank(candidates, self.min_similarity, top_k)
                return [{"id": k, "text": self._text(kind, items[k].value), "created_at": self._created(items[k])}
                        for k in ranked if k in items]
        # words only: most matching words first, then newest
        every = stems(query)
        scored = sorted(((len(every & stems(self._text(kind, it.value))), k) for k, it in items.items()),
                        key=lambda t: t[0], reverse=True)
        return [{"id": k, "text": self._text(kind, items[k].value), "created_at": self._created(items[k])}
                for n, k in scored[:top_k] if n > 0]

    # ---- facts

    def _has_fact(self, content: str) -> bool:
        return any(fact_text(it.value) == content for it in self._items(self.facts_ns))

    def closest_fact(self, content: str) -> tuple[float, str] | None:
        if self.embedder is None or self.embeddings is None:
            return None
        try:
            with self.embeddings.as_documents():
                found = self.store.search(self.facts_ns, query=content, limit=1)
        except Exception as exc:
            log.warning("couldn't compare facts by meaning: %s", exc)
            return None
        if not found or found[0].score is None:
            return None
        return float(found[0].score), fact_text(found[0].value)

    def _insert_fact(self, content: str, source: str, created_at: str | None = None) -> None:
        value = {"kind": "Memory", "content": {"content": content}, "source": source,
                 "created_at": created_at or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")}
        self._put(self.facts_ns, value)

    def _put(self, namespace, value) -> None:
        try:
            self.store.put(namespace, str(uuid.uuid4()), value)
        except Exception as exc:
            if self.embeddings is None:
                raise
            # The model failed while indexing: save without a vector, like the
            # other backends do, rather than losing the memory.
            log.warning("couldn't embed now, saving without a vector: %s", exc)
            self.store.put(namespace, str(uuid.uuid4()), value, index=False)

    def list_facts(self) -> list[dict]:
        return [{"id": it.key, "content": fact_text(it.value), "source": it.value.get("source", "langmem"),
                 "created_at": self._created(it)} for it in self._items(self.facts_ns)]

    def _delete_fact(self, fact_id) -> None:
        self.store.delete(self.facts_ns, str(fact_id))

    # ---- episodes

    def _insert_episode(self, summary: str, session_id: str, created_at: str | None = None) -> None:
        self._put(self.episodes_ns, {
            "summary": summary, "session_id": session_id,
            "created_at": created_at or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")})

    def list_episodes(self) -> list[dict]:
        return [{"id": it.key, "summary": it.value.get("summary", ""), "session_id": it.value.get("session_id", ""),
                 "created_at": self._created(it)} for it in self._items(self.episodes_ns)]

    def _delete_episode(self, episode_id) -> None:
        self.store.delete(self.episodes_ns, str(episode_id))

    def clear(self) -> None:
        for namespace in (self.facts_ns, self.episodes_ns):
            for it in self.store.search(namespace, limit=ALL):
                self.store.delete(namespace, it.key)
