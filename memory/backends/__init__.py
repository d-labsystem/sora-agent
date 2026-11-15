"""Choosing where facts and episodes live (SORA_MEMORY).

    sqlite      (default) in memory.db, next to sessions and messages
    pgvector    Postgres tables with a vector column     needs SORA_DATABASE_URL
    langgraph   LangGraph's PostgresStore (pgvector)      needs SORA_DATABASE_URL;
                required by SORA_CONSOLIDATOR=langmem

Each home folder is its own scope in Postgres, so one database can hold your
~/.sora and every eval case's temporary home without them mixing.
"""

from __future__ import annotations

from pathlib import Path

from memory.backends.base import MemoryBackend

BACKENDS = ("sqlite", "pgvector", "langgraph")


def scope_for(home: Path) -> str:
    return str(Path(home).expanduser().resolve())


def open_backend(kind: str, home: Path, url: str = "", embedder=None,
                 min_similarity: float = 0.45) -> MemoryBackend | None:
    """The backend for `kind`; None for sqlite (MemoryStore makes its own,
    on the connection it already has)."""
    if kind == "sqlite":
        return None
    if kind not in BACKENDS:
        raise ValueError(f"SORA_MEMORY must be one of {', '.join(BACKENDS)}, not {kind!r}")
    if not url:
        raise ValueError(f"SORA_MEMORY={kind} needs SORA_DATABASE_URL "
                         "(e.g. postgresql://postgres:sora@localhost:5433/postgres)")
    scope = scope_for(home)
    try:
        if kind == "pgvector":
            from memory.backends.pgvector import PgVectorMemory
            return PgVectorMemory(url, scope, embedder, min_similarity)
        import psycopg
        from psycopg.rows import dict_row
        from langgraph.store.postgres import PostgresStore
    except ImportError as exc:
        extra = "memory-pg" if kind == "pgvector" else "memory-langmem"
        raise ValueError(f"SORA_MEMORY={kind} needs extra packages: uv sync --extra {extra} ({exc})") from exc
    from memory.backends.langgraph import LangGraphMemory, index_config
    conn = psycopg.connect(url, autocommit=True, prepare_threshold=0, row_factory=dict_row)
    index, embeddings = index_config(embedder) if embedder is not None else (None, None)
    store = PostgresStore(conn, index=index)
    store.setup()                       # creates its tables and runs its migrations once
    return LangGraphMemory(store, scope, embedder, min_similarity, embeddings, owner=conn)


def open_memory(settings, embedder=None, kind: str | None = None):
    """A MemoryStore for this home with the configured backend."""
    from memory.store import MemoryStore
    backend = open_backend(kind or settings.memory_backend, settings.home, settings.database_url,
                           embedder, settings.memory_min_similarity)
    return MemoryStore(settings.home / "memory.db", embedder, settings.memory_min_similarity, backend)
