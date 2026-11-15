"""The memory backend contract: every backend passes the same tests.

    sqlite                always
    langgraph (memory)    always: LangGraph's InMemoryStore, same code as Postgres
    pgvector              with SORA_TEST_DATABASE_URL set (a real Postgres)
    langgraph (postgres)  with SORA_TEST_DATABASE_URL set

Start a database for the last two:
    docker run -d --name sora-pg -p 5433:5432 -e POSTGRES_PASSWORD=sora pgvector/pgvector:pg17
    SORA_TEST_DATABASE_URL=postgresql://postgres:sora@localhost:5433/postgres uv run pytest
"""

import os
import re
import uuid
import zlib
from types import SimpleNamespace as NS

import numpy as np
import pytest

from conftest import CONCEPTS
from memory import mirror
from memory.backends.base import rank, Candidate
from memory.store import MemoryStore

DB = os.getenv("SORA_TEST_DATABASE_URL", "")
needs_db = pytest.mark.skipif(not DB, reason="set SORA_TEST_DATABASE_URL to test against Postgres")


class Fake384:
    """conftest's FakeEmbedder in 384 dimensions, the size pgvector's column holds."""
    model_name = "fake-384"

    def _vector(self, text):
        v = np.zeros(384, dtype=np.float32)
        for word in re.findall(r"\w+", text.lower()):
            v[zlib.crc32(CONCEPTS.get(word, word).encode()) % 384] += 1
        n = np.linalg.norm(v)
        return v / n if n else v

    def documents(self, texts):
        return np.array([self._vector(t) for t in texts])

    def query(self, text):
        return self._vector(text)


def _langgraph_memory(store_factory):
    from memory.backends.langgraph import LangGraphMemory, index_config

    def make(scope, embedder, min_similarity):
        index, embeddings = index_config(embedder) if embedder else (None, None)
        return LangGraphMemory(store_factory(index), scope, embedder, min_similarity, embeddings)
    return make


def _memory_store(tmp_path):
    def make(scope, embedder, min_similarity):
        return MemoryStore(tmp_path / f"{scope}.db", embedder, min_similarity).semantic
    return make


def _pgvector():
    from memory.backends.pgvector import PgVectorMemory
    return lambda scope, embedder, min_similarity: PgVectorMemory(DB, scope, embedder, min_similarity)


def _langgraph_postgres():
    import psycopg
    from psycopg.rows import dict_row
    from langgraph.store.postgres import PostgresStore

    def store(index):
        conn = psycopg.connect(DB, autocommit=True, prepare_threshold=0, row_factory=dict_row)
        s = PostgresStore(conn, index=index)
        s.setup()
        return s
    return _langgraph_memory(store)


def _in_memory(index):
    from langgraph.store.memory import InMemoryStore
    return InMemoryStore(index=index) if index else InMemoryStore()


@pytest.fixture(params=["sqlite", "langgraph-memory",
                        pytest.param("pgvector", marks=needs_db),
                        pytest.param("langgraph-postgres", marks=needs_db)])
def backend(request, tmp_path):
    """make(embedder=Fake384(), min_similarity=0.3, scope=None) -> a fresh backend."""
    factory = {"sqlite": lambda: _memory_store(tmp_path),
               "langgraph-memory": lambda: _langgraph_memory(_in_memory),
               "pgvector": _pgvector, "langgraph-postgres": _langgraph_postgres}[request.param]()
    made = []
    # one in-memory store per test, shared by its scopes (like one database)
    if request.param == "langgraph-memory":
        shared = {}
        factory = _langgraph_memory(lambda index: shared.setdefault(bool(index), _in_memory(index)))

    def make(embedder="fake", min_similarity=0.3, scope=None):
        b = factory(scope or f"test-{uuid.uuid4().hex[:8]}", Fake384() if embedder == "fake" else embedder,
                    min_similarity)
        made.append(b)
        return b
    make.kind = request.param
    yield make
    for b in made:
        b.clear()
        b.close()


# ---- facts

def test_meaning_finds_what_words_miss(backend):
    m = backend()
    m.add_fact("Aryan is allergic to peanuts")
    m.add_fact("Aryan's manager is Rohan")
    assert m.search_facts("snickers") == ["Aryan is allergic to peanuts"]
    assert m.search_facts("who is my boss") == ["Aryan's manager is Rohan"]


def test_unrelated_questions_find_nothing(backend):
    m = backend(min_similarity=0.45)
    m.add_fact("Aryan is allergic to peanuts")
    assert m.search_facts("weather tomorrow") == []


def test_an_exact_rare_word_is_found_below_the_threshold(backend):
    m = backend(min_similarity=0.99)
    m.add_fact("Rohan's employee ID is EMP-4471")
    assert m.search_facts("EMP-4471 please") == ["Rohan's employee ID is EMP-4471"]


def test_duplicates_and_rewordings_are_stored_once(backend):
    m = backend()
    assert m.add_fact("Kiran birthday is on 3 December") is True
    assert m.add_fact("Kiran   birthday is on 3 December") is False      # same text, other spacing
    assert m.add_fact("On 3 December is Kiran birthday") is False         # same meaning
    assert m.add_fact("Kiran birthday is on 5 December") is True          # an update is kept
    assert [f["content"] for f in m.list_facts()] == ["Kiran birthday is on 5 December",
                                                      "Kiran birthday is on 3 December"]
    closest = m.closest_fact("Kiran birthday is on 3 December")
    assert closest[1] == "Kiran birthday is on 3 December" and closest[0] > 0.99


def test_forget_and_delete(backend):
    m = backend()
    for f in ("Alex prefers mornings", "Sam likes tea", "Alex runs on Sundays"):
        m.add_fact(f)
    assert m.forget("ALEX") == 2 and m.forget("   ") == 0
    [left] = m.list_facts()
    assert left["content"] == "Sam likes tea" and left["source"] == "tool"
    assert re.fullmatch(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d", left["created_at"])
    m.delete_fact(left["id"])
    assert m.list_facts() == [] and m.search_facts("tea") == []


def test_words_only_without_an_embedder(backend):
    m = backend(embedder=None)
    m.add_fact("Alex prefers morning meetings.")
    assert m.search_facts("meetings") == ["Alex prefers morning meetings."]
    assert m.search_facts('Alex" OR NEAR(') == ["Alex prefers morning meetings."]
    assert m.search_facts("!!!") == [] and m.closest_fact("x") is None


# ---- episodes

def test_episodes_are_dated_searchable_and_deletable(backend):
    m = backend()
    m.add_episode("Planned a trip to Pune with Priya.", "s1")
    m.add_episode("Talked about nut allergies", "s2")
    [found] = m.search_episodes("snickers", 1)
    assert re.fullmatch(r"\(\d{4}-\d\d-\d\d\) Talked about nut allergies", found)
    newest = m.list_episodes()[0]
    assert newest["summary"] == "Talked about nut allergies" and newest["session_id"] == "s2"
    m.delete_episode(newest["id"])
    assert [e["summary"] for e in m.list_episodes()] == ["Planned a trip to Pune with Priya."]


# ---- scopes, copying, change hook

def test_scopes_do_not_see_each_other(backend):
    if backend.kind == "sqlite":
        pytest.skip("SQLite scopes are separate files")
    mine, eval_case = backend(scope="home-a"), backend(scope="home-b")
    mine.add_fact("Aryan lives in Pune")
    eval_case.add_fact("Test user likes chai")
    assert [f["content"] for f in mine.list_facts()] == ["Aryan lives in Pune"]
    assert mine.search_facts("chai") == []
    eval_case.clear()
    assert eval_case.list_facts() == [] and len(mine.list_facts()) == 1
    mine.clear()


def test_import_keeps_dates_and_runs_once(backend, tmp_path):
    source = MemoryStore(tmp_path / "old.db", Fake384()).semantic
    source._insert_fact("Priya lives in Pune", "consolidation", "2026-01-02 03:04:05")
    source._insert_episode("Planned the Pune trip", "s1", "2026-01-03 10:00:00")
    target = backend()
    assert target.import_from(source) == {"facts": 1, "episodes": 1}
    assert target.import_from(source) == {"facts": 0, "episodes": 0}
    [fact] = target.list_facts()
    assert (fact["content"], fact["source"], fact["created_at"]) == \
        ("Priya lives in Pune", "consolidation", "2026-01-02 03:04:05")
    assert target.search_episodes("Pune trip") == ["(2026-01-03) Planned the Pune trip"]
    assert target.search_facts("where does priya live") == ["Priya lives in Pune"]   # embedded on copy


def test_changes_fire_the_hook_and_the_mirror_reads_well(backend, tmp_path):
    m = backend()
    path = tmp_path / "MEMORY.md"
    m.on_change = lambda: mirror.write(path, m, m.name)
    m.add_fact("Aryan drinks masala chai")
    m.add_episode("Talked about tea", "s1")
    text = path.read_text()
    assert "edits here are not read back" in text and "1 facts · 1 episodes" in text
    assert "- Aryan drinks masala chai  _(" in text and "Talked about tea" in text
    m.forget("chai")
    assert "Aryan drinks masala chai" not in path.read_text()


# ---- the shared ranking

def test_rank_merges_a_row_found_twice_and_applies_the_rules():
    found = [Candidate(1, 0.50, False), Candidate(1, None, True),     # meaning + words: bonus
             Candidate(2, 0.52, False), Candidate(3, 0.10, False),    # 3: below the threshold
             Candidate(4, None, True)]                                 # words only: kept
    assert rank(found, 0.45, 5) == [1, 2, 4]
    assert rank(found, 0.45, 1) == [1]


# ---- wiring: settings choose the backend

def test_settings_choose_the_backend_and_explain_what_is_missing(tmp_path):
    from harness.app import Settings
    from memory.backends import open_memory
    assert open_memory(Settings(home=tmp_path)).semantic.name == "sqlite"
    with pytest.raises(ValueError, match="SORA_DATABASE_URL"):
        open_memory(Settings(home=tmp_path, memory_backend="pgvector"))
    with pytest.raises(ValueError, match="one of sqlite"):
        open_memory(Settings(home=tmp_path, memory_backend="redis", database_url="x"))


def test_the_agent_writes_memory_md_and_reports_its_backend(tmp_path):
    from fake import FakeClient
    from harness.app import Settings, Sora
    agent = Sora(Settings(home=tmp_path), FakeClient([]), connect_mcp=False)
    assert "_None yet._" in (tmp_path / "MEMORY.md").read_text()
    agent.store.add_fact("Aryan lives in Pune")
    assert "Aryan lives in Pune" in (tmp_path / "MEMORY.md").read_text()
    assert agent.status()["memory"]["backend"] == "sqlite"
    with pytest.raises(ValueError, match="SORA_MEMORY=langgraph"):
        Sora(Settings(home=tmp_path / "b", consolidator="langmem"), FakeClient([]), connect_mcp=False)


def test_cli_migrate_copies_into_another_backend(tmp_path, monkeypatch):
    """sqlite -> langgraph with an in-memory store standing in for Postgres."""
    import memory.backends as backends
    from harness.cli import main
    from memory.backends.langgraph import LangGraphMemory
    old = MemoryStore(tmp_path / "memory.db")
    old.add_fact("Priya lives in Pune")
    old.add_episode("Planned the Pune trip", "s1")
    made = []

    def fake_open(kind, home, url="", embedder=None, min_similarity=0.45):
        if kind == "sqlite":
            return None
        made.append(LangGraphMemory(_in_memory(None), str(home)))
        return made[-1]
    monkeypatch.setattr(backends, "open_backend", fake_open)
    monkeypatch.setenv("SORA_EMBEDDINGS", "0")
    out = __import__("io").StringIO()
    assert main(["--home", str(tmp_path), "memory", "migrate", "--to", "langgraph"], out=out) == 0
    assert "copied 1 facts and 1 episodes from sqlite to langgraph" in out.getvalue()
    assert [f["content"] for f in made[0].list_facts()] == ["Priya lives in Pune"]


# ---- LangMem consolidation (the manager is faked; the live check runs the real one)

class FakeManager:
    """Does what LangMem's manager does to the store: rewrites one memory,
    deletes one, adds one; and reports a model call through the callbacks."""

    def __init__(self, backend):
        self.backend = backend

    def invoke(self, payload, config):
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, LLMResult
        store, ns = self.backend.store, self.backend.facts_ns
        by_text = {f["content"]: f["id"] for f in self.backend.list_facts()}
        store.put(ns, by_text["Aryan lives in Pune"], {"kind": "Memory", "content": {"content": "Aryan lives in Mumbai"}})
        store.delete(ns, by_text["Aryan's office is in Pune"])
        store.put(ns, "new-1", {"kind": "Memory", "content": {"content": "Aryan's new office is in Bandra"}})
        [spans] = config["callbacks"]
        run = uuid.uuid4()
        spans.on_chat_model_start({}, [], run_id=run)
        message = AIMessage("", usage_metadata={"input_tokens": 1200, "output_tokens": 80, "total_tokens": 1280,
                                                "input_token_details": {"cache_read": 200}},
                            response_metadata={"model_name": "claude-haiku-5-5"})
        spans.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]), run_id=run)
        return []


def test_langmem_updates_deletes_and_adds_and_every_call_is_a_span(tmp_path):
    from fake import FakeClient
    from memory.backends.langgraph import LangGraphMemory, index_config
    from memory.langmem_consolidator import consolidate_langmem
    index, embeddings = index_config(Fake384())
    backend = LangGraphMemory(_in_memory(index), "home", Fake384(), 0.3, embeddings)
    store = MemoryStore(tmp_path / "memory.db", backend=backend)
    store.add_fact("Aryan lives in Pune")
    store.add_fact("Aryan's office is in Pune")
    sid = store.ensure_session(None, "x")
    for i in range(6):
        store.add_exchange(sid, f"message {i}", f"reply {i}")
    seen = []
    client = FakeClient(small={"episode": "Aryan said he moved to Mumbai"})
    result = consolidate_langmem(store, "claude-haiku-5-5", client, "claude-haiku-5-5", sid,
                                 lambda kind, **d: seen.append((kind, d)), manager=FakeManager(backend))
    assert result["updated"] == ["Aryan lives in Mumbai"]
    assert result["removed"] == ["Aryan's office is in Pune"]
    assert result["facts"] == ["Aryan's new office is in Bandra"]
    assert store.unconsolidated(sid) == [] and "moved to Mumbai" in store.search_episodes("Mumbai")[0]
    [span] = [d for kind, d in seen if kind == "llm" and d["name"] == "langmem"]
    assert (span["input_tokens"], span["cache_read"], span["output_tokens"]) == (1000, 200, 80)
    assert span["cost_usd"] > 0 and result["langmem_cost_usd"] == pytest.approx(span["cost_usd"], abs=1e-6)
    assert {f["content"] for f in store.list_facts()} == {"Aryan lives in Mumbai", "Aryan's new office is in Bandra"}


def test_langmem_failure_loses_nothing(tmp_path):
    from memory.backends.langgraph import LangGraphMemory
    from memory.langmem_consolidator import consolidate_langmem
    store = MemoryStore(tmp_path / "memory.db", backend=LangGraphMemory(_in_memory(None), "home"))
    sid = store.ensure_session(None, "x")
    for i in range(6):
        store.add_exchange(sid, f"m{i}", f"r{i}")
    broken = NS(invoke=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("rate limited")))
    seen = []
    assert consolidate_langmem(store, "claude-haiku-5-5", None, "m", sid,
                               lambda kind, **d: seen.append((kind, d)), manager=broken) is None
    assert len(store.unconsolidated(sid)) == 12
    assert seen[-1][1]["name"] == "consolidation failed" and "rate limited" in seen[-1][1]["error"]


def test_model_specs_map_to_langchain_names():
    from memory.langmem_consolidator import chat_model_name
    assert chat_model_name("claude-haiku-5-5") == "anthropic:claude-haiku-5-5"
    assert chat_model_name("openai:gpt-5.4-mini") == "openai:gpt-5.4-mini"
    with pytest.raises(ValueError):
        chat_model_name("hf:meta-llama/x")
