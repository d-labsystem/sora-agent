"""memory/store.py: sessions, history, facts and episodes in one SQLite file."""

import numpy as np
import pytest

from memory.store import MemoryStore, fts_query


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path / "state.db")


def test_a_new_session_is_titled_after_its_first_message(store):
    sid = store.ensure_session(None, "Plan   my trip to Pune next week " + "x" * 80)
    assert store.list_sessions()[0]["title"] == ("Plan my trip to Pune next week " + "x" * 80)[:60]
    assert store.ensure_session(sid) == sid


def test_sessions_keep_separate_histories(store):
    a, b = store.ensure_session(None, "a"), store.ensure_session(None, "b")
    store.add_exchange(a, "hi from a", "hello a", "t1")
    store.add_exchange(b, "hi from b", "hello b", "t2")
    assert [m["content"] for m in store.history(a, 10)] == ["hi from a", "hello a"]
    assert store.last_exchange(b) == {"user": "hi from b", "reply": "hello b", "turn_id": "t2"}


def test_history_is_a_bounded_window_starting_with_the_user(store):
    sid = store.ensure_session(None, "s")
    for i in range(5):
        store.add_exchange(sid, f"q{i}", f"a{i}")
    assert [m["content"] for m in store.history(sid, 2)] == ["q3", "a3", "q4", "a4"]
    assert store.history(sid, 2)[0]["role"] == "user"


def test_empty_reply_is_stored_as_a_placeholder(store):
    sid = store.ensure_session(None, "s")
    store.add_exchange(sid, "q", "")
    assert store.history(sid, 1)[1]["content"] == "(no reply)"


def test_fact_search_matches_word_forms_and_ignores_query_syntax(store):
    store.add_fact("Alex prefers morning meetings.")
    assert store.search_facts("meeting") == ["Alex prefers morning meetings."]
    assert store.search_facts('Alex" OR NEAR(') == ["Alex prefers morning meetings."]
    assert store.search_facts("!!!") == []
    assert fts_query("hi there") == '"hi" OR "there"'


def test_duplicate_facts_are_stored_once(store):
    assert store.add_fact("A fact.") is True
    assert store.add_fact("A   fact.") is False
    assert [f["content"] for f in store.list_facts()] == ["A fact."]


def test_forget_removes_matching_facts_only(store):
    store.add_fact("Alex prefers mornings.")
    store.add_fact("Sam likes tea.")
    assert store.forget("ALEX") == 1
    assert [f["content"] for f in store.list_facts()] == ["Sam likes tea."]
    assert store.forget("   ") == 0


def test_episodes_carry_their_date(store):
    sid = store.ensure_session(None, "s")
    store.add_episode("Planned a trip to Pune with Priya.", sid)
    [found] = store.search_episodes("Pune trip")
    assert found.endswith("Planned a trip to Pune with Priya.") and found.startswith("(20")


def test_consolidation_marks_are_per_session(store):
    a, b = store.ensure_session(None, "a"), store.ensure_session(None, "b")
    store.add_exchange(a, "1", "2")
    store.add_exchange(b, "3", "4")
    store.mark_consolidated(a, store.unconsolidated(a)[-1]["id"])
    assert store.unconsolidated(a) == []
    assert len(store.unconsolidated(b)) == 2


def test_memory_survives_reopening_the_file(tmp_path):
    MemoryStore(tmp_path / "state.db").add_fact("Persisted.")
    assert MemoryStore(tmp_path / "state.db").search_facts("persisted") == ["Persisted."]


# ---- hybrid search (fake embedder from conftest: "snickers" ~ "peanuts", "boss" ~ "manager")

from pathlib import Path  # noqa: E402

from conftest import FakeEmbedder  # noqa: E402


def test_meaning_finds_what_words_miss(tmp_path):
    words = MemoryStore(tmp_path / "w.db")
    hybrid = MemoryStore(tmp_path / "h.db", FakeEmbedder(), min_similarity=0.3)
    for s in (words, hybrid):
        s.add_fact("Aryan is allergic to peanuts")
        s.add_fact("Aryan's manager is Rohan")
    assert words.search_facts("snickers") == []
    assert hybrid.search_facts("snickers") == ["Aryan is allergic to peanuts"]
    assert hybrid.search_facts("who is my boss") == ["Aryan's manager is Rohan"]


def test_unrelated_facts_stay_out(tmp_path):
    s = MemoryStore(tmp_path / "m.db", FakeEmbedder(), min_similarity=0.45)
    s.add_fact("Aryan is allergic to peanuts")
    assert s.search_facts("weather tomorrow") == []


def test_exact_word_always_counts(tmp_path):
    s = MemoryStore(tmp_path / "m.db", FakeEmbedder(), min_similarity=0.99)
    s.add_fact("Rohan's employee ID is EMP-4471")
    assert s.search_facts("EMP-4471 please") == ["Rohan's employee ID is EMP-4471"]


def test_backfill_delete_and_model_change(tmp_path):
    path = tmp_path / "m.db"
    old = MemoryStore(path)                                   # facts saved before embeddings
    old.add_fact("Aryan is allergic to peanuts")
    old.add_episode("Talked about nut allergies", "s1")
    s = MemoryStore(path, FakeEmbedder(), min_similarity=0.3)
    assert s.search_facts("snickers") == ["Aryan is allergic to peanuts"]   # embedded on demand
    assert "nut allergies" in s.search_episodes("snickers")[0]
    count = lambda: s.conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
    assert count() == 2
    s.forget("peanuts")
    assert count() == 1                                        # vectors go with their facts
    other = FakeEmbedder()
    other.model_name = "another-model"
    s2 = MemoryStore(path, other, min_similarity=0.3)
    s2.search_episodes("nuts")
    assert s2.conn.execute("SELECT COUNT(*) FROM vectors WHERE model='another-model'").fetchone()[0] == 1


def test_broken_model_falls_back_to_words(tmp_path):
    class Broken(FakeEmbedder):
        def query(self, text):
            raise OSError("no network to download the model")
    s = MemoryStore(tmp_path / "m.db", Broken())
    s.add_fact("Alex prefers morning meetings")
    assert s.search_facts("Alex meetings") == ["Alex prefers morning meetings"]
    assert s.embedder is None                                   # stays on words from then on


REAL_MODEL = Path.home() / ".sora" / "models"


@pytest.mark.skipif(not any(REAL_MODEL.glob("*bge-small*")), reason="real model not downloaded")
def test_real_model_paraphrase(tmp_path):
    from memory.embeddings import Embedder
    s = MemoryStore(tmp_path / "m.db", Embedder(REAL_MODEL))
    for f in ("Aryan is allergic to peanuts", "Alex prefers morning meetings",
              "Aryan goes to the gym on Tuesdays and Thursdays"):
        s.add_fact(f)
    assert s.search_facts("Can I eat a Snickers bar?", 1) == ["Aryan is allergic to peanuts"]
    assert s.search_facts("Which days do I work out?", 1) == ["Aryan goes to the gym on Tuesdays and Thursdays"]


def test_history_carries_tool_notes_but_chat_stays_clean(store):
    sid = store.ensure_session(None, "x")
    store.add_exchange(sid, "remember chai", "Saved!", "t1",
                       actions="[Recorded by the app, not written by you. Tools used in this reply: remember(...)]")
    history = store.history(sid, 5)
    assert history[1]["content"].startswith("Saved!\n\n[Recorded by the app")
    shown = store.messages(sid)
    assert shown[1]["content"] == "Saved!" and shown[1]["actions"].startswith("[Recorded")


def test_old_database_gets_actions_column(tmp_path):
    import sqlite3
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript("""CREATE TABLE sessions (id TEXT PRIMARY KEY, title TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT (datetime('now')), updated_at TEXT NOT NULL DEFAULT (datetime('now')));
        CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id),
        role TEXT NOT NULL CHECK (role IN ('user', 'assistant')), content TEXT NOT NULL, turn_id TEXT,
        consolidated INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL DEFAULT (datetime('now')));
        INSERT INTO sessions (id, title) VALUES ('s', 't');
        INSERT INTO messages (session_id, role, content) VALUES ('s', 'user', 'hi'), ('s', 'assistant', 'hello');""")
    conn.commit(); conn.close()
    s = MemoryStore(path)
    assert [m["content"] for m in s.history("s", 5)] == ["hi", "hello"]


def test_rewording_is_a_duplicate(tmp_path):
    s = MemoryStore(tmp_path / "m.db", FakeEmbedder())
    assert s.add_fact("Kiran birthday is on 3 December")
    assert not s.add_fact("On 3 December is Kiran birthday")     # same words, new order: same meaning
    assert len(s.list_facts()) == 1


class FixedEmbedder(FakeEmbedder):
    """Returns preset vectors, to place facts at exact similarities."""
    def __init__(self, table):
        self.table, self.model_name = table, "fixed"

    def documents(self, texts):
        return np.array([self.table[t] for t in texts], dtype=np.float32)

    def query(self, text):
        return self.table[text]


def test_related_fact_is_stored_with_a_warning(tmp_path):
    import math
    from tools.builtin import TurnContext, builtin_tools
    angle = lambda sim: np.array([sim, math.sqrt(1 - sim * sim)], dtype=np.float32)
    table = {"birthday 3 Dec": angle(1.0), "birthday 5 Dec": angle(0.92),
             "birthday 3rd Dec": angle(0.97), "likes chai": angle(0.10)}
    s = MemoryStore(tmp_path / "m.db", FixedEmbedder(table))
    remember = next(t for t in builtin_tools(s, None, TurnContext()) if t.name == "remember").function
    assert remember("birthday 3 Dec") == "Remembered: birthday 3 Dec"
    assert remember("birthday 3rd Dec").startswith("Already known, not saved again: birthday 3 Dec")
    reply = remember("birthday 5 Dec")                           # an update: kept, and flagged
    assert reply.startswith("Remembered: birthday 5 Dec") and "related fact is also stored: birthday 3 Dec" in reply
    assert remember("likes chai") == "Remembered: likes chai"
    assert {f["content"] for f in s.list_facts()} == {"birthday 3 Dec", "birthday 5 Dec", "likes chai"}


@pytest.mark.skipif(not any(REAL_MODEL.glob("*bge-small*")), reason="real model not downloaded")
def test_real_model_dedupe_keeps_updates(tmp_path):
    from memory.embeddings import Embedder
    s = MemoryStore(tmp_path / "m.db", Embedder(REAL_MODEL))
    assert s.add_fact("Kiran (the user's teammate) has a birthday on 3 December.")
    assert not s.add_fact("The user's teammate Kiran's birthday is on 3 December.")   # live-session duplicate
    assert s.add_fact("The user's teammate Kiran's birthday is on 5 December.")       # an update survives
