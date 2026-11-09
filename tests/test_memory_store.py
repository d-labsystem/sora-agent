"""memory/store.py: sessions, history, facts and episodes in one SQLite file."""

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
