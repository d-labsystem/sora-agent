"""Consolidation: when it runs, what it saves, and that failure loses nothing."""

import re

import pytest

from fake import FakeClient
from memory.consolidation import KNOWN_FACTS, consolidate
from memory.store import MemoryStore


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path / "memory.db")


def chat(store, sid, n):
    for i in range(n):
        store.add_exchange(sid, f"message {i}", f"reply {i}")


ANSWER = {"facts": ["Priya lives in Pune", "Alex prefers trains"], "episode": "Planned a Pune trip"}


def test_waits_until_enough_exchanges(store):
    client = FakeClient(small=ANSWER)
    sid = store.ensure_session(None, "x")
    chat(store, sid, 5)
    assert consolidate(store, client, "m", sid) is None
    assert client.small_requests == []
    chat(store, sid, 1)
    result = consolidate(store, client, "m", sid)
    assert result["messages"] == 12 and result["facts"] == ANSWER["facts"]
    assert store.unconsolidated(sid) == []
    assert "Planned a Pune trip" in store.search_episodes("Pune")[0]


def test_force_runs_early_and_empty_session_does_nothing(store):
    client = FakeClient(small=ANSWER)
    sid = store.ensure_session(None, "x")
    assert consolidate(store, client, "m", sid, force=True) is None
    chat(store, sid, 1)
    assert consolidate(store, client, "m", sid, force=True) is not None


def test_failure_keeps_messages_for_retry(store):
    events = []
    client = FakeClient(small=[RuntimeError("overloaded"), ANSWER])
    sid = store.ensure_session(None, "x")
    chat(store, sid, 6)
    notify = lambda kind, **kw: events.append((kind, kw))
    assert consolidate(store, client, "m", sid, notify) is None
    assert len(store.unconsolidated(sid)) == 12 and store.list_facts() == []
    assert ("memory", "consolidation failed") in [(k, e.get("name")) for k, e in events]
    assert consolidate(store, client, "m", sid, notify)["facts"]
    assert store.unconsolidated(sid) == []


def test_known_facts_are_shown_and_duplicates_skipped(store):
    store.add_fact("Priya lives in Pune")
    client = FakeClient(small=ANSWER)
    sid = store.ensure_session(None, "x")
    chat(store, sid, 6)
    result = consolidate(store, client, "m", sid)
    assert result["facts"] == ["Alex prefers trains"]
    assert re.search(r"- \(\d{4}-\d\d-\d\d\) Priya lives in Pune", client.small_requests[0]["messages"][0]["content"])


def test_the_writer_sees_related_facts_even_when_old(store):
    """It used to see only the 40 newest, which needn't include the one being talked about."""
    store.add_fact("Jo's pre-1920 coin collection has 37 coins", "consolidation", "2023-05-27 10:00:00")
    for i in range(KNOWN_FACTS + 5):
        store.add_fact(f"Jo plans errand number {i} downtown")
    sid = store.ensure_session(None, "x")
    store.add_exchange(sid, "I just added a quarter to my pre-1920 coin collection!", "Nice!")
    client = FakeClient(small={"facts": [], "episode": ""})
    consolidate(store, client, "m", sid, force=True)
    prompt = client.small_requests[0]["messages"][0]["content"]
    assert "- (2023-05-27) Jo's pre-1920 coin collection has 37 coins" in prompt
    assert prompt.count("\n- (") == KNOWN_FACTS


def test_only_this_session_is_consolidated(store):
    client = FakeClient(small=ANSWER)
    a, b = store.ensure_session(None, "a"), store.ensure_session(None, "b")
    chat(store, a, 6)
    chat(store, b, 6)
    consolidate(store, client, "m", a)
    assert store.unconsolidated(a) == [] and len(store.unconsolidated(b)) == 12
    assert "message 0" in client.small_requests[0]["messages"][0]["content"]


