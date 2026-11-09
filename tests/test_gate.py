"""The retrieval gate: when it searches, what it searches for, and how it fails."""

import pytest

from fake import FakeClient, message, text
from memory.gate import decide, recall
from memory.store import MemoryStore


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(tmp_path / "memory.db")
    s.add_fact("Priya's birthday is on 14 March")
    s.add_fact("Alex prefers morning meetings")
    return s


def gate(retrieve, query="", correction=False, reason="r"):
    return {"retrieve": retrieve, "query": query, "correction": correction, "reason": reason}


def test_no_retrieval_means_no_search(store):
    client = FakeClient(small=gate(False))
    sid = store.ensure_session(None, "hi")
    result = recall(store, client, "claude-haiku-5-5", sid, "hi")
    assert not result.retrieve and result.facts == []


def test_rewritten_query_finds_the_fact(store):
    events = []
    client = FakeClient(small=gate(True, "Priya birthday"))
    sid = store.ensure_session(None, "x")
    store.add_exchange(sid, "Tell me about Priya", "She is your colleague.")
    result = recall(store, client, "claude-haiku-5-5", sid, "when is her birthday?",
                    lambda kind, **kw: events.append((kind, kw)))
    assert result.facts == ["Priya's birthday is on 14 March"]
    # the previous exchange was shown to the gate, so it could resolve "her"
    assert "Tell me about Priya" in client.small_requests[0]["messages"][0]["content"]
    assert client.small_requests[0]["output_config"]["format"]["type"] == "json_schema"
    kinds = [k for k, _ in events]
    assert kinds == ["llm", "gate"]
    assert events[0][1]["name"] == "retrieval gate" and events[0][1]["ok"]


def test_empty_query_falls_back_to_message(store):
    client = FakeClient(small=gate(True, "  "))
    result = decide(client, "m", "Alex meetings?", None)
    assert result.query == "Alex meetings?"


def test_failure_fails_open(store):
    events = []
    client = FakeClient(small=RuntimeError("overloaded"))
    sid = store.ensure_session(None, "x")
    result = recall(store, client, "m", sid, "morning meetings",
                    lambda kind, **kw: events.append((kind, kw)))
    assert result.failed and result.retrieve and not result.correction
    assert result.facts == ["Alex prefers morning meetings"]
    assert events[0][0] == "llm" and events[0][1]["ok"] is False


def test_refusal_is_a_failure():
    client = FakeClient()
    client.messages.create = lambda **_: message(text(""), stop_reason="refusal")
    assert decide(client, "m", "hello", None).failed


def test_correction_needs_a_previous_reply(store):
    client = FakeClient(small=gate(False, correction=True))
    assert decide(client, "m", "no, that's wrong", None).correction is False
    sid = store.ensure_session(None, "x")
    store.add_exchange(sid, "What's 2+2?", "5")
    assert recall(store, client, "m", sid, "no, that's wrong").correction is True
