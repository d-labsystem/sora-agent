import re
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
    assert [f.split(") ", 1)[1] for f in result.facts] == ["Priya's birthday is on 14 March"]
    assert re.match(r"\(\d{4}-\d\d-\d\d\) ", result.facts[0])                      # recalled facts carry their date
    # the previous exchange was shown to the gate, so it could resolve "her"
    assert "Tell me about Priya" in client.small_requests[0]["messages"][0]["content"]
    assert client.small_requests[0]["output_config"]["format"]["type"] == "json_schema"
    kinds = [k for k, _ in events]
    assert kinds == ["llm_start", "llm", "gate"]
    assert events[1][1]["name"] == "retrieval gate" and events[1][1]["ok"]


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
    assert [f.split(") ", 1)[1] for f in result.facts] == ["Alex prefers morning meetings"]
    assert events[1][0] == "llm" and events[1][1]["ok"] is False


def test_messages_about_the_user_always_search(store):
    """A small gate model called "suggest activities I can do in the evening" general
    knowledge and skipped memory (Qwen3.5-9B, LongMemEval-S 2026-11-17)."""
    client = FakeClient(small=gate(False))
    sid = store.ensure_session(None, "x")
    result = recall(store, client, "m", sid, "Can you suggest a time for my meeting with Alex?")
    assert result.retrieve and result.forced
    assert [f.split(") ", 1)[1] for f in result.facts] == ["Alex prefers morning meetings"]
    for general in ("hi", "what's the capital of France?", "thanks!", "Summarise this: mine shafts"):
        result = decide(client, "m", general, None)
        assert result.retrieve == (general.endswith("shafts")) and result.forced == result.retrieve
    assert decide(FakeClient(small=gate(True, "q")), "m", "my plans?", None).forced is False


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


# ---- "all" questions: every mention, not only the best one (LongMemEval-S, 2026-11-16)

def test_all_questions_search_each_phrasing_and_keep_more(tmp_path):
    from memory.gate import WIDE
    s = MemoryStore(tmp_path / "m.db")
    for fact in ["The user plays a black Fender Stratocaster electric guitar",
                 "The user owns a 5-piece Pearl Export drum set they want to sell",
                 "The user has had a Korg B1 piano for about 3 years",
                 "The user's niece is learning the violin"]:
        s.add_fact(fact)
    client = FakeClient(small={**gate(True, "musical instruments owned"), "scope": "all",
                               "queries": ["guitar", "drum set", "piano"]})
    sid = s.ensure_session(None, "x")
    result = recall(s, client, "m", sid, "How many musical instruments do I own?")
    found = " ".join(result.facts)
    assert result.scope == "all" and result.queries == ["guitar", "drum set", "piano"]
    assert all(w in found for w in ("Stratocaster", "Pearl Export", "Korg B1"))
    assert len(result.facts) <= WIDE["facts"] and len(result.facts) == len(set(result.facts))


def test_one_questions_ignore_extra_queries():
    client = FakeClient(small={**gate(True, "Priya birthday"), "scope": "one", "queries": ["cake"]})
    result = decide(client, "m", "When is Priya's birthday?", None)
    assert result.scope == "one" and result.queries == []


def test_merged_takes_each_searchs_best_first_without_repeats():
    from memory.gate import _merged
    assert _merged([["a", "b", "c"], ["d", "a", "e"], ["f"]], 5) == ["a", "d", "f", "b", "c"]


def test_memory_section_carries_the_answer_rules_and_the_all_note():
    from datetime import datetime
    from harness.app import MEMORY_RULES, build_system
    from memory.gate import GateResult
    one = GateResult(True, "q", False, "r", facts=["(2023-05-20) a fact"])
    every = GateResult(True, "q", False, "r", scope="all", facts=["(2023-05-20) a fact"])
    text = lambda found: build_system("soul", datetime(2026, 11, 16), "", [], [], found)[1]["text"]
    assert MEMORY_RULES in text(one) and "every relevant mention" not in text(one)
    assert "every relevant mention" in text(every)
    assert "give it plainly first" in MEMORY_RULES and "Never add people" in MEMORY_RULES
