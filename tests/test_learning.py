"""Self-learning: signals, reflection, and lessons coming back into prompts."""

import pytest

from fake import FakeClient
from memory.learning import LessonStore, detect, reflect
from tools.registry import Tool, ToolRegistry


@pytest.fixture
def lessons(tmp_path):
    return LessonStore(tmp_path / "memory.db")


def lesson(situation, text, tool=""):
    return {"situation": situation, "lesson": text, "tool": tool}


def test_detect_signals():
    assert detect() == []
    assert detect(tool_errors=2, thumbs_down=True) == ["tool_error", "thumbs_down"]
    assert detect(limit_reached=True, correction=True) == ["iteration_limit", "user_correction"]


def test_no_signal_means_no_model_call(lessons):
    client = FakeClient()
    assert reflect(lessons, client, "m", [], "hi", "hello", []) == []
    assert client.small_requests == []


def test_reflection_saves_lessons_with_provenance(lessons):
    events = []
    client = FakeClient(small={"lessons": [
        lesson("creating calendar events", "Pass dates as YYYY-MM-DD", "calendar__create"),
        lesson("the user asks about weather", "Search with the city name and today's date")]})
    calls = [{"tool": "calendar__create", "args": {"date": "next friday"},
              "output": "Invalid date", "is_error": True}]
    saved = reflect(lessons, client, "m", ["tool_error"], "book friday", "Sorry", calls,
                    turn_id="t1", notify=lambda k, **kw: events.append((k, kw)))
    assert len(saved) == 2
    prompt = client.small_requests[0]["messages"][0]["content"]
    assert "ERROR: Invalid date" in prompt and "tool_error" in prompt
    row = lessons.list()[-1]
    assert row["tool"] == "calendar__create" and row["source_turn"] == "t1" and row["signal"] == "tool_error"
    assert [k for k, _ in events] == ["llm_start", "llm", "learning"]


def test_correction_shows_the_corrected_reply(lessons):
    client = FakeClient(small={"lessons": []})
    reflect(lessons, client, "m", ["user_correction"], "no, I said Pune", "ok",
            [], previous={"user": "where does Priya live?", "reply": "Mumbai"})
    assert "Assistant: Mumbai" in client.small_requests[0]["messages"][0]["content"]


def test_reflection_failure_is_safe(lessons):
    client = FakeClient(small=RuntimeError("overloaded"))
    assert reflect(lessons, client, "m", ["thumbs_down"], "x", "y", []) == []
    assert lessons.list() == []


def test_duplicates_and_cap(lessons):
    client = FakeClient(small={"lessons": [lesson("s", f"lesson {i}") for i in range(5)] })
    assert len(reflect(lessons, client, "m", ["thumbs_down"], "x", "y", [])) == 3
    assert not lessons.add("s", "LESSON   0")


def test_lessons_come_back(lessons):
    lessons.add("the user asks about weather", "Search with the city name and today's date")
    lessons.add("dates", "Pass dates as YYYY-MM-DD", tool="calendar__create")
    found = lessons.relevant("what's the weather in Pune?")
    assert found == ["When the user asks about weather: Search with the city name and today's date"]
    assert lessons.list()[-1]["uses"] == 1
    # tool lessons stay out of the general lessons, and go into the per-turn prompt,
    # never into the tool description (that would invalidate the prompt cache)
    assert lessons.relevant("YYYY") == []
    notes = lessons.tool_notes(["calendar__create", "current_time"])
    assert notes == {"calendar__create": ["Pass dates as YYYY-MM-DD"]}
    assert next(l for l in lessons.list() if l["tool"])["uses"] == 1
    registry = ToolRegistry()
    registry.register(Tool("calendar__create", "Create an event.", {"type": "object"}, lambda: ""))
    assert registry.schemas()[0]["description"] == "Create an event."
    from datetime import datetime
    from harness.app import build_system
    with_notes = build_system("soul", datetime(2026, 11, 16), "", [], [], None, tool_notes=notes)
    without = build_system("soul", datetime(2026, 11, 16), "", [], [], None)
    assert with_notes[0] == without[0]                                  # cached prefix unchanged
    assert "- calendar__create: Pass dates as YYYY-MM-DD" in with_notes[1]["text"]
