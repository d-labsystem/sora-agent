"""Tracing: spans from notify, totals, feedback, stats and export."""

import io
import json

import pytest

from ops.tracing import Tracer


@pytest.fixture
def tracer(tmp_path):
    return Tracer(tmp_path / "traces.db")


def run_turn(tracer, message="plan my week", **finish):
    live = []
    turn = tracer.start(message, session_id="s1", model="claude-opus-5-5",
                        listener=lambda kind, data: live.append(kind))
    turn.notify("llm", name="retrieval gate", ms=5, ok=True, input_tokens=100,
                output_tokens=10, cost_usd=0.0001)
    turn.notify("gate", name="memory recall", retrieve=True, facts=["a"])
    turn.notify("text", delta="Hel")
    turn.notify("llm", name="model call 1", ms=20, ok=True, input_tokens=1000,
                output_tokens=200, web_searches=2, cost_usd=0.03)
    turn.notify("tool", name="workspace__read_file", args={"path": "x"}, output="missing",
                ok=False, ms=3)
    turn.notify("tool", name="remember", args={}, output="ok", ok=True, ms=1)
    return turn, turn.finish("Hello", iterations=2, **finish), live


def test_spans_and_rolled_up_totals(tracer):
    turn, row, live = run_turn(tracer)
    assert live == ["llm", "gate", "text", "llm", "tool", "tool"]       # UI sees everything
    assert [s["kind"] for s in row["spans"]] == ["llm", "gate", "llm", "tool", "tool"]  # no text
    assert row["status"] == "ok" and row["reply"] == "Hello" and row["iterations"] == 2
    assert (row["input_tokens"], row["output_tokens"], row["web_searches"]) == (1100, 210, 2)
    assert row["cost_usd"] == pytest.approx(0.0301)
    assert (row["tool_calls"], row["tool_errors"]) == (2, 1)
    assert row["spans"][3]["ok"] is False and row["spans"][3]["data"]["output"] == "missing"
    assert turn.finish("again")["reply"] == "Hello"                     # finishing twice is a no-op


def test_span_start_is_end_minus_duration(tracer):
    turn = tracer.start("x")
    turn.notify("llm", name="slow", ms=10_000)
    assert tracer.spans(turn.id)[0]["start_ms"] == 0                    # clamped, never negative


def test_long_values_are_clipped(tracer):
    turn = tracer.start("x")
    turn.notify("tool", name="t", output="a" * 10_000, nested={"k": ["b" * 9000]}, obj=object())
    data = tracer.spans(turn.id)[0]["data"]
    assert len(data["output"]) < 4100 and "[10000 chars]" in data["output"]
    assert len(data["nested"]["k"][0]) < 4100 and data["obj"].startswith("<object")


def test_broken_listener_does_not_break_the_turn(tracer):
    def boom(kind, data):
        raise RuntimeError("socket closed")
    turn = tracer.start("x", listener=boom)
    turn.notify("tool", name="t")
    assert len(tracer.spans(turn.id)) == 1


def test_feedback_judge_and_checks(tracer):
    turn, _, _ = run_turn(tracer, checks={"no_tool_errors": False})
    assert tracer.set_feedback(turn.id, -1, "wrong file")
    tracer.set_judge(turn.id, 0.4, "missed the file")
    row = tracer.turn(turn.id)
    assert (row["feedback"], row["feedback_note"]) == (-1, "wrong file")
    assert row["judge_score"] == 0.4 and row["checks"] == {"no_tool_errors": False}
    assert not tracer.set_feedback("nope", 1)
    with pytest.raises(ValueError):
        tracer.set_feedback(turn.id, 5)


def test_crash_recovery(tmp_path):
    first = Tracer(tmp_path / "traces.db")
    turn = first.start("cut off")
    again = Tracer(tmp_path / "traces.db")
    row = again.turn(turn.id)
    assert row["status"] == "error" and "interrupted" in row["error"]


def test_listing_filters_and_delete(tracer):
    run_turn(tracer)
    eval_turn = tracer.start("case 1", source="eval", eval_run="r1")
    eval_turn.finish("ok")
    assert len(tracer.turns()) == 2
    assert [t["message"] for t in tracer.turns(source="eval")] == ["case 1"]
    assert len(tracer.turns(eval_run="r1")) == 1
    tracer.delete(eval_turn.id)
    assert tracer.turn(eval_turn.id) is None and tracer.spans(eval_turn.id) == []


def test_stats(tracer):
    for _ in range(3):
        run_turn(tracer)
    tracer.start("failed").finish(None, status="error", error="boom")
    tracer.start("from an eval", source="eval").finish("x")
    s = tracer.stats()
    assert s["turns"] == 4 and s["failed"] == 1 and s["error_rate"] == 0.25    # evals excluded
    assert s["cost_usd"] == pytest.approx(0.0903)
    assert s["per_tool"][0] == {"tool": "workspace__read_file", "calls": 3, "errors": 3,
                                "avg_ms": 3.0}
    assert s["p50_latency_ms"] is not None and len(s["per_day"]) == 1


def test_export_jsonl(tracer):
    run_turn(tracer)
    tracer.start("still running")
    out = io.StringIO()
    assert tracer.export_jsonl(out) == 1
    record = json.loads(out.getvalue())
    assert record["message"] == "plan my week" and len(record["spans"]) == 5
