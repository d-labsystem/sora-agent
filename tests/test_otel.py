"""OpenTelemetry export: a finished turn becomes the same trace in Phoenix and Langfuse."""

import json
from types import SimpleNamespace as NS

import pytest

pytest.importorskip("opentelemetry.sdk")
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter  # noqa: E402

from fake import FakeClient, message, text, tool_use  # noqa: E402
from harness.app import Settings, Sora  # noqa: E402
from ops import otel  # noqa: E402
from ops.tracing import Tracer  # noqa: E402

TURN = {
    "id": "abc123", "session_id": "s1", "source": "chat", "status": "ok", "model": "claude-opus-5-5",
    "message": "What did I save about Pune?", "reply": "You live in Mumbai now.", "iterations": 2,
    "cost_usd": 0.0123, "latency_ms": 2000, "eval_run": None, "checks": {"replied": True},
    "started_at": "2026-11-16T10:00:00.000+00:00", "ended_at": "2026-11-16T10:00:02.000+00:00",
    "spans": [
        {"kind": "gate", "name": "memory recall", "start_ms": 100, "duration_ms": 0, "ok": True,
         "data": {"query": "Pune", "facts": ["Aryan lives in Mumbai"], "retrieve": True}},
        {"kind": "llm", "name": "model call 1", "start_ms": 200, "duration_ms": 900, "ok": True,
         "data": {"model": "claude-opus-5-5", "input_tokens": 100, "output_tokens": 40,
                  "cache_read_tokens": 1000, "cache_write_tokens": 50, "cost_usd": 0.01,
                  "output": "Let me check.", "reasoning_tokens": 12}},
        {"kind": "tool", "name": "recall", "start_ms": 1100, "duration_ms": 50, "ok": False,
         "data": {"args": {"query": "Pune"}, "output": "boom", "error": "boom"}},
    ]}


def exporter(content_phoenix=True, content_langfuse=False):
    sinks = {"phoenix": InMemorySpanExporter(), "langfuse": InMemorySpanExporter()}
    e = otel.OTelExporter([otel.Destination("phoenix", "http://localhost:6006/v1/traces", content=content_phoenix),
                           otel.Destination("langfuse", "https://cloud.langfuse.com/x", content=content_langfuse)],
                          span_exporters=sinks)
    return e, sinks


def by_name(spans):
    return {s.name: s for s in spans}


def test_a_turn_becomes_one_trace_with_exact_timings():
    e, sinks = exporter()
    e.export(TURN)
    spans = by_name(sinks["phoenix"].get_finished_spans())
    root, call, tool = spans["What did I save about Pune?"], spans["model call 1"], spans["recall"]
    assert len(spans) == 4 and root.parent is None and root.status.status_code.name == "OK"
    assert {s.context.trace_id for s in spans.values()} == {root.context.trace_id}
    assert all(s.parent.span_id == root.context.span_id for s in (call, tool, spans["memory recall"]))
    assert (call.end_time - call.start_time) == 900_000_000
    assert (call.start_time - root.start_time) == 200_000_000
    assert (root.end_time - root.start_time) == 2_000_000_000
    assert root.attributes["openinference.span.kind"] == "AGENT" and root.attributes["session.id"] == "s1"
    assert spans["memory recall"].attributes["openinference.span.kind"] == "RETRIEVER"
    assert tool.status.status_code.name == "ERROR"


def test_llm_spans_carry_tokens_and_cost_in_both_vocabularies():
    e, sinks = exporter()
    e.export(TURN)
    a = by_name(sinks["phoenix"].get_finished_spans())["model call 1"].attributes
    assert a["openinference.span.kind"] == "LLM" and a["langfuse.observation.type"] == "generation"
    assert a["llm.model_name"] == "claude-opus-5-5" and a["llm.provider"] == "anthropic"
    assert (a["llm.token_count.prompt"], a["llm.token_count.completion"]) == (1150, 40)   # input incl. cache
    assert a["llm.token_count.prompt_details.cache_read"] == 1000
    assert a["llm.token_count.completion_details.reasoning"] == 12
    assert a["llm.cost.total"] == 0.01
    assert json.loads(a["langfuse.observation.usage_details"]) == {
        "input": 100, "output": 40, "cache_read_input_tokens": 1000, "cache_creation_input_tokens": 50}
    assert json.loads(a["langfuse.observation.cost_details"]) == {"total": 0.01}
    assert a["output.value"] == "Let me check."


def test_text_goes_only_where_content_is_on():
    e, sinks = exporter(content_phoenix=True, content_langfuse=False)
    e.export(TURN)
    local = by_name(sinks["phoenix"].get_finished_spans())
    remote = by_name(sinks["langfuse"].get_finished_spans())
    assert local["What did I save about Pune?"].attributes["input.value"] == "What did I save about Pune?"
    assert local["recall"].attributes["input.value"] == '{"query": "Pune"}'
    everything = json.dumps([dict(s.attributes) for s in remote.values()]) + " ".join(remote)
    for secret in ("Pune", "Mumbai", "Let me check", "boom"):
        assert secret not in everything
    assert remote["chat turn"].attributes["langfuse.trace.name"] == "chat turn"
    assert remote["model call 1"].attributes["llm.cost.total"] == 0.01     # numbers still go
    assert json.loads(remote["memory recall"].attributes["output.value"]) == {"retrieve": True}


def test_the_same_turn_always_gets_the_same_trace_id():
    e, sinks = exporter()
    e.export(TURN)
    e.export(TURN)
    roots = [s for s in sinks["phoenix"].get_finished_spans() if s.parent is None]
    assert len(roots) == 2 and roots[0].context.trace_id == roots[1].context.trace_id
    assert roots[0].context.trace_id == otel.ids_for("abc123")[0]
    assert roots[0].context.span_id == otel.ids_for("abc123")[1]
    first, second = sinks["phoenix"].get_finished_spans()[:4], sinks["phoenix"].get_finished_spans()[4:8]
    assert [x.context.span_id for x in first] == [x.context.span_id for x in second]   # children too
    other = dict(TURN, id="zzz")
    e.export(other)
    assert sinks["phoenix"].get_finished_spans()[-1].context.trace_id != roots[0].context.trace_id


def test_a_broken_destination_never_raises():
    class Broken(InMemorySpanExporter):
        def export(self, spans):
            raise ConnectionError("collector down")
    e = otel.OTelExporter([otel.Destination("phoenix", "http://localhost:1/v1/traces")],
                          span_exporters={"phoenix": Broken()})
    e.export(TURN)                     # logged, not raised
    e.export({"id": "x"})              # unfinished or malformed: ignored
    assert e.delivery() == {"phoenix": {"accepted": 0, "rejected": 4}}      # and counted


def test_delivery_counts_what_was_accepted():
    e, sinks = exporter()
    e.export(TURN)
    assert e.delivery()["phoenix"] == {"accepted": 4, "rejected": 0}


def test_settings_pick_destinations_and_the_content_default():
    s = Settings(home=NS(), phoenix_endpoint="http://localhost:6006/", langfuse_public_key="pk",
                 langfuse_secret_key="sk")
    phoenix, langfuse = otel.destinations(s)
    assert phoenix.endpoint == "http://localhost:6006/v1/traces" and phoenix.content
    assert langfuse.endpoint == "https://cloud.langfuse.com/api/public/otel/v1/traces" and not langfuse.content
    assert langfuse.headers["Authorization"] == "Basic cGs6c2s="          # base64("pk:sk")
    assert langfuse.headers["x-langfuse-ingestion-version"] == "4"
    s.otel_content = "1"
    assert all(d.content for d in otel.destinations(s))
    assert otel.destinations(Settings(home=NS())) == []


def test_a_real_turn_reaches_the_exporter_when_it_finishes(tmp_path):
    tracer = Tracer(tmp_path / "traces.db")
    sink = InMemorySpanExporter()
    tracer.sink = otel.OTelExporter([otel.Destination("phoenix", "http://localhost:6006/v1/traces", content=True)],
                                    span_exporters={"phoenix": sink}).export
    client = FakeClient([message(tool_use("remember", {"fact": "Aryan drinks chai"}, "c1"), stop_reason="tool_use"),
                         message(text("Saved it."))])
    agent = Sora(Settings(home=tmp_path, memory_mirror=False), client, tracer=tracer, connect_mcp=False)
    turn = agent.respond("remember I drink chai")
    spans = by_name(sink.get_finished_spans())
    spans["turn"] = next(s for s in sink.get_finished_spans() if s.parent is None)
    assert spans["turn"].name == "remember I drink chai"
    assert spans["turn"].attributes["sora.turn_id"] == turn["id"]
    assert spans["turn"].attributes["output.value"] == "Saved it."
    assert spans["remember"].attributes["openinference.span.kind"] == "TOOL"
    assert spans["model call 2"].attributes["output.value"] == "Saved it."
    assert spans["turn"].attributes["sora.cost_usd"] == pytest.approx(turn["cost_usd"])


def test_backfill_sends_finished_turns(tmp_path, monkeypatch):
    from harness.cli import main
    tracer = Tracer(tmp_path / "traces.db")
    for message in ("one", "two"):
        tracer.start(message).finish("ok")
    tracer.start("still running")
    sent = []
    monkeypatch.setattr(otel.OTelExporter, "__init__", lambda self, targets, span_exporters=None:
                        setattr(self, "targets", targets))
    monkeypatch.setattr(otel.OTelExporter, "export", lambda self, turn: sent.append(turn["message"]))
    monkeypatch.setattr(otel.OTelExporter, "close", lambda self: None)
    monkeypatch.setattr(otel.OTelExporter, "delivery", lambda self: {"phoenix": {"accepted": 8, "rejected": 0}})
    monkeypatch.setenv("SORA_PHOENIX_ENDPOINT", "http://localhost:6006")
    out = __import__("io").StringIO()
    assert main(["--home", str(tmp_path), "export", "--otel"], out=out) == 0
    assert sent == ["one", "two"] and "exported 2 turns" in out.getvalue()
    assert "phoenix (with text): 8 spans accepted" in out.getvalue()


def test_langfuse_base_url_is_read_as_the_host(tmp_path, monkeypatch):
    from harness.app import load_settings
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com")
    assert load_settings(tmp_path).langfuse_host == "https://us.cloud.langfuse.com"
