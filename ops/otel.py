"""OpenTelemetry export: each finished turn, copied to Phoenix and Langfuse.

traces.db stays the source of truth and the dashboard reads only it. This
module sends a copy of every finished turn as an OpenTelemetry trace, so
dedicated trace tools can slice them: Phoenix (Arize, runs locally) and
Langfuse (cloud or self-hosted). Both receive traces over OTLP/HTTP.

Why after the turn, not live. Turn.notify writes a span when its step ends,
with its duration; OpenTelemetry lets a span be created with an explicit
start and end time. So when the turn finishes we rebuild the whole tree with
exact timings and the turn's totals on the root:

    turn                 AGENT    session, source, status, cost
    ├── retrieval gate   LLM      model, tokens, cost
    ├── memory recall    CHAIN
    ├── model call 1     LLM
    ├── read_file        TOOL     args -> output
    └── consolidation    CHAIN

Two vocabularies on every span, because each tool reads a different one:
    OpenInference   openinference.span.kind, input.value, llm.token_count.*,
                    llm.cost.total ...                       Phoenix
    langfuse.*      langfuse.observation.type, usage_details, cost_details,
                    and it also reads input.value / output.value / session.id
sora's own cost goes on each llm span, so all three views show the same dollars.

Content (messages, replies, tool arguments and output, memory text) is sent
only where `content` is on: by default to local endpoints (localhost) and
not to remote ones, so a cloud service gets timings, tokens and costs but
never your conversations unless SORA_OTEL_CONTENT=1.

Sending runs on a background thread (BatchSpanProcessor): a collector that
is slow or down never delays a reply. Failures are logged and dropped.

The trace id is derived from the turn id, so a turn exported twice (say by
a backfill) is the same trace, and Langfuse updates it instead of copying.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import urlparse

log = logging.getLogger(__name__)

MAX_TEXT = 8000         # characters per attribute value
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]"}

KINDS = {   # sora span kind -> (OpenInference kind, Langfuse observation type)
    "llm": ("LLM", "generation"),
    "tool": ("TOOL", "tool"),
    "gate": ("RETRIEVER", "retriever"),
}
CHAIN = ("CHAIN", "span")


@dataclass
class Destination:
    name: str
    endpoint: str                       # the full OTLP traces URL
    headers: dict = field(default_factory=dict)
    content: bool = False               # send message and reply text


def is_local(url: str) -> bool:
    return (urlparse(url).hostname or "") in LOCAL_HOSTS


def destinations(settings) -> list[Destination]:
    """The destinations configured in settings; empty means export is off."""
    found = []
    override = {"1": True, "0": False}.get(settings.otel_content.strip())
    if settings.phoenix_endpoint:
        url = settings.phoenix_endpoint.rstrip("/") + "/v1/traces"
        headers = {"Authorization": f"Bearer {settings.phoenix_api_key}"} if settings.phoenix_api_key else {}
        found.append(Destination("phoenix", url, headers,
                                 override if override is not None else is_local(url)))
    if settings.langfuse_public_key and settings.langfuse_secret_key:
        url = settings.langfuse_host.rstrip("/") + "/api/public/otel/v1/traces"
        auth = base64.b64encode(f"{settings.langfuse_public_key}:{settings.langfuse_secret_key}".encode()).decode()
        found.append(Destination("langfuse", url,
                                 {"Authorization": f"Basic {auth}", "x-langfuse-ingestion-version": "4"},
                                 override if override is not None else is_local(url)))
    return found


def ids_for(turn_id: str) -> tuple[int, int]:
    """A stable 128-bit trace id and 64-bit root span id for a turn."""
    digest = hashlib.sha256(turn_id.encode()).digest()
    return int.from_bytes(digest[:16], "big") or 1, int.from_bytes(digest[16:24], "big") or 1


def span_id_for(turn_id: str, index: int) -> int:
    """A stable id for the turn's index-th child span, so exporting a turn
    again replaces its spans instead of adding copies."""
    return int.from_bytes(hashlib.sha256(f"{turn_id}/{index}".encode()).digest()[:8], "big") or 1


def to_ns(iso: str) -> int:
    return int(datetime.fromisoformat(iso).timestamp() * 1e9)


def text(value) -> str:
    s = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return s[:MAX_TEXT]


def provider_of(model: str) -> str:
    m = (model or "").lower()
    return "openai" if m.startswith(("gpt", "o1", "o3", "o4")) else "anthropic" if m.startswith("claude") else "other"


def llm_attributes(d: dict) -> dict:
    """Tokens and cost in both vocabularies. sora counts uncached input
    apart from cache reads and writes; OpenInference's prompt count is the
    total, with the cached parts as details."""
    fresh, out = d.get("input_tokens", 0) or 0, d.get("output_tokens", 0) or 0
    read, write = d.get("cache_read_tokens", d.get("cache_read", 0)) or 0, \
        d.get("cache_write_tokens", d.get("cache_write", 0)) or 0
    model = d.get("model") or ""
    attrs = {"llm.model_name": model, "llm.provider": provider_of(model),
             "llm.token_count.prompt": fresh + read + write, "llm.token_count.completion": out,
             "llm.token_count.total": fresh + read + write + out,
             "llm.token_count.prompt_details.cache_read": read,
             "llm.token_count.prompt_details.cache_write": write,
             "llm.cost.total": float(d.get("cost_usd") or 0.0),
             # Phoenix ignores client-sent llm.cost.* and prices tokens from its own
             # table (Settings > Models); sora's figure stays visible here regardless.
             "sora.cost_usd": float(d.get("cost_usd") or 0.0),
             "langfuse.observation.model.name": model,
             "langfuse.observation.usage_details": json.dumps(
                 {"input": fresh, "output": out, "cache_read_input_tokens": read,
                  "cache_creation_input_tokens": write}),
             "langfuse.observation.cost_details": json.dumps({"total": float(d.get("cost_usd") or 0.0)})}
    if d.get("reasoning_tokens"):
        attrs["llm.token_count.completion_details.reasoning"] = d["reasoning_tokens"]
    if d.get("stop_reason"):
        attrs["sora.stop_reason"] = d["stop_reason"]
    return attrs


def numbers_only(d: dict) -> dict:
    """Without content, a span keeps only its counts and flags."""
    return {k: v for k, v in d.items() if isinstance(v, (int, float, bool)) and not isinstance(v, str)}


def span_attributes(span: dict, content: bool) -> dict:
    kind, d = span["kind"], dict(span.get("data") or {})
    oi, lf = KINDS.get(kind, CHAIN)
    attrs = {"openinference.span.kind": oi, "langfuse.observation.type": lf, "sora.kind": kind}
    if kind == "llm":
        attrs.update(llm_attributes(d))
        if content and d.get("output"):
            attrs["output.value"] = text(d["output"])
    elif kind == "tool":
        attrs["tool.name"] = span.get("name") or ""
        if content:
            attrs["input.value"] = text(d.get("args", {}))
            attrs["input.mime_type"] = "application/json"
            attrs["output.value"] = text(d.get("output", ""))
    else:
        details = d if content else numbers_only(d)
        if details:
            attrs["output.value"] = text(details)
            attrs["output.mime_type"] = "application/json"
    if d.get("error") and content:
        attrs["sora.error"] = text(d["error"])
    return attrs


def counting(inner):
    """Wraps a span exporter to count what the collector accepted and
    rejected. Sending happens on a background thread, so without this a
    rejected upload (say HTTP 401) would be invisible: handing a span to
    the exporter is not delivering it."""
    from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

    class Counting(SpanExporter):
        def __init__(self):
            self.inner, self.accepted, self.rejected = inner, 0, 0

        def export(self, spans):
            try:
                result = self.inner.export(spans)
            except Exception as exc:
                log.warning("OTel export failed: %s", exc)
                result = SpanExportResult.FAILURE
            if result == SpanExportResult.SUCCESS:
                self.accepted += len(spans)
            else:
                self.rejected += len(spans)
            return result

        def shutdown(self):
            self.inner.shutdown()

        def force_flush(self, timeout_millis: int = 30000) -> bool:
            return self.inner.force_flush(timeout_millis)
    return Counting()


def _turn_ids():
    """An IdGenerator whose next trace id is the current turn's, so every
    export of a turn lands on the same trace."""
    from opentelemetry.sdk.trace.id_generator import RandomIdGenerator

    class TurnIds(RandomIdGenerator):
        def __init__(self):
            self.local = threading.local()

        def generate_trace_id(self) -> int:
            return getattr(self.local, "trace_id", None) or super().generate_trace_id()

        def generate_span_id(self) -> int:
            preset = getattr(self.local, "span_id", None)
            if preset is not None:
                self.local.span_id = None
                return preset
            return super().generate_span_id()

        def is_trace_id_random(self) -> bool:
            return False                     # derived from the turn id, not random
    return TurnIds()


class OTelExporter:
    """Sends finished turns to every destination. `span_exporters` replaces
    the OTLP exporters (tests pass in-memory ones) and makes sending
    synchronous."""

    def __init__(self, targets: list[Destination], span_exporters: dict | None = None):
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
        self.targets = targets
        self.ids = _turn_ids()
        self.lock = threading.Lock()
        self.providers = []
        resource = Resource.create({"service.name": "sora", "openinference.project.name": "sora"})
        self.counters = {}
        for target in targets:
            if span_exporters is not None:
                self.counters[target.name] = counting(span_exporters[target.name])
                processor = SimpleSpanProcessor(self.counters[target.name])
            else:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
                self.counters[target.name] = counting(
                    OTLPSpanExporter(endpoint=target.endpoint, headers=target.headers, timeout=10))
                processor = BatchSpanProcessor(self.counters[target.name])
            # One provider per destination: content can differ between them.
            provider = TracerProvider(resource=resource, id_generator=self.ids)
            provider.add_span_processor(processor)
            self.providers.append((target, provider, provider.get_tracer("sora")))

    def export(self, turn: dict) -> None:
        """Copy one finished turn. Never raises: tracing must not break a turn."""
        if not turn or not turn.get("started_at"):
            return
        for target, _, tracer in self.providers:
            try:
                with self.lock:
                    self._send(tracer, turn, target.content)
            except Exception as exc:
                log.warning("OTel export to %s failed for turn %s: %s", target.name, turn.get("id"), exc)

    def _send(self, tracer, turn: dict, content: bool) -> None:
        from opentelemetry import trace
        from opentelemetry.trace import Status, StatusCode
        start = to_ns(turn["started_at"])
        end = to_ns(turn["ended_at"]) if turn.get("ended_at") else start + int((turn.get("latency_ms") or 0) * 1e6)
        self.ids.local.trace_id, self.ids.local.span_id = ids_for(turn["id"])
        try:
            title = (turn.get("message") or "")[:60] if content else f"{turn.get('source', 'chat')} turn"
            root = tracer.start_span(title or "turn", start_time=start, attributes={
                "openinference.span.kind": "AGENT", "langfuse.observation.type": "agent",
                "langfuse.trace.name": title or "turn",
                "session.id": turn.get("session_id") or "", "sora.turn_id": turn["id"],
                "sora.source": turn.get("source") or "", "sora.status": turn.get("status") or "",
                "sora.model": turn.get("model") or "", "sora.iterations": turn.get("iterations") or 0,
                "sora.cost_usd": float(turn.get("cost_usd") or 0), "sora.latency_ms": turn.get("latency_ms") or 0,
                "langfuse.trace.metadata.source": turn.get("source") or "",
                "langfuse.trace.metadata.status": turn.get("status") or "",
                "langfuse.trace.metadata.cost_usd": float(turn.get("cost_usd") or 0),
                **({"langfuse.trace.metadata.eval_run": turn["eval_run"], "sora.eval_run": turn["eval_run"]}
                   if turn.get("eval_run") else {}),
                **({"input.value": text(turn.get("message") or ""), "output.value": text(turn.get("reply") or "")}
                   if content else {}),
                **({"sora.checks": json.dumps(turn["checks"])} if turn.get("checks") else {}),
            })
        finally:
            # Children inherit the trace from the root; each gets a preset span id.
            self.ids.local.trace_id = self.ids.local.span_id = None
        parent = trace.set_span_in_context(root)
        for index, s in enumerate(turn.get("spans") or []):
            s_start = start + int(s.get("start_ms", 0) * 1e6)
            self.ids.local.span_id = span_id_for(turn["id"], index)
            child = tracer.start_span(s.get("name") or s["kind"], context=parent, start_time=s_start,
                                      attributes=span_attributes(s, content))
            if not s.get("ok", True):
                child.set_status(Status(StatusCode.ERROR, str((s.get("data") or {}).get("error", ""))[:200]
                                        if content else None))
            else:
                child.set_status(Status(StatusCode.OK))
            child.end(end_time=max(s_start, s_start + int(s.get("duration_ms", 0) * 1e6)))
        if turn.get("status") == "error":
            root.set_status(Status(StatusCode.ERROR, (turn.get("error") or "")[:200] if content else None))
        else:
            root.set_status(Status(StatusCode.OK))
        root.end(end_time=max(start, end))

    def delivery(self) -> dict:
        """Spans each collector accepted and rejected so far."""
        return {name: {"accepted": c.accepted, "rejected": c.rejected} for name, c in self.counters.items()}

    def flush(self, timeout_ms: int = 5000) -> None:
        for _, provider, _ in self.providers:
            try:
                provider.force_flush(timeout_ms)
            except Exception as exc:
                log.warning("OTel flush failed: %s", exc)

    def close(self) -> None:
        for _, provider, _ in self.providers:
            try:
                provider.shutdown()
            except Exception as exc:
                log.warning("OTel shutdown failed: %s", exc)


def from_settings(settings) -> OTelExporter | None:
    targets = destinations(settings)
    if not targets:
        return None
    try:
        exporter = OTelExporter(targets)
    except ImportError as exc:
        log.error("OTel export is configured but its packages are missing (uv sync --extra otel): %s", exc)
        return None
    import atexit
    atexit.register(exporter.close)          # send what's queued when sora exits
    return exporter
