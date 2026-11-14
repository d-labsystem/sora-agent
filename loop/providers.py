"""Model providers: one interface in front of Anthropic, OpenAI and Hugging Face.

The loop and the small helper calls (gate, consolidation, reflection, judge)
talk to a Provider, never to an SDK directly. A provider does two things:

    step()      one model call with tools, text streamed as it is generated
    ask_json()  one call that must answer with JSON in a given shape

Choosing a provider. A model name may carry a prefix:

    claude-opus-5-5          Anthropic (no prefix)
    openai:gpt-5.1           OpenAI
    hf:Qwen/Qwen3-32B        Hugging Face Inference Providers
    local:my-model           any OpenAI-compatible server (vLLM, SGLang, TGI...)

One message format inside. Conversations are kept in Anthropic's format,
because it is the richest: thinking blocks, server tool results and
compaction blocks all have a place in it. The OpenAI-compatible provider
translates that format to Chat Completions on every call and translates the
answer back, so the loop never knows which provider ran. Blocks with no
Chat Completions equivalent (thinking, server tool results) are left out of
the translation; they only exist in conversations Anthropic produced.

What only Anthropic has: server-side web search and fetch, prompt caching
breakpoints, refusal fallbacks and thinking. `server_tools` says whether a
provider can run web_search/web_fetch; the app leaves them out otherwise.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Top-level automatic caching: the API puts a breakpoint on the last block of
# each request, so every step of the loop reads the steps before it from cache.
AUTO_CACHE = {"type": "ephemeral"}
MAX_TOKENS = 64000
HF_BASE_URL = "https://router.huggingface.co/v1"

OnText = Callable[[str], None]


@dataclass
class Usage:
    input_tokens: int = 0          # not served from cache
    output_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0
    web_searches: int = 0


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class Step:
    """One model call's answer, the same shape for every provider."""
    content: list                  # assistant content to append, in Anthropic's format
    text: str
    tool_calls: list[ToolCall]
    stop_reason: str               # end_turn | tool_use | max_tokens | refusal | pause_turn
    model: str                     # the model that actually answered
    usage: Usage
    server_events: list[tuple[str, dict]] = field(default_factory=list)   # for the trace


class Provider(Protocol):
    name: str
    server_tools: bool

    def step(self, model: str, system, messages: list, tools: list, effort: str,
             on_text: OnText, final: bool = False) -> Step: ...

    def ask_json(self, model: str, prompt: str, output_format: dict,
                 max_tokens: int) -> tuple[dict, Usage, str]: ...


def _get(block, key: str, default=None):
    """Blocks are SDK objects (from a response) or dicts (written by us)."""
    return block.get(key, default) if isinstance(block, dict) else getattr(block, key, default)


# ---- Anthropic

def _server_results(block) -> dict[str, Any]:
    """A short, traceable summary of what a server tool returned."""
    content = getattr(block, "content", None)
    if isinstance(content, list):
        return {"results": [{"title": getattr(r, "title", ""), "url": getattr(r, "url", "")}
                            for r in content][:10]}
    error = getattr(content, "error_code", None)
    if error:
        return {"error": error}
    url = getattr(content, "url", None)
    return {"url": url} if url else {}


class AnthropicProvider:
    name = "anthropic"
    server_tools = True

    def __init__(self, client):
        self.client = client

    def step(self, model, system, messages, tools, effort, on_text, final=False) -> Step:
        extra = {"tool_choice": {"type": "none"}} if final else {}
        with self.client.beta.messages.stream(
            model=model, max_tokens=MAX_TOKENS, system=system, messages=messages,
            tools=tools, output_config={"effort": effort},
            betas=[FALLBACK_BETA], fallbacks="default",   # a declined request is retried on another model
            cache_control=AUTO_CACHE, **extra,
        ) as stream:
            for delta in stream.text_stream:
                on_text(delta)
            response = stream.get_final_message()

        u = response.usage
        usage = Usage(u.input_tokens, u.output_tokens,
                      getattr(u, "cache_read_input_tokens", 0) or 0,
                      getattr(u, "cache_creation_input_tokens", 0) or 0,
                      getattr(getattr(u, "server_tool_use", None), "web_search_requests", 0) or 0)
        events = []
        for block in response.content:
            if block.type == "server_tool_use":
                events.append(("server_tool", {"name": block.name, "input": block.input}))
            elif block.type.endswith("_tool_result") and block.type != "tool_result":
                events.append(("server_result", {"name": block.type, **_server_results(block)}))
        return Step(
            content=response.content,
            text="".join(b.text for b in response.content if b.type == "text"),
            tool_calls=[ToolCall(b.id, b.name, b.input) for b in response.content
                        if b.type == "tool_use"],
            stop_reason=response.stop_reason,
            model=getattr(response, "model", "") or model,
            usage=usage, server_events=events)

    def ask_json(self, model, prompt, output_format, max_tokens):
        response = self.client.messages.create(
            model=model, max_tokens=max_tokens,
            output_config={"effort": "low", "format": output_format},
            messages=[{"role": "user", "content": prompt}])
        usage = Usage(response.usage.input_tokens, response.usage.output_tokens)
        if response.stop_reason in ("refusal", "max_tokens"):
            raise NoAnswer(response.stop_reason, usage)
        text = "".join(b.text for b in response.content if b.type == "text")
        return json.loads(text), usage, model


class NoAnswer(ValueError):
    """The model stopped without a usable answer; the usage is still billed."""

    def __init__(self, reason: str, usage: Usage):
        super().__init__(f"no usable answer ({reason})")
        self.usage = usage


# ---- OpenAI-compatible (OpenAI, Hugging Face, local servers)

STOP_REASONS = {"stop": "end_turn", "tool_calls": "tool_use", "length": "max_tokens",
                "content_filter": "refusal", "function_call": "tool_use"}


def to_openai_tools(tools: list[dict]) -> list[dict]:
    """Our tool schemas -> Chat Completions functions. Server tools (web_search,
    web_fetch: dicts with a `type`) only run on Anthropic, so they are dropped."""
    return [{"type": "function", "function": {
                "name": t["name"], "description": t.get("description", ""),
                "parameters": t.get("input_schema") or {"type": "object", "properties": {}}}}
            for t in tools if "input_schema" in t]


def _system_text(system) -> str:
    if isinstance(system, str):
        return system
    return "\n\n".join(_get(b, "text", "") for b in system or [])


def _result_text(content) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(_get(b, "text", "") for b in content or [] if _get(b, "type") == "text")


def to_openai_messages(system, messages: list) -> list[dict]:
    """Anthropic-format conversation -> Chat Completions messages.

        user text            {"role": "user"}
        user tool_result     one {"role": "tool", "tool_call_id"} per result
        assistant text       {"role": "assistant", "content"}
        assistant tool_use   {"role": "assistant", "tool_calls": [...]}
        thinking, server tool blocks, compaction   no equivalent: left out
    """
    out: list[dict] = []
    text = _system_text(system)
    if text:
        out.append({"role": "system", "content": text})
    for message in messages:
        role, content = message["role"], message["content"]
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        if role == "assistant":
            texts, calls = [], []
            for block in content:
                kind = _get(block, "type")
                if kind == "text":
                    texts.append(_get(block, "text", ""))
                elif kind == "tool_use":
                    calls.append({"id": _get(block, "id"), "type": "function", "function": {
                        "name": _get(block, "name"),
                        "arguments": json.dumps(_get(block, "input") or {})}})
            entry: dict[str, Any] = {"role": "assistant", "content": "".join(texts) or None}
            if calls:
                entry["tool_calls"] = calls
            if entry["content"] or calls:
                out.append(entry)
            continue
        # user or system with a list of blocks
        texts = []
        for block in content:
            kind = _get(block, "type")
            if kind == "tool_result":
                result = _result_text(_get(block, "content"))
                if _get(block, "is_error"):
                    result = f"Error: {result}"
                out.append({"role": "tool", "tool_call_id": _get(block, "tool_use_id"),
                            "content": result})
            elif kind == "text":
                texts.append(_get(block, "text", ""))
        if texts:
            out.append({"role": role, "content": "\n".join(texts)})
    return out


def _parse_args(raw: str) -> dict:
    """Streamed tool arguments arrive as JSON text in pieces. Bad or missing
    JSON becomes {}, and the registry then reports which arguments are missing."""
    try:
        args = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return args if isinstance(args, dict) else {}


def _usage(u) -> Usage:
    if u is None:
        return Usage()
    cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
    return Usage(input_tokens=(u.prompt_tokens or 0) - cached,
                 output_tokens=u.completion_tokens or 0, cache_read=cached)


class OpenAICompatProvider:
    """Chat Completions, which OpenAI, Hugging Face's router and local servers
    (vLLM, SGLang, TGI) all speak. `openai_params` turns on the two parameters
    only OpenAI's own API accepts: max_completion_tokens and reasoning_effort."""

    server_tools = False

    def __init__(self, client, name: str = "openai", openai_params: bool = True):
        self.client = client
        self.name = name
        self.openai_params = openai_params

    def _limits(self, model: str, effort: str | None, max_tokens: int) -> dict:
        if not self.openai_params:
            return {"max_tokens": max_tokens}
        params: dict[str, Any] = {"max_completion_tokens": max_tokens}
        # Only reasoning models take an effort; sending it to others is an error.
        if effort and (model.startswith(("gpt-5", "o1", "o3", "o4"))):
            params["reasoning_effort"] = effort
        return params

    def step(self, model, system, messages, tools, effort, on_text, final=False) -> Step:
        functions = to_openai_tools(tools)
        request: dict[str, Any] = {
            "model": model, "messages": to_openai_messages(system, messages),
            "stream": True, "stream_options": {"include_usage": True},
            **self._limits(model, effort, 32000)}
        if functions:
            request["tools"] = functions
            if final:
                request["tool_choice"] = "none"

        text, finish, served, usage = [], None, model, None
        slots: dict[int, dict[str, str]] = {}     # tool calls stream in pieces, by index
        for chunk in self.client.chat.completions.create(**request):
            served = getattr(chunk, "model", None) or served
            if getattr(chunk, "usage", None):
                usage = chunk.usage                # the last chunk, no choices
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            if getattr(delta, "content", None):
                text.append(delta.content)
                on_text(delta.content)
            for piece in getattr(delta, "tool_calls", None) or []:
                slot = slots.setdefault(piece.index, {"id": "", "name": "", "args": ""})
                if piece.id:
                    slot["id"] = piece.id
                if piece.function is not None:
                    slot["name"] += piece.function.name or ""
                    slot["args"] += piece.function.arguments or ""
            if choice.finish_reason:
                finish = choice.finish_reason

        calls = [ToolCall(s["id"] or f"call_{uuid.uuid4().hex[:12]}", s["name"], _parse_args(s["args"]))
                 for _, s in sorted(slots.items()) if s["name"]]
        reply = "".join(text)
        content: list[dict] = [{"type": "text", "text": reply}] if reply else []
        content += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.input} for c in calls]
        stop = STOP_REASONS.get(finish or "stop", "end_turn")
        if calls and stop == "end_turn":
            stop = "tool_use"                       # some servers say "stop" with tool calls
        return Step(content=content, text=reply, tool_calls=calls if stop == "tool_use" else [],
                    stop_reason=stop, model=served, usage=_usage(usage))

    def ask_json(self, model, prompt, output_format, max_tokens):
        response = self.client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_schema", "json_schema": {
                "name": "answer", "schema": output_format["schema"], "strict": True}},
            **self._limits(model, "low", max_tokens))
        usage = _usage(response.usage)
        choice = response.choices[0]
        if choice.finish_reason == "length" or getattr(choice.message, "refusal", None):
            raise NoAnswer("max_tokens" if choice.finish_reason == "length" else "refusal", usage)
        return json.loads(choice.message.content or ""), usage, getattr(response, "model", model)


# ---- choosing one

PREFIXES = ("openai", "hf", "local")


def split_model(spec: str) -> tuple[str, str]:
    """'openai:gpt-5.1' -> ('openai', 'gpt-5.1'); no known prefix -> Anthropic.
    Hugging Face ids contain '/' and sometimes ':' (a provider suffix), so only
    the first ':' counts, and only when it names a known provider."""
    head, sep, rest = spec.partition(":")
    if sep and head in PREFIXES and rest:
        return head, rest
    return "anthropic", spec


def _openai_client(**kwargs):
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("OpenAI-compatible models need the openai package: "
                           "uv sync --extra providers") from None
    return OpenAI(**kwargs)


def build(kind: str, anthropic_client=None) -> Provider:
    if kind == "anthropic":
        if anthropic_client is None:
            import anthropic
            anthropic_client = anthropic.Anthropic()
        return AnthropicProvider(anthropic_client)
    if kind == "openai":
        return OpenAICompatProvider(_openai_client(), "openai")
    if kind == "hf":
        token = os.getenv("HF_TOKEN")
        if not token:
            raise RuntimeError("Hugging Face models need HF_TOKEN in .env")
        return OpenAICompatProvider(_openai_client(base_url=os.getenv("SORA_HF_BASE_URL", HF_BASE_URL),
                                                   api_key=token), "hf", openai_params=False)
    if kind == "local":
        base = os.getenv("SORA_LOCAL_BASE_URL")
        if not base:
            raise RuntimeError("local: models need SORA_LOCAL_BASE_URL, e.g. http://127.0.0.1:8000/v1")
        return OpenAICompatProvider(_openai_client(base_url=base,
                                                   api_key=os.getenv("SORA_LOCAL_API_KEY", "none")),
                                    "local", openai_params=False)
    raise ValueError(f"unknown provider {kind!r}")


class Providers:
    """One provider per kind, built on first use. `anthropic_client` lets the
    app (and the tests) supply the Anthropic client."""

    def __init__(self, anthropic_client=None):
        self.anthropic_client = anthropic_client
        self.built: dict[str, Provider] = {}

    def get(self, spec: str) -> tuple[Provider, str]:
        kind, model = split_model(spec)
        if kind not in self.built:
            self.built[kind] = build(kind, self.anthropic_client)
        return self.built[kind], model


def as_provider(client_or_provider) -> Provider:
    """Accept a Provider, or a bare Anthropic client (older callers, tests)."""
    if hasattr(client_or_provider, "step") and hasattr(client_or_provider, "ask_json"):
        return client_or_provider
    return AnthropicProvider(client_or_provider)
