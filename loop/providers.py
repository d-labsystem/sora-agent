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

Three adapters:

    AnthropicProvider        Messages API (the native format)
    OpenAIResponsesProvider  openai: models, Responses API: reasoning stays on
                             during tool use, built-in web search and code
    OpenAICompatProvider     hf: and local: models, Chat Completions

One message format inside. Conversations are kept in Anthropic's format,
because it is the richest: thinking blocks, server tool results and
compaction blocks all have a place in it. The OpenAI adapters translate that
format on every call and translate the answer back, so the loop never knows
which provider ran. What a format can't express is left out of the
translation; OpenAI's own output items (encrypted reasoning, web search
calls) travel inside one "openai_items" block that only the Responses
adapter reads.

Capabilities. Each provider says what it can do (CAPABILITIES). The app
gives the model only the server tools its provider can run, and tells it in
one line what it can't do, so it says "I can't search" instead of pretending.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from tools.registry import web_tools

FALLBACK_BETA = "server-side-fallback-2026-07-01"
COMPACT_BETA = "compact-2026-09-04"     # on-demand compaction (loop/compaction.py)
# Top-level automatic caching: the API puts a breakpoint on the last block of
# each request, so every step of the loop reads the steps before it from cache.
AUTO_CACHE = {"type": "ephemeral"}
MAX_TOKENS = 64000      # the API's ceiling for one reply; sora's own cap is Settings.max_output_tokens
HF_BASE_URL = "https://router.huggingface.co/v1"

OnText = Callable[[str], None]


@dataclass
class Usage:
    input_tokens: int = 0          # not served from cache
    output_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0
    web_searches: int = 0
    tool_fees_usd: float = 0.0     # per-use fees beyond tokens and searches (code containers)
    reasoning_tokens: int = 0      # part of output_tokens, when the provider reports it


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
    files: list[str] = field(default_factory=list)   # ids of files server-side code saved for the user


CAPABILITIES = ("reasoning_with_tools", "web_search", "web_fetch", "code_execution",
                "prompt_cache", "refusal_fallback")


class Provider(Protocol):
    name: str
    capabilities: dict[str, bool]

    def server_tool_specs(self, web: dict | None, code: bool) -> list[dict]: ...

    def step(self, model: str, system, messages: list, tools: list, effort: str,
             on_text: OnText, final: bool = False) -> Step: ...

    def ask_json(self, model: str, prompt: str, output_format: dict,
                 max_tokens: int) -> tuple[dict, Usage, str]: ...


def _get(block, key: str, default=None):
    """Blocks are SDK objects (from a response) or dicts (written by us)."""
    return block.get(key, default) if isinstance(block, dict) else getattr(block, key, default)


# ---- Anthropic

SERVER_TEXT = 3000          # characters of a server tool's output kept for the trace (and the judge)


def _server_results(block) -> dict[str, Any]:
    """What a server tool returned, as evidence for the trace and the judge:
    search hits (title, URL), the printed output of server-side code (with
    today's web tools, the model filters search results in code and reads
    that output), and the start of a fetched page's text. Outputs the
    provider encrypts can't be read by anyone but the model; that's noted."""
    content = getattr(block, "content", None)
    if isinstance(content, list):                                      # web search hits
        return {"results": [{"title": getattr(r, "title", ""), "url": getattr(r, "url", ""),
                             **({"age": r.page_age} if getattr(r, "page_age", None) else {})}
                            for r in content][:10]}
    error = getattr(content, "error_code", None)
    if error:
        return {"error": error}
    out: dict[str, Any] = {}
    if hasattr(content, "stdout") or hasattr(content, "encrypted_stdout"):   # code execution
        stdout = getattr(content, "stdout", None)
        if stdout:
            out["stdout"] = stdout[:SERVER_TEXT]
        elif getattr(content, "encrypted_stdout", None):
            out["stdout"] = "(output encrypted by the provider: only the model could read it)"
        if getattr(content, "stderr", ""):
            out["stderr"] = content.stderr[:500]
        return out
    url = getattr(content, "url", None)                                 # web fetch
    if url:
        out["url"] = url
        source = getattr(getattr(content, "content", None), "source", None)
        data = getattr(source, "data", None)
        if isinstance(data, str) and getattr(source, "type", "") == "text":
            out["text"] = data[:SERVER_TEXT]
        elif source is not None:
            out["text"] = f"({getattr(source, 'media_type', 'a document')}, not text)"
    return out


def _output_files(block) -> list[str]:
    """Ids of the files a server code run saved to $OUTPUT_DIR. They live on
    the provider's servers until downloaded (download_file); the result block
    only points at them, like "content": [{"type": "...output", "file_id": "file_…"}]."""
    items = getattr(getattr(block, "content", None), "content", None)
    if not isinstance(items, list):
        return []
    return [i.file_id for i in items if getattr(i, "file_id", None)]


ANTHROPIC_CODE_TOOL = {"type": "code_execution_20260521", "name": "code_execution"}


def _has_compaction(messages: list) -> bool:
    """A compacted conversation starts with the compaction block."""
    if not messages or isinstance(messages[0]["content"], str):
        return False
    return any(_get(b, "type") == "compaction" for b in messages[0]["content"])


class AnthropicProvider:
    name = "anthropic"
    capabilities = dict.fromkeys(CAPABILITIES, True)
    max_output = MAX_TOKENS          # per call; the harness lowers it (a runaway reply can't cost more)
    # Cache the conversation so the next step reads it at 1/10 the price. A cache write costs
    # 1.25x, so a prompt sent once only pays more: LongMemEval full-context (2026-11-17) wrote
    # 167k tokens to the cache per question and never read them, $0.17 of each $0.84.
    cache_conversation = True

    def __init__(self, client):
        self.client = client

    def server_tool_specs(self, web, code):
        """`web` holds the search/fetch limits, or None for no web tools."""
        return (web_tools(**web) if web else []) + ([dict(ANTHROPIC_CODE_TOOL)] if code else [])

    def step(self, model, system, messages, tools, effort, on_text, final=False) -> Step:
        extra = {"tool_choice": {"type": "none"}} if final else {}
        betas = [FALLBACK_BETA]
        if _has_compaction(messages):
            betas.append(COMPACT_BETA)        # needed on every request that carries the signed block
        with self.client.beta.messages.stream(
            model=model, max_tokens=self.max_output, system=system, messages=messages,
            tools=tools, output_config={"effort": effort},
            betas=betas, fallbacks="default",   # a declined request is retried on another model
            **({"cache_control": AUTO_CACHE} if self.cache_conversation else {}), **extra,
        ) as stream:
            for delta in stream.text_stream:
                on_text(delta)
            response = stream.get_final_message()

        u = response.usage
        usage = Usage(u.input_tokens, u.output_tokens,
                      getattr(u, "cache_read_input_tokens", 0) or 0,
                      getattr(u, "cache_creation_input_tokens", 0) or 0,
                      getattr(getattr(u, "server_tool_use", None), "web_search_requests", 0) or 0)
        events, files = [], []
        for block in response.content:
            if block.type == "server_tool_use":
                events.append(("server_tool", {"name": block.name, "input": block.input}))
            elif block.type.endswith("_tool_result") and block.type != "tool_result":
                found = _output_files(block)
                files += found
                events.append(("server_result", {"name": block.type, **_server_results(block),
                                                 **({"files": found} if found else {})}))
        return Step(
            content=response.content,
            text="".join(b.text for b in response.content if b.type == "text"),
            tool_calls=[ToolCall(b.id, b.name, b.input) for b in response.content
                        if b.type == "tool_use"],
            stop_reason=response.stop_reason,
            model=getattr(response, "model", "") or model,
            usage=usage, server_events=events, files=files)

    def download_file(self, file_id: str) -> tuple[str, bytes]:
        """(file name, bytes) of a file server-side code saved (Files API)."""
        name = self.client.files.retrieve_metadata(file_id).filename
        return name, self.client.files.download(file_id).read()

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


KEY_NAMES = {"openai": "OPENAI_API_KEY", "hf": "HF_TOKEN", "local": "SORA_LOCAL_API_KEY"}


class ProviderError(RuntimeError):
    """A provider's HTTP error, with the reason a person can act on."""

    @classmethod
    def wrap(cls, provider: str, exc: Exception) -> Exception:
        status = getattr(exc, "status_code", None)
        if status is None:
            return exc                                  # network errors etc. pass through
        # The SDK unwraps {"error": ...}: body is then the message string itself
        # (Hugging Face) or a dict with a "message" (OpenAI).
        body = getattr(exc, "body", None)
        detail = body if isinstance(body, str) else None
        if isinstance(body, dict):
            detail = body.get("message") or body.get("error")
            if isinstance(detail, dict):
                detail = detail.get("message")
        detail = detail or getattr(exc, "message", "") or str(exc)
        if status == 401:
            return cls(f"{provider} rejected the key in {KEY_NAMES.get(provider, 'its key')}: {detail}")
        if status == 402:
            return cls(f"{provider} needs credits on the account (402): {detail}")
        if status == 429:
            return cls(f"{provider} rate limit or quota (429): {detail}")
        return cls(f"{provider} returned {status}: {detail}")


def _effort_with_tools_refused(exc: Exception) -> bool:
    """Newer OpenAI models accept reasoning OR function tools on Chat
    Completions, not both (400: 'Function tools with reasoning_effort are not
    supported ... use /v1/responses'). Found in a live test on 2026-11-15."""
    return getattr(exc, "status_code", None) == 400 and "reasoning_effort" in str(exc)


class OpenAICompatProvider:
    max_output = 32000
    """Chat Completions, which OpenAI, Hugging Face's router and local servers
    (vLLM, SGLang, TGI) all speak. `openai_params` turns on the two parameters
    only OpenAI's own API accepts: max_completion_tokens and reasoning_effort.

    Reasoning and tools: some OpenAI models refuse reasoning_effort when tools
    are sent. The first refusal is remembered per model, and from then on that
    model runs tool steps without reasoning (effort "none"). The Responses API
    would keep reasoning on with tools; that is a separate adapter."""

    capabilities = {**dict.fromkeys(CAPABILITIES, False)}

    def __init__(self, client, name: str = "openai", openai_params: bool = True):
        self.client = client
        self.name = name
        self.openai_params = openai_params
        self.no_effort_with_tools: set[str] = set()

    def server_tool_specs(self, web, code):
        return []                                  # Chat Completions has no server tools

    def _limits(self, model: str, effort: str | None, max_tokens: int, tools: bool = False) -> dict:
        if not self.openai_params:
            return {"max_tokens": max_tokens}
        params: dict[str, Any] = {"max_completion_tokens": max_tokens}
        # Only reasoning models take an effort; sending it to others is an error.
        if effort and model.startswith(("gpt-5", "o1", "o3", "o4")) \
                and not (tools and model in self.no_effort_with_tools):
            params["reasoning_effort"] = effort
        return params

    def _create(self, request: dict):
        try:
            return self.client.chat.completions.create(**request)
        except Exception as exc:
            if "reasoning_effort" in request and request.get("tools") and _effort_with_tools_refused(exc):
                self.no_effort_with_tools.add(request["model"])
                request = {k: v for k, v in request.items() if k != "reasoning_effort"}
                try:
                    return self.client.chat.completions.create(**request)
                except Exception as retry_exc:
                    raise ProviderError.wrap(self.name, retry_exc) from retry_exc
            raise ProviderError.wrap(self.name, exc) from exc

    def step(self, model, system, messages, tools, effort, on_text, final=False) -> Step:
        functions = to_openai_tools(tools)
        request: dict[str, Any] = {
            "model": model, "messages": to_openai_messages(system, messages),
            "stream": True, "stream_options": {"include_usage": True},
            **self._limits(model, effort, self.max_output, tools=bool(functions))}
        if functions:
            request["tools"] = functions
            if final:
                request["tool_choice"] = "none"

        text, finish, served, usage = [], None, model, None
        slots: dict[int, dict[str, str]] = {}     # tool calls stream in pieces, by index
        for chunk in self._create(request):
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
        response = self._create(dict(
            model=model, messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_schema", "json_schema": {
                "name": "answer", "schema": output_format["schema"], "strict": True}},
            **self._limits(model, "low", max_tokens)))
        usage = _usage(response.usage)
        choice = response.choices[0]
        if choice.finish_reason == "length" or getattr(choice.message, "refusal", None):
            raise NoAnswer("max_tokens" if choice.finish_reason == "length" else "refusal", usage)
        return json.loads(choice.message.content or ""), usage, getattr(response, "model", model)


# ---- OpenAI Responses API

OPENAI_BUILTIN = ("web_search", "code_interpreter")
CODE_SESSION_USD = float(os.getenv("SORA_OPENAI_CODE_SESSION_USD", "0.03"))   # list price per container


def is_reasoning(model: str) -> bool:
    return model.startswith(("gpt-5", "o1", "o3", "o4"))


def to_responses_tools(tools: list[dict]) -> list[dict]:
    """Function tools in Responses shape, plus OpenAI's own built-in tools.
    Anthropic server tools (web_search_2026..., code_execution_2026...) are dropped."""
    out = []
    for t in tools:
        if "input_schema" in t:
            out.append({"type": "function", "name": t["name"], "description": t.get("description", ""),
                        "parameters": t["input_schema"], "strict": False})
        elif t.get("type") in OPENAI_BUILTIN:
            out.append(t)
    return out


def to_responses_input(messages: list) -> list[dict]:
    """Anthropic-format conversation -> Responses input items.

        assistant turn made by OpenAI   its saved output items, as they were
                                        (encrypted reasoning, searches, calls)
        other assistant text/tool_use   message / function_call items
        user tool_result                function_call_output items
    """
    out: list[dict] = []
    for message in messages:
        role, content = message["role"], message["content"]
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        saved = [b for b in content if _get(b, "type") == "openai_items"]
        if role == "assistant" and saved:
            for block in saved:
                out.extend(_get(block, "items") or [])
            continue
        texts: list[str] = []

        def flush():                       # keep text and calls in the order they were said
            if texts:
                out.append({"role": role, "content": "\n".join(texts)})
                texts.clear()

        for block in content:
            kind = _get(block, "type")
            if kind == "text":
                texts.append(_get(block, "text", ""))
            elif kind == "tool_use" and role == "assistant":
                flush()
                out.append({"type": "function_call", "call_id": _get(block, "id"), "name": _get(block, "name"),
                            "arguments": json.dumps(_get(block, "input") or {})})
            elif kind == "tool_result":
                flush()
                result = _result_text(_get(block, "content"))
                out.append({"type": "function_call_output", "call_id": _get(block, "tool_use_id"),
                            "output": f"Error: {result}" if _get(block, "is_error") else result})
        flush()
    return out


def _search_event(action) -> tuple[str, dict]:
    kind = _get(action, "type")
    if kind == "open_page":
        return "server_tool", {"name": "web_fetch", "input": {"url": _get(action, "url")}}
    if kind == "find":
        return "server_tool", {"name": "web_fetch", "input": {"url": _get(action, "url"),
                                                              "find": _get(action, "pattern")}}
    query = _get(action, "query") or "; ".join(_get(action, "queries") or [])
    return "server_tool", {"name": "web_search", "input": {"query": query}}


class OpenAIResponsesProvider:
    max_output = 32000
    """OpenAI's Responses API. Compared with Chat Completions it keeps the
    model's reasoning during tool use and runs web search and code on
    OpenAI's side, like Anthropic's server tools.

    Nothing is stored at OpenAI (store=False). The model's reasoning comes
    back encrypted and is sent back on the next step, so it carries on from
    where it was without OpenAI keeping the conversation."""

    name = "openai"
    capabilities = {"reasoning_with_tools": True, "web_search": True, "web_fetch": True,
                    "code_execution": True, "prompt_cache": True, "refusal_fallback": False}

    def __init__(self, client):
        self.client = client
        self.max_builtin_calls = 0

    def server_tool_specs(self, web, code):
        specs = []
        if web:
            specs.append({"type": "web_search"})
            # One cap for all built-in tool calls per response (searches + page opens).
            self.max_builtin_calls = (web.get("max_searches") or 0) + (web.get("max_fetches") or 0)
        if code:
            specs.append({"type": "code_interpreter", "container": {"type": "auto"}})
        return specs

    def _create(self, request: dict):
        try:
            return self.client.responses.create(**request)
        except Exception as exc:
            raise ProviderError.wrap(self.name, exc) from exc

    def step(self, model, system, messages, tools, effort, on_text, final=False) -> Step:
        functions = to_responses_tools(tools)
        request: dict[str, Any] = {
            "model": model, "input": to_responses_input(messages), "stream": True,
            "store": False, "max_output_tokens": self.max_output, "prompt_cache_key": f"sora-{model}"}
        instructions = _system_text(system)
        if instructions:
            request["instructions"] = instructions
        include = []
        if is_reasoning(model):
            request["reasoning"] = {"effort": effort}
            include.append("reasoning.encrypted_content")
        if functions:
            request["tools"] = functions
            if final:
                request["tool_choice"] = "none"
            if any(t["type"] == "code_interpreter" for t in functions):
                include.append("code_interpreter_call.outputs")
            if self.max_builtin_calls and any(t["type"] in OPENAI_BUILTIN for t in functions):
                request["max_tool_calls"] = self.max_builtin_calls
        if include:
            request["include"] = include

        response = None
        for event in self._create(request):
            kind = getattr(event, "type", "")
            if kind == "response.output_text.delta":
                on_text(event.delta)
            elif kind in ("response.completed", "response.incomplete", "response.failed"):
                response = event.response
            elif kind == "error":
                raise ProviderError(f"openai stream error: {getattr(event, 'message', event)}")
        if response is None:
            raise ProviderError("openai stream ended without a response")
        if response.status == "failed":
            raise ProviderError(f"openai response failed: {getattr(response.error, 'message', response.error)}")
        return self._to_step(response, model)

    @staticmethod
    def _to_step(response, model: str) -> Step:
        text, calls, events, refused = [], [], [], False
        searches, containers = 0, set()
        for item in response.output:
            kind = item.type
            if kind == "message":
                for part in item.content:
                    if part.type == "output_text":
                        text.append(part.text)
                    elif part.type == "refusal":
                        refused = True
            elif kind == "function_call":
                calls.append(ToolCall(item.call_id, item.name, _parse_args(item.arguments)))
            elif kind == "web_search_call":
                event = _search_event(item.action)
                searches += event[1]["name"] == "web_search"
                events.append(event)
            elif kind == "code_interpreter_call":
                containers.add(item.container_id)
                events.append(("server_tool", {"name": "code_execution", "input": {"code": item.code}}))
                logs = "".join(_get(o, "logs", "") or "" for o in item.outputs or [])
                events.append(("server_result", {"name": "code_interpreter_result", "logs": logs[:2000]}))

        reply = "".join(text)
        reason = getattr(response.incomplete_details, "reason", None) if response.status == "incomplete" else None
        stop = ("refusal" if refused or reason == "content_filter" else
                "max_tokens" if reason == "max_output_tokens" else
                "tool_use" if calls else "end_turn")
        u = response.usage
        cached = getattr(getattr(u, "input_tokens_details", None), "cached_tokens", 0) or 0
        usage = Usage(input_tokens=(u.input_tokens if u else 0) - cached,
                      output_tokens=u.output_tokens if u else 0, cache_read=cached,
                      web_searches=searches, tool_fees_usd=len(containers) * CODE_SESSION_USD,
                      reasoning_tokens=getattr(getattr(u, "output_tokens_details", None),
                                               "reasoning_tokens", 0) or 0)
        items = [item.model_dump(mode="json", exclude_none=True) for item in response.output]
        content: list[dict] = [{"type": "openai_items", "items": items}]
        if reply:
            content.append({"type": "text", "text": reply})
        content += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.input} for c in calls]
        return Step(content=content, text=reply, tool_calls=calls if stop == "tool_use" else [],
                    stop_reason=stop, model=getattr(response, "model", "") or model,
                    usage=usage, server_events=events)

    def ask_json(self, model, prompt, output_format, max_tokens):
        request: dict[str, Any] = {
            "model": model, "input": [{"role": "user", "content": prompt}], "store": False,
            "max_output_tokens": max_tokens,
            "text": {"format": {"type": "json_schema", "name": "answer",
                                "schema": output_format["schema"], "strict": True}}}
        if is_reasoning(model):
            request["reasoning"] = {"effort": "low"}
        response = self._create(request)
        u = response.usage
        cached = getattr(getattr(u, "input_tokens_details", None), "cached_tokens", 0) or 0
        usage = Usage((u.input_tokens if u else 0) - cached, u.output_tokens if u else 0, cache_read=cached)
        if response.status == "incomplete":
            reason = getattr(response.incomplete_details, "reason", "")
            raise NoAnswer("refusal" if reason == "content_filter" else "max_tokens", usage)
        refused = any(p.type == "refusal" for item in response.output if item.type == "message"
                      for p in item.content)
        if refused:
            raise NoAnswer("refusal", usage)
        return json.loads(response.output_text or ""), usage, getattr(response, "model", model)


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


# How long one API call may wait before it is retried. The SDKs' default read timeout is
# 600 s: when the Mac slept mid-call (LongMemEval, 2026-11-16, 00:20), five calls sat on
# dead connections and the run stalled until morning, since each would be noticed only
# after 10 minutes, and retried just twice. 120 s is far above a normal gap: streamed
# replies send pings while the server works, and small JSON calls finish in under a minute.
API_TIMEOUT = float(os.getenv("SORA_API_TIMEOUT", "120"))
API_RETRIES = int(os.getenv("SORA_API_RETRIES", "4"))   # with the SDK's backoff between tries


def api_timeout(sdk, read: float | None = None):
    """`sdk.Timeout`: each SDK takes its own httpx's Timeout (anthropic uses httpx2)."""
    read = API_TIMEOUT if read is None else read
    return sdk.Timeout(read, connect=min(10.0, read))


def new_anthropic_client(timeout: float | None = None, max_retries: int | None = None, **kwargs):
    """The Anthropic client sora uses everywhere: short timeouts, more retries."""
    import anthropic
    return anthropic.Anthropic(timeout=api_timeout(anthropic, timeout),
                               max_retries=API_RETRIES if max_retries is None else max_retries, **kwargs)


def _openai_client(**kwargs):
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("OpenAI-compatible models need the openai package: "
                           "uv sync --extra providers") from None
    import openai
    kwargs.setdefault("timeout", api_timeout(openai))
    kwargs.setdefault("max_retries", API_RETRIES)
    return OpenAI(**kwargs)


def build(kind: str, anthropic_client=None) -> Provider:
    if kind == "anthropic":
        if anthropic_client is None:
            anthropic_client = new_anthropic_client()
        return AnthropicProvider(anthropic_client)
    if kind == "openai":
        return OpenAIResponsesProvider(_openai_client())
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
