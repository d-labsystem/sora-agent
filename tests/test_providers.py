"""Providers: translating to Chat Completions and back, offline (no key, no cost)."""

import json
from types import SimpleNamespace as NS

import pytest

import loop.providers as providers
from fake import FakeClient, text, tool_use
from datetime import datetime, timezone

from harness.app import Settings, Sora, build_system
from loop.agent import run_loop
from loop.llm import ask_json, schema
from loop.providers import (NoAnswer, OpenAICompatProvider, OpenAIResponsesProvider, Providers,
                            split_model, to_openai_messages, to_openai_tools, to_responses_input,
                            to_responses_tools)
from tools.registry import WEB_TOOLS, Tool, ToolRegistry


NOW = datetime(2026, 11, 15, 9, 0, tzinfo=timezone.utc)


# ---- a fake OpenAI client that streams chunks like the real API

def chunk(content=None, calls=None, finish=None, usage=None, model="gpt-5.1-2026"):
    choices = [] if usage else [NS(delta=NS(content=content, tool_calls=calls), finish_reason=finish)]
    return NS(choices=choices, model=model, usage=usage)


def call_piece(index, id=None, name=None, args=None):
    return NS(index=index, id=id, function=NS(name=name, arguments=args))


def usage(prompt=100, completion=20, cached=0):
    return NS(prompt_tokens=prompt, completion_tokens=completion,
              prompt_tokens_details=NS(cached_tokens=cached))


class FakeOpenAI:
    """`streams` is a list of chunk lists, one per streamed call; `answers`
    are the non-streamed responses for ask_json."""

    def __init__(self, streams=(), answers=()):
        self.streams, self.answers = list(streams), list(answers)
        self.requests = []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **request):
        self.requests.append(json.loads(json.dumps(request, default=str)))
        if request.get("stream"):
            return iter(self.streams.pop(0))
        return self.answers.pop(0)


def echo_registry():
    tools = ToolRegistry(server_tools=WEB_TOOLS)
    tools.register(Tool("echo", "Echo.",
                        {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
                        lambda x: f"echo {x}"))
    return tools


# ---- choosing a provider

@pytest.mark.parametrize("spec, expected", [
    ("claude-opus-5-5", ("anthropic", "claude-opus-5-5")),
    ("openai:gpt-5.1", ("openai", "gpt-5.1")),
    ("hf:Qwen/Qwen3-32B", ("hf", "Qwen/Qwen3-32B")),
    ("hf:Qwen/Qwen3-32B:together", ("hf", "Qwen/Qwen3-32B:together")),   # only the first ':' counts
    ("local:qwen3", ("local", "qwen3")),
    ("weird:thing", ("anthropic", "weird:thing")),                        # unknown prefix
])
def test_split_model(spec, expected):
    assert split_model(spec) == expected


def test_providers_are_built_once_and_need_their_settings(monkeypatch):
    monkeypatch.setattr(providers, "_openai_client", lambda **kw: FakeOpenAI())
    p = Providers(FakeClient())
    first, model = p.get("openai:gpt-5.1")
    again, _ = p.get("openai:gpt-4.1")
    assert first is again and model == "gpt-5.1" and first.name == "openai"
    assert p.get("claude-haiku-5-5")[0].name == "anthropic"
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        p.get("hf:Qwen/Qwen3-32B")


# ---- translation

def test_tools_drop_server_tools():
    schemas = echo_registry().schemas()
    assert [t["function"]["name"] for t in to_openai_tools(schemas)] == ["echo"]


def test_messages_translate_both_ways_of_writing_blocks():
    thinking = NS(type="thinking", thinking="", signature="x")
    server = NS(type="server_tool_use", id="s1", name="web_search", input={"query": "q"})
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [thinking, server, text("Let me check."),
                                          tool_use("echo", {"x": "a"}, "call_9")]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "call_9", "content": "echo a", "is_error": False},
            {"type": "tool_result", "tool_use_id": "call_8", "content": "boom", "is_error": True}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
    ]
    system = [{"type": "text", "text": "stable", "cache_control": {"type": "ephemeral"}},
              {"type": "text", "text": "per turn"}]
    out = to_openai_messages(system, history)
    assert out[0] == {"role": "system", "content": "stable\n\nper turn"}
    assert out[1] == {"role": "user", "content": "hi"}
    assert out[2]["content"] == "Let me check."             # thinking and server blocks left out
    assert out[2]["tool_calls"] == [{"id": "call_9", "type": "function", "function": {
        "name": "echo", "arguments": '{"x": "a"}'}}]
    assert out[3] == {"role": "tool", "tool_call_id": "call_9", "content": "echo a"}
    assert out[4]["content"] == "Error: boom"
    assert out[5] == {"role": "assistant", "content": "Done."}


# ---- one streamed step

def test_step_assembles_streamed_text_and_tool_calls():
    client = FakeOpenAI(streams=[[
        chunk(content="Checking"), chunk(content="..."),
        chunk(calls=[call_piece(0, id="call_a", name="echo", args='{"x"')]),
        chunk(calls=[call_piece(0, args=': "hi"}')]),
        chunk(calls=[call_piece(1, id="call_b", name="echo", args='not json')]),
        chunk(finish="tool_calls"),
        chunk(usage=usage(prompt=120, completion=30, cached=100)),
    ]])
    streamed = []
    step = OpenAICompatProvider(client).step("gpt-5.1", "sys", [{"role": "user", "content": "go"}],
                                             echo_registry().schemas(), "high", streamed.append)
    assert streamed == ["Checking", "..."] and step.text == "Checking..."
    assert step.stop_reason == "tool_use" and step.model == "gpt-5.1-2026"
    assert [(c.id, c.input) for c in step.tool_calls] == [("call_a", {"x": "hi"}), ("call_b", {})]
    assert step.content[1] == {"type": "tool_use", "id": "call_a", "name": "echo", "input": {"x": "hi"}}
    # cached prompt tokens are counted as cache reads, not as full-price input
    assert (step.usage.input_tokens, step.usage.cache_read, step.usage.output_tokens) == (20, 100, 30)
    request = client.requests[0]
    assert request["stream_options"] == {"include_usage": True}
    assert request["max_completion_tokens"] and request["reasoning_effort"] == "high"
    assert [t["function"]["name"] for t in request["tools"]] == ["echo"]


def test_openai_only_params_are_left_out_for_other_servers():
    client = FakeOpenAI(streams=[[chunk(content="ok", finish="stop")]] * 2)
    OpenAICompatProvider(client).step("gpt-4.1", "s", [], [], "high", print)       # not a reasoning model
    OpenAICompatProvider(client, "hf", openai_params=False).step("Qwen/Qwen3-32B", "s", [], [], "high", print)
    assert "reasoning_effort" not in client.requests[0]
    assert "max_tokens" in client.requests[1] and "max_completion_tokens" not in client.requests[1]


@pytest.mark.parametrize("finish, stop", [("stop", "end_turn"), ("length", "max_tokens"),
                                          ("content_filter", "refusal")])
def test_finish_reasons_map_to_the_loop_stop_reasons(finish, stop):
    client = FakeOpenAI(streams=[[chunk(content="x", finish=finish)]])
    assert OpenAICompatProvider(client).step("gpt-5.1", "s", [], [], "low", print).stop_reason == stop


# ---- the whole loop on an OpenAI model

def test_loop_runs_tools_on_an_openai_model():
    client = FakeOpenAI(streams=[
        [chunk(calls=[call_piece(0, id="call_1", name="echo", args='{"x": "hi"}')]),
         chunk(finish="tool_calls")],
        [chunk(content="The tool said echo hi."), chunk(finish="stop"), chunk(usage=usage())],
        [chunk(content="final", finish="stop")],
    ])
    provider = OpenAICompatProvider(client)
    messages = [{"role": "user", "content": "use echo"}]
    result = run_loop(provider, "gpt-5.1", "sys", messages, echo_registry(), lambda *a, **k: None)
    assert result.reply == "The tool said echo hi." and result.iterations == 2
    assert result.tool_calls[0]["output"] == "echo hi"
    second = client.requests[1]["messages"]
    assert second[-1] == {"role": "tool", "tool_call_id": "call_1", "content": "echo hi"}
    # the final tools-off call at the step limit
    run_loop(provider, "gpt-5.1", "sys", [{"role": "user", "content": "x"}], echo_registry(),
             lambda *a, **k: None, max_iterations=0)
    assert client.requests[2]["tool_choice"] == "none"


# ---- structured answers

def test_ask_json_and_cut_off_answers():
    answer = lambda content, finish="stop": NS(
        choices=[NS(message=NS(content=content, refusal=None), finish_reason=finish)],
        usage=usage(50, 10), model="gpt-5.1")
    client = FakeOpenAI(answers=[answer('{"ok": true}'), answer('{"ok"', "length")])
    provider = OpenAICompatProvider(client)
    spans = []
    fmt = schema({"ok": {"type": "boolean"}})
    assert ask_json(provider, "gpt-5.1", "q", fmt, lambda kind, **d: spans.append((kind, d))) == {"ok": True}
    assert client.requests[0]["response_format"]["json_schema"]["schema"] == fmt["schema"]
    with pytest.raises(NoAnswer):
        ask_json(provider, "gpt-5.1", "q", fmt, lambda kind, **d: spans.append((kind, d)))
    billed = [d for kind, d in spans if kind == "llm"]
    assert [d["ok"] for d in billed] == [True, False] and billed[1]["input_tokens"] == 50


# ---- the app

def test_app_gives_each_provider_its_own_server_tools(tmp_path, monkeypatch):
    monkeypatch.setattr(providers, "_openai_client", lambda **kw: FakeOpenAI())
    monkeypatch.setenv("HF_TOKEN", "test-token")
    hf = Sora(Settings(home=tmp_path / "hf", model="hf:Qwen/Qwen3-32B", code_execution=True),
              FakeClient(), connect_mcp=False)
    assert hf.registry.server_tools == []
    assert len(hf.limits) == 2 and "cannot search the web" in hf.limits[0]
    assert "## Limits of this setup" in build_system("soul", NOW, "", [], [], None, hf.limits)[0]["text"]

    oa = Sora(Settings(home=tmp_path / "oa", model="openai:gpt-5.4-mini", code_execution=True),
              FakeClient(), connect_mcp=False)
    assert [t["type"] for t in oa.registry.server_tools] == ["web_search", "code_interpreter"]
    assert oa.limits == [] and oa.provider.max_builtin_calls == 12
    status = oa.status()["providers"]
    assert status["main"] == "openai" and status["web_tools"] and status["capabilities"]["code_execution"]

    claude = Sora(Settings(home=tmp_path / "c", code_execution=True), FakeClient(), connect_mcp=False)
    assert [t["name"] for t in claude.registry.server_tools] == ["web_search", "web_fetch", "code_execution"]


# ---- errors found in the live test

class StatusError(Exception):
    def __init__(self, status, message, body=None):
        super().__init__(message)
        self.status_code, self.message, self.body = status, message, body


def test_effort_refused_with_tools_is_retried_once_then_remembered():
    refusal = StatusError(400, "Error code: 400 - Function tools with reasoning_effort are not "
                               "supported for gpt-5.4-mini in /v1/chat/completions.")
    calls = []

    def create(**request):
        calls.append(request)
        if "reasoning_effort" in request and request.get("tools"):
            raise refusal
        return iter([chunk(content="ok", finish="stop")])

    client = NS(chat=NS(completions=NS(create=create)))
    provider = OpenAICompatProvider(client)
    tools = echo_registry().schemas()
    assert provider.step("gpt-5.4-mini", "s", [], tools, "low", print).text == "ok"
    assert len(calls) == 2 and "reasoning_effort" not in calls[1]
    provider.step("gpt-5.4-mini", "s", [], tools, "low", print)        # remembered: one call, no effort
    assert len(calls) == 3 and "reasoning_effort" not in calls[2]
    provider.step("gpt-5.4-mini", "s", [], [], "low", print)           # without tools, effort stays on
    assert calls[3]["reasoning_effort"] == "low"


def test_http_errors_say_what_to_do():
    from harness.app import explain
    payment = StatusError(402, "Error code: 402", body="Purchase pre-paid credits to continue.")

    def create(**request):
        raise payment

    provider = OpenAICompatProvider(NS(chat=NS(completions=NS(create=create))), "hf", openai_params=False)
    with pytest.raises(providers.ProviderError) as info:
        provider.step("Qwen/Qwen3-32B", "s", [], [], "low", print)
    assert explain(info.value) == "hf needs credits on the account (402): Purchase pre-paid credits to continue."
    bad_key = providers.ProviderError.wrap("openai", StatusError(401, "x", body={"error": {"message": "Incorrect API key"}}))
    assert "OPENAI_API_KEY" in explain(bad_key) and "ANTHROPIC" not in explain(bad_key)


def test_error_reason_when_the_sdk_already_unwrapped_it():
    # live Hugging Face 402: the SDK hands over body as the message string
    exc = StatusError(402, "Error code: 402", body="Purchase pre-paid credits to continue.")
    assert str(providers.ProviderError.wrap("hf", exc)).endswith("Purchase pre-paid credits to continue.")
    exc = StatusError(401, "x", body={"message": "Incorrect API key provided", "type": "invalid_request_error"})
    assert "Incorrect API key provided" in str(providers.ProviderError.wrap("openai", exc))


# ---- OpenAI Responses API

class Item(NS):
    """An output item: attributes like the SDK's, and model_dump like pydantic."""

    def model_dump(self, **kw):
        def plain(v):
            if isinstance(v, NS):
                return {k: plain(x) for k, x in vars(v).items() if x is not None}
            return [plain(x) for x in v] if isinstance(v, list) else v
        return plain(self)


def response(*items, status="completed", reason=None, inp=500, out=80, cached=0, model="gpt-5.4-mini-2026"):
    return NS(output=list(items), status=status, model=model, error=None,
              incomplete_details=NS(reason=reason) if reason else None,
              usage=NS(input_tokens=inp, output_tokens=out, input_tokens_details=NS(cached_tokens=cached)),
              output_text="".join(p.text for i in items if i.type == "message" for p in i.content
                                  if p.type == "output_text"))


def msg(text_):
    return Item(type="message", id="msg_1", role="assistant", status="completed",
                content=[Item(type="output_text", text=text_, annotations=[])])


REASONING = Item(type="reasoning", id="rs_1", summary=[], encrypted_content="gAAAA-secret")


class FakeResponses:
    def __init__(self, *responses_):
        self.queue, self.requests = list(responses_), []
        self.responses = NS(create=self._create)

    def _create(self, **request):
        self.requests.append(json.loads(json.dumps(request, default=str)))
        r = self.queue.pop(0)
        if not request.get("stream"):
            return r
        deltas = [NS(type="response.output_text.delta", delta=p.text) for i in r.output
                  if i.type == "message" for p in i.content if p.type == "output_text"]
        done = "response.incomplete" if r.status == "incomplete" else "response.completed"
        return iter([NS(type="response.created"), *deltas, NS(type=done, response=r)])


def test_responses_tools_keep_builtins_and_drop_anthropic_server_tools():
    schemas = echo_registry().schemas() + [{"type": "web_search"}, {"type": "code_execution_20260521",
                                                                   "name": "code_execution"}]
    out = to_responses_tools(schemas)
    assert out[0] == {"type": "function", "name": "echo", "description": "Echo.", "strict": False,
                      "parameters": schemas[0]["input_schema"]}
    assert [t["type"] for t in out] == ["function", "web_search"]


def test_reasoning_carries_over_tool_steps_on_the_responses_api():
    call = Item(type="function_call", id="fc_1", call_id="call_1", name="echo", arguments='{"x": "hi"}',
                status="completed")
    client = FakeResponses(response(REASONING, call, cached=400),
                           response(msg("It said echo hi.")))
    provider = OpenAIResponsesProvider(client)
    messages = [{"role": "user", "content": "use echo"}]
    result = run_loop(provider, "gpt-5.4-mini", [{"type": "text", "text": "sys"}], messages,
                      echo_registry(), lambda *a, **k: None, effort="medium")
    assert result.reply == "It said echo hi." and result.tool_calls[0]["output"] == "echo hi"
    first, second = client.requests
    assert first["reasoning"] == {"effort": "medium"} and first["store"] is False
    assert "reasoning.encrypted_content" in first["include"] and first["instructions"] == "sys"
    # the second step sends back the encrypted reasoning and the call, then the result
    sent = second["input"]
    assert sent[1] == {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "gAAAA-secret"}
    assert sent[2]["type"] == "function_call" and sent[2]["call_id"] == "call_1"
    assert sent[3] == {"type": "function_call_output", "call_id": "call_1", "output": "echo hi"}
    assert (result.input_tokens, result.cache_read_tokens) == (600, 400)   # 100 + 500 uncached


def test_web_search_and_code_show_up_as_server_tools():
    search = Item(type="web_search_call", id="ws_1", status="completed",
                  action=Item(type="search", query="vllm vs sglang 3090"))
    page = Item(type="web_search_call", id="ws_2", status="completed",
                action=Item(type="open_page", url="https://docs.vllm.ai"))
    code = Item(type="code_interpreter_call", id="ci_1", code="print(17*23)", container_id="cntr_1",
                status="completed", outputs=[Item(type="logs", logs="391\n")])
    client = FakeResponses(response(REASONING, search, page, code, msg("391, and vLLM.")))
    step = OpenAIResponsesProvider(client).step("gpt-5.4-mini", "s", [{"role": "user", "content": "q"}],
                                                [{"type": "web_search"}], "low", print)
    assert step.stop_reason == "end_turn" and step.usage.web_searches == 1
    assert step.usage.tool_fees_usd == providers.CODE_SESSION_USD
    assert step.server_events == [
        ("server_tool", {"name": "web_search", "input": {"query": "vllm vs sglang 3090"}}),
        ("server_tool", {"name": "web_fetch", "input": {"url": "https://docs.vllm.ai"}}),
        ("server_tool", {"name": "code_execution", "input": {"code": "print(17*23)"}}),
        ("server_result", {"name": "code_interpreter_result", "logs": "391\n"})]
    assert step.content[0]["type"] == "openai_items" and len(step.content[0]["items"]) == 5


def test_responses_stop_reasons_and_final_call():
    refusal = Item(type="message", id="m", role="assistant", status="completed",
                   content=[Item(type="refusal", refusal="no")])
    client = FakeResponses(response(msg("cut"), status="incomplete", reason="max_output_tokens"),
                           response(refusal), response(msg("ok"), model="gpt-4.1"))
    p = OpenAIResponsesProvider(client)
    assert p.step("gpt-5.4-mini", "s", [], [], "low", print).stop_reason == "max_tokens"
    assert p.step("gpt-5.4-mini", "s", [], [], "low", print).stop_reason == "refusal"
    p.step("gpt-4.1", "s", [], echo_registry().schemas(), "low", print, final=True)
    last = client.requests[2]
    assert last["tool_choice"] == "none" and "reasoning" not in last   # gpt-4.1 isn't a reasoning model


def test_history_from_other_providers_translates_to_responses_input():
    history = [{"role": "user", "content": "hi"},
               {"role": "assistant", "content": [text("Checking."), tool_use("echo", {"x": "a"}, "c9")]},
               {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c9",
                                             "content": "boom", "is_error": True}]}]
    assert to_responses_input(history) == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Checking."},
        {"type": "function_call", "call_id": "c9", "name": "echo", "arguments": '{"x": "a"}'},
        {"type": "function_call_output", "call_id": "c9", "output": "Error: boom"}]


def test_responses_ask_json():
    client = FakeResponses(response(msg('{"ok": true}')), response(msg('{"o'), status="incomplete",
                                                                    reason="max_output_tokens"))
    p = OpenAIResponsesProvider(client)
    fmt = schema({"ok": {"type": "boolean"}})
    assert ask_json(p, "gpt-5.4-mini", "q", fmt) == {"ok": True}
    assert client.requests[0]["text"]["format"]["schema"] == fmt["schema"]
    with pytest.raises(NoAnswer):
        ask_json(p, "gpt-5.4-mini", "q", fmt)
