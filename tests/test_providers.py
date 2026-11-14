"""Providers: translating to Chat Completions and back, offline (no key, no cost)."""

import json
from types import SimpleNamespace as NS

import pytest

import loop.providers as providers
from fake import FakeClient, text, tool_use
from harness.app import Settings, Sora
from loop.agent import run_loop
from loop.llm import ask_json, schema
from loop.providers import (NoAnswer, OpenAICompatProvider, Providers, split_model,
                            to_openai_messages, to_openai_tools)
from tools.registry import WEB_TOOLS, Tool, ToolRegistry


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

def test_app_leaves_out_web_tools_for_providers_without_them(tmp_path, monkeypatch):
    monkeypatch.setattr(providers, "_openai_client", lambda **kw: FakeOpenAI())
    agent = Sora(Settings(home=tmp_path, model="openai:gpt-5.1"), FakeClient(), connect_mcp=False)
    names = [t["name"] for t in agent.registry.describe()]
    assert "web_search" not in names and "remember" in names
    status = agent.status()["providers"]
    assert status == {"main": "openai", "small": "anthropic", "web_tools": False,
                      "prices_known": ["claude-haiku-5-5"]}
