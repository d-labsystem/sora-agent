"""The loop and the tool registry, against a scripted model (no key, no cost)."""

import pytest

from fake import FakeClient, message, text, tool_use
from loop.agent import REFUSED, TRUNCATED, run_loop
from tools.registry import WEB_TOOLS, Tool, ToolError, ToolRegistry, invalid_args


def quiet(*args, **kwargs):
    pass


def registry():
    tools = ToolRegistry(server_tools=WEB_TOOLS)
    tools.register(Tool("echo", "Echo.",
                        {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
                        lambda x: f"echo {x}"))
    return tools


def loop(client, tools=None, **kwargs):
    return run_loop(client, "claude-opus-5-5", "sys", [{"role": "user", "content": "hi"}],
                    tools or registry(), quiet, **kwargs)


def test_stops_when_the_model_asks_for_no_tools():
    result = loop(FakeClient([message(text("hello"))]))
    assert (result.reply, result.iterations) == ("hello", 1)


def test_tool_result_goes_back_to_the_model():
    client = FakeClient([message(tool_use("echo", {"x": "a"}), stop_reason="tool_use"),
                         message(text("done"))])
    result = loop(client)
    assert result.reply == "done"
    sent = client.requests[1]["messages"][-1]["content"][0]
    assert (sent["tool_use_id"], sent["content"], sent["is_error"]) == ("call_1", "echo a", False)


def test_parallel_calls_return_in_one_message():
    client = FakeClient([message(tool_use("echo", {"x": "a"}, "1"), tool_use("echo", {"x": "b"}, "2"),
                                 stop_reason="tool_use"),
                         message(text("done"))])
    loop(client)
    assert [r["content"] for r in client.requests[1]["messages"][-1]["content"]] == ["echo a", "echo b"]


def test_iteration_limit_makes_one_tools_off_call():
    calling = message(tool_use("echo", {"x": "a"}), stop_reason="tool_use")
    client = FakeClient([calling, calling, message(text("partial answer"))])
    result = loop(client, max_iterations=2)
    assert (result.reply, result.limit_reached) == ("partial answer", True)
    assert client.requests[-1]["tool_choice"] == {"type": "none"}


def test_paused_server_search_is_resumed_without_a_new_user_message():
    client = FakeClient([message(text("searching"), stop_reason="pause_turn"), message(text("found it"))])
    result = loop(client)
    assert result.reply == "found it"
    assert client.requests[1]["messages"][-1]["role"] == "assistant"


def test_refusal_is_not_read_as_content():
    assert loop(FakeClient([message(stop_reason="refusal")])).reply == REFUSED


def test_truncated_tool_call_never_runs():
    ran = []
    tools = ToolRegistry()
    tools.register(Tool("t", "T.", {"type": "object", "properties": {}}, lambda: ran.append(1) or ""))
    result = loop(FakeClient([message(text("so"), tool_use("t", {}), stop_reason="max_tokens")]), tools)
    assert ran == [] and result.reply == "so" + TRUNCATED


def test_cost_is_counted_from_tokens():
    result = loop(FakeClient([message(text("hi"))]))
    assert result.cost_usd == pytest.approx((10 * 4 + 5 * 20) / 1_000_000)


def test_web_tools_are_sent_and_lessons_reach_tool_descriptions():
    client = FakeClient([message(text("hi"))])
    run_loop(client, "m", "s", [{"role": "user", "content": "hi"}], registry(), quiet,
             tool_notes={"echo": ["Pass plain text only."]})
    tools = client.requests[0]["tools"]
    assert tools[0]["description"].endswith("- Pass plain text only.")
    assert {t["name"] for t in tools[1:]} == {"web_search", "web_fetch"}
    assert "eager_input_streaming" not in tools[1]


@pytest.mark.parametrize("args, problem", [
    ({}, "missing x"), ({"x": 1}, "x must be a string"), ("nope", "arguments must be an object")])
def test_bad_tool_input_is_refused(args, problem):
    assert invalid_args(registry().get("echo").input_schema, args) == problem


def test_tool_failures_reach_the_model_as_errors():
    tools = registry()

    def careful():
        raise ToolError("Folder not found.")
    tools.register(Tool("careful", "C.", {"type": "object", "properties": {}}, careful))
    tools.register(Tool("boom", "B.", {"type": "object", "properties": {}}, lambda: 1 / 0))
    assert tools.execute("missing", {}) == ("Unknown tool: missing", True)
    assert tools.execute("careful", {}) == ("Folder not found.", True)
    assert tools.execute("boom", {}) == ("ZeroDivisionError: division by zero", True)
