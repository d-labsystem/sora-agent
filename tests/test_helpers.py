"""Parallel helpers: concurrency, what helpers may touch, caps, and tracing."""

import json
import threading
import time

import pytest

from fake import FakeClient, message, text, tool_use
from loop.helpers import HELPER_SYSTEM, Helpers
from loop.providers import AnthropicProvider
from tools.approvals import Approvals
from tools.builtin import TurnContext
from tools.permissions import Permissions
from tools.registry import Tool, ToolError, ToolRegistry


def registry_with(tmp_path, rules=None):
    calls = []
    reg = ToolRegistry([{"type": "web_search_20260209", "name": "web_search"}])
    tools = {"workspace__read_file": "read", "workspace__write_file": "local", "remember": "local",
             "run_code": "local", "browser__open": "read"}
    for name, kind in tools.items():
        reg.register(Tool(name, name, {"type": "object", "properties": {}},
                          lambda _n=name, **a: calls.append(_n) or f"{_n} ok", kind=kind))
    path = tmp_path / "permissions.json"
    path.write_text(json.dumps({"rules": rules or [{"tool": "*", "action": "default"}]}))
    gate = Approvals(TurnContext("s", "t"), Permissions(path))
    gate.guard(reg)
    return reg, calls


def answer_by_task(delay=0.0, steps=None):
    """A fake model: each helper's reply depends on its task; optional delay per call."""
    def script(request):
        time.sleep(delay)
        task = request["messages"][0]["content"]
        if steps and task.startswith("use tools") and len(request["messages"]) == 1:
            return message(*[tool_use(name, {}, f"c{i}") for i, name in enumerate(steps)], stop_reason="tool_use")
        return message(text(f"Report on: {task.splitlines()[0]}"))
    return script


def helpers(tmp_path, script, **kw):
    reg, calls = registry_with(tmp_path, kw.pop("rules", None))
    client = FakeClient(script)
    h = Helpers(AnthropicProvider(client), "claude-opus-5-5", reg, **kw)
    return h, client, calls


def test_tasks_run_at_the_same_time(tmp_path):
    h, client, _ = helpers(tmp_path, answer_by_task(delay=0.4))
    started = time.perf_counter()
    out = h.delegate([{"task": f"Laptop {i}"} for i in range(4)])
    took = time.perf_counter() - started
    assert took < 1.2, took                                    # 4 x 0.4 s sequential would be 1.6 s+
    assert [line for line in out.splitlines() if line.startswith("## Helper")] == \
        [f"## Helper {i}: Laptop {i - 1}" for i in range(1, 5)]
    assert all(f"Report on: Laptop {i}" in out for i in range(4))
    assert all(r["system"] == HELPER_SYSTEM for r in client.requests)


def test_helpers_see_only_research_tools(tmp_path):
    h, client, _ = helpers(tmp_path, answer_by_task())
    h.delegate([{"task": "x"}])
    names = {t.get("name") for t in client.requests[0]["tools"]}
    assert names == {"workspace__read_file", "run_code", "web_search"}
    assert not names & {"workspace__write_file", "remember", "browser__open", "delegate"}


def test_asks_are_refused_inside_helpers_but_allowed_tools_run(tmp_path):
    h, client, calls = helpers(tmp_path, answer_by_task(steps=["workspace__read_file", "run_code"]),
                               rules=[{"tool": "run_code", "action": "ask"}, {"tool": "*", "action": "default"}])
    seen = []
    h.notify = lambda kind, **d: seen.append((kind, d))
    h.delegate([{"task": "use tools please"}])
    results = client.requests[1]["messages"][-1]["content"]
    by_id = {r["tool_use_id"]: r for r in results}
    assert by_id["c0"]["content"] == "workspace__read_file ok" and not by_id["c0"]["is_error"]
    assert by_id["c1"]["is_error"] and "Helpers can't ask the user" in by_id["c1"]["content"]
    assert calls == ["workspace__read_file"]                  # run_code never ran
    names = [d["name"] for k, d in seen]
    assert "helper 1 · started" in names and "helper 1 · workspace__read_file" in names
    assert "helper 1 · model call 1" in names and all(k != "text" for k, _ in seen)   # no streaming into chat


def test_limits_failures_and_bad_input(tmp_path):
    h, _, _ = helpers(tmp_path, answer_by_task(), max_parallel=2)
    with pytest.raises(ToolError, match="at most 2"):
        h.delegate([{"task": "a"}, {"task": "b"}, {"task": "c"}])
    with pytest.raises(ToolError, match="needs a 'task'"):
        h.delegate([{"task": " "}])
    with pytest.raises(ToolError, match="non-empty"):
        h.delegate([])

    def flaky(request):
        if "boom" in request["messages"][0]["content"]:
            raise RuntimeError("network down")
        return message(text("fine"))
    h2, _, _ = helpers(tmp_path, flaky)
    out = h2.delegate([{"task": "boom"}, {"task": "ok"}])
    assert "Failed: RuntimeError: network down" in out and "## Helper 2: ok\nfine" in out


def test_a_helper_stops_at_its_step_limit(tmp_path):
    def forever(request):                                       # always wants another tool call
        return message(tool_use("workspace__read_file", {}, f"c{len(request['messages'])}"), stop_reason="tool_use")
    h, client, _ = helpers(tmp_path, forever, max_steps=3)
    out = h.delegate([{"task": "dig"}])
    assert "stopped at its step limit" in out and len(client.requests) == 4     # 3 steps + the final answer call


def test_in_the_agent_helper_costs_count_and_the_tool_is_offered(tmp_path):
    from harness.app import Settings, Sora
    gate = {"retrieve": False, "query": "", "correction": False, "complex": False, "reason": "q"}
    lock = threading.Lock()
    state = {"main": 0}

    def script(request):
        if request["system"] == HELPER_SYSTEM:
            return message(text("helper report"))
        with lock:
            state["main"] += 1
            first = state["main"] == 1
        if first:
            return message(tool_use("delegate", {"tasks": [{"task": "a"}, {"task": "b"}]}, "d1"), stop_reason="tool_use")
        return message(text("combined answer"))
    agent = Sora(Settings(home=tmp_path, memory_mirror=False), FakeClient(script, small=gate), connect_mcp=False)
    turn = agent.respond("compare a and b")
    names = [s["name"] for s in turn["spans"]]
    assert "helper 1 · model call 1" in names and "helper 2 · model call 1" in names
    llm_cost = sum(s["data"].get("cost_usd", 0) for s in turn["spans"] if s["kind"] == "llm")
    assert turn["cost_usd"] == pytest.approx(llm_cost, abs=1e-6) and turn["reply"] == "combined answer"   # stored to 6 places
    bare = Sora(Settings(home=tmp_path / "bare", memory_mirror=False, bare=True), FakeClient([]), connect_mcp=False)
    assert bare.registry.get("delegate") is None
