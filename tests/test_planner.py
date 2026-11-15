"""The planner: the checklist tool, the completion check, and when planning is asked for."""

import pytest

from fake import FakeClient, message, text, tool_use
from loop import agent as loop_agent
from loop.agent import run_loop
from tools.planner import NUDGE, PLAN_NOTE, Plan
from tools.registry import ToolError, ToolRegistry

GATE_SIMPLE = {"retrieve": False, "query": "", "correction": False, "complex": False, "reason": "chat"}
GATE_COMPLEX = {**GATE_SIMPLE, "complex": True, "reason": "several parts"}


def steps(*statuses):
    return [{"title": f"step {i}", "status": s, **({"note": "no access"} if s == "blocked" else {})}
            for i, s in enumerate(statuses, 1)]


def plan_registry(plan):
    registry = ToolRegistry()
    registry.register(plan.tool())
    return registry


def test_the_checklist_is_validated_and_traced():
    seen = []
    plan = Plan()
    plan.reset(lambda kind, **d: seen.append((kind, d)))
    out = plan.update(steps("done", "in_progress", "pending"))
    assert "[x] 1. step 1\n[~] 2. step 2\n[ ] 3. step 3" in out and "2 step(s) to go" in out
    assert seen[-1][0] == "plan" and seen[-1][1]["open"] == 2
    for bad, problem in [([], "non-empty"), (steps("in_progress", "in_progress"), "only one"),
                         ([{"title": "x", "status": "later"}], "status must be"),
                         ([{"title": "x", "status": "blocked"}], "add a note"),
                         ([{"title": " ", "status": "done"}], "needs a title"),
                         (steps(*["pending"] * 13), "at most")]:
        with pytest.raises(ToolError, match=problem):
            plan.update(bad)
    assert "All steps closed" in plan.update(steps("done", "blocked"))
    assert plan.open_steps() == []


def test_finishing_with_open_steps_sends_the_model_back_once():
    seen = []
    notify = lambda kind, **d: seen.append((kind, d))
    plan = Plan()
    plan.reset(notify)                     # the harness passes the turn's notify, as here
    client = FakeClient([
        message(tool_use("update_plan", {"steps": steps("in_progress", "pending")}, "c1"), stop_reason="tool_use"),
        message(text("Here's step 1.")),                                   # tries to stop early
        message(tool_use("update_plan", {"steps": steps("done", "done")}, "c2"), stop_reason="tool_use"),
        message(text("Both steps done."))])
    result = run_loop(client, "m", "s", [{"role": "user", "content": "do two things"}], plan_registry(plan),
                      notify, plan=plan)
    assert result.reply == "Both steps done." and result.plan_nudged
    nudge = client.requests[2]["messages"][-1]
    assert nudge == {"role": "user", "content": NUDGE.format(steps="1. step 1; 2. step 2")}
    assert [d["name"] for k, d in seen if k == "plan"] == ["plan", "completion check", "plan"]


def test_the_nudge_is_never_a_wall():
    plan = Plan()
    plan.reset()
    plan.update(steps("pending"))
    client = FakeClient([message(text("I can't do that.")), message(text("Still can't, sorry."))])
    result = run_loop(client, "m", "s", [{"role": "user", "content": "x"}], plan_registry(plan),
                      lambda *a, **k: None, plan=plan)
    assert result.reply == "Still can't, sorry." and len(client.requests) == 2      # one nudge, then accept


def test_no_plan_or_a_finished_plan_changes_nothing():
    for plan in (None, Plan()):
        client = FakeClient([message(text("hi"))])
        result = run_loop(client, "m", "s", [{"role": "user", "content": "hi"}], ToolRegistry(),
                          lambda *a, **k: None, plan=plan)
        assert result.reply == "hi" and not result.plan_nudged and len(client.requests) == 1


def test_compaction_keeps_the_checklist(monkeypatch):
    seen = {}

    def fake_compact(provider, model, system, messages, schemas, effort, task, result, notify, size):
        seen["task"] = task
        return False
    monkeypatch.setattr(loop_agent, "_compact", fake_compact)
    plan = Plan()
    plan.reset()
    big = lambda *blocks, **kw: message(*blocks, **kw)
    client = FakeClient([big(tool_use("update_plan", {"steps": steps("done", "pending")}, "c1"), stop_reason="tool_use"),
                         message(tool_use("update_plan", {"steps": steps("done", "done")}, "c2"), stop_reason="tool_use"),
                         message(text("done"))])
    run_loop(client, "m", "s", [{"role": "user", "content": "the task"}], plan_registry(plan),
             lambda *a, **k: None, compact_at=1, plan=plan)
    assert seen["task"].startswith("the task") and "[x] 1. step 1\n[ ] 2. step 2" in seen["task"]


# ---- in the agent: only complex turns are asked to plan, and the cache prefix never changes

def agent_with(tmp_path, gate, script):
    from harness.app import Settings, Sora
    client = FakeClient(script, small=gate)
    return Sora(Settings(home=tmp_path, memory_mirror=False), client, connect_mcp=False), client


def test_complex_turns_are_asked_to_plan_and_simple_ones_are_not(tmp_path):
    agent, client = agent_with(tmp_path, [GATE_SIMPLE, GATE_COMPLEX],
                               [message(text("hi")), message(text("planned answer"))])
    agent.respond("hello")
    agent.respond("research three laptops and save a note")
    simple, complex_ = client.requests
    assert PLAN_NOTE not in simple["system"][1]["text"] and PLAN_NOTE in complex_["system"][1]["text"]
    assert simple["tools"] == complex_["tools"] and simple["system"][0] == complex_["system"][0]
    assert any(t["name"] == "update_plan" for t in simple["tools"])


def test_long_tasks_always_plan(tmp_path):
    agent, client = agent_with(tmp_path, GATE_SIMPLE, [message(text("ok"))])
    agent.respond("go", long=True)
    assert PLAN_NOTE in client.requests[0]["system"][1]["text"]
