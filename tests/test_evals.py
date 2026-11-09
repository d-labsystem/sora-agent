"""Evals: checks, judge, datasets, the runner and the release gate."""

import json
from pathlib import Path

import pytest

from fake import FakeClient
from memory.store import MemoryStore
from ops.evals import (append_case, case_checks, judge, load_cases, main, run_eval,
                       trace_to_case, turn_checks)
from ops.tracing import Tracer

ROOT = Path(__file__).parent.parent


def fake_turn(**over):
    turn = {"id": "t1", "message": "hi", "reply": "Priya lives in Pune.", "status": "ok",
            "tool_errors": 0, "cost_usd": 0.01, "spans": [
                {"kind": "gate", "name": "memory recall", "data": {"retrieve": True, "correction": False}},
                {"kind": "tool", "name": "search_memory", "data": {}},
                {"kind": "server_tool", "name": "web_search", "data": {}}]}
    return {**turn, **over}


def test_turn_checks():
    assert all(turn_checks(fake_turn()).values())
    checks = turn_checks(fake_turn(status="limit", tool_errors=1))
    assert checks == {"replied": False, "no_tool_errors": False, "within_step_limit": False}


def test_case_checks_each_expectation():
    expect = {"tools": ["search_memory", "web_search", "remember"], "no_tools": ["forget"],
              "reply_contains": ["PUNE"], "reply_not_contains": ["Mumbai"], "no_tool_errors": True,
              "gate": {"retrieve": True}, "facts_contain": ["pune"], "max_cost_usd": 0.005}
    checks = case_checks(expect, fake_turn(), ["Priya lives in Pune"])
    failed = [name for name, ok in checks.items() if not ok]
    assert failed == ["called remember", "cost <= $0.005"]
    with pytest.raises(ValueError, match="unknown"):
        case_checks({"tool": ["x"]}, fake_turn(), [])


def test_judge_scales_grade():
    client = FakeClient(small={"score": 4, "reason": "good"})
    verdict = judge(client, "m", "q", "a", "be brief", ["remember"])
    assert verdict == {"score": 0.75, "grade": 4, "reason": "good"}
    prompt = client.small_requests[0]["messages"][0]["content"]
    assert "Criteria: be brief" in prompt and "Tools used: remember" in prompt


def test_load_cases_validates(tmp_path):
    good = tmp_path / "good.jsonl"
    good.write_text('// comment\n{"id": "a", "message": "hi"}\n\n{"messages": ["x", "y"]}\n')
    assert [c["id"] for c in load_cases(good)] == ["a", "case-4"]
    for content, error in [('{"id": "a"}', "needs"), ('{bad', "valid JSON"),
                           ('{"id":"a","message":"x"}\n{"id":"a","message":"y"}', "duplicate"),
                           ('{"message": "x", "expect": {"toolz": []}}', "unknown")]:
        bad = tmp_path / "bad.jsonl"
        bad.write_text(content)
        with pytest.raises(ValueError, match=error):
            load_cases(bad)


def test_bundled_dataset_is_valid():
    assert len(load_cases(ROOT / "evals" / "smoke.jsonl")) >= 5


class ScriptedAgent:
    """Stands in for the harness: replies from a dict, traces through the real Tracer."""

    def __init__(self, home, tracer, replies):
        self.store = MemoryStore(home / "memory.db")
        self.tracer, self.replies = tracer, replies

    def respond(self, message, session_id=None, source="chat", eval_run=None):
        if message == "crash":
            raise RuntimeError("agent exploded")
        session_id = self.store.ensure_session(session_id, message)
        turn = self.tracer.start(message, session_id, source=source, eval_run=eval_run)
        turn.notify("llm", name="model call 1", input_tokens=10, output_tokens=5, cost_usd=0.002)
        if "remember" in message:
            turn.notify("tool", name="remember", ok=True)
            self.store.add_fact("tea is masala chai")
        return turn.finish(self.replies.get(message, "ok"))


def test_run_eval_end_to_end(tmp_path):
    tracer = Tracer(tmp_path / "traces.db")
    replies = {"where?": "Pune", "remember tea": "Saved."}
    factory = lambda home, tr: ScriptedAgent(home, tr, replies)
    cases = [
        {"id": "pass", "setup_facts": ["x"], "message": "where?", "expect": {"reply_contains": ["pune"]}},
        {"id": "fail", "message": "where?", "expect": {"reply_contains": ["Mumbai"]}},
        {"id": "tools", "messages": ["hello", "remember tea"],
         "expect": {"tools": ["remember"], "facts_contain": ["masala"], "judge_min": 0.5}},
        {"id": "crash", "message": "crash"},
    ]
    client = FakeClient(small={"score": 3, "reason": "ok"})
    seen = []
    run = run_eval(cases, factory, tracer, "test.jsonl", "opus", client, "haiku", seen.append)
    assert [r["passed"] for r in run["results"]] == [True, False, True, False]
    assert run["passed"] == 2 and run["pass_rate"] == 0.5 and len(seen) == 4
    assert "agent exploded" in run["results"][3]["error"]
    assert run["cost_usd"] == pytest.approx(0.008)          # 4 turns ran (one case had 2)
    # traces are real, marked as eval, and carry the judge score
    traced = tracer.turns(eval_run=run["id"])
    assert len(traced) == 4 and {t["source"] for t in traced} == {"eval"}
    assert tracer.turn(run["results"][2]["turn_id"])["judge_score"] == 0.5
    assert tracer.eval_run(run["id"])["results"][1]["checks"] == {"reply contains 'Mumbai'": False}
    assert tracer.eval_runs()[0]["pass_rate"] == 0.5
    # each case got its own empty memory
    assert run["results"][0]["passed"]


def test_trace_to_case_round_trip(tmp_path):
    good = trace_to_case(fake_turn(feedback=1))
    assert good["expect"]["tools"] == ["search_memory"] and good["source_turn"] == "t1"
    bad = trace_to_case(fake_turn(feedback=-1, feedback_note="wrong city"))
    assert "tools" not in bad["expect"] and "wrong city" in bad["criteria"]
    path = tmp_path / "evals" / "mine.jsonl"
    append_case(path, good)
    with pytest.raises(ValueError, match="already"):
        append_case(path, good)
    assert load_cases(path)[0]["message"] == "hi"


def test_release_gate_exit_code(tmp_path, monkeypatch, capsys):
    tracer = Tracer(tmp_path / "traces.db")
    import harness.app
    monkeypatch.setattr(harness.app, "eval_setup",
                        lambda: (tracer, None, "opus", "haiku",
                                 lambda home, tr: ScriptedAgent(home, tr, {"q": "yes"})),
                        raising=False)
    data = tmp_path / "d.jsonl"
    data.write_text(json.dumps({"id": "a", "message": "q", "expect": {"reply_contains": ["yes"]}}) + "\n"
                    + json.dumps({"id": "b", "message": "q", "expect": {"reply_contains": ["no"]}}) + "\n")
    assert main([str(data), "--min-pass", "0.5"]) == 0
    assert main([str(data), "--min-pass", "0.8"]) == 1
    out = capsys.readouterr().out
    assert "PASS  a" in out and "FAIL  b  (reply contains 'no')" in out and "1/2 passed (50%)" in out
