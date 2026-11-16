"""Permission rules: matching, defaults by kind, failing closed, and the gate carrying them out."""

import json
from types import SimpleNamespace as NS

import pytest

from tools.approvals import Approvals
from tools.builtin import TurnContext
from tools.permissions import STARTER, Permissions, kind_from_annotations
from tools.registry import Tool, ToolRegistry


def rules(tmp_path, *items):
    path = tmp_path / "permissions.json"
    path.write_text(json.dumps({"rules": list(items)}))
    return Permissions(path, {"workspace__read_file": "read", "workspace__write_file": "local",
                              "email__send": "external", "mystery__tool": "unknown"})


def test_first_matching_rule_wins_with_globs_and_conditions(tmp_path):
    p = rules(tmp_path,
              {"tool": "workspace__write_file", "when": {"path": "finance/**"}, "action": "deny", "why": "money"},
              {"tool": "workspace__*", "action": "ask"},
              {"tool": "*", "action": "default"})
    assert p.decide("workspace__write_file", {"path": "finance/2026/tax.md"}).describe() == "deny (rule 1: money)"
    assert p.decide("workspace__write_file", {"path": "notes/a.md"}).action == "ask"           # rule 2
    assert p.decide("email__send", {}).describe() == "ask (default for its kind: external tool)"


@pytest.mark.parametrize("when, args, hit", [
    ({"attendees": "nonempty"}, {"attendees": ["a@x.com"]}, True),
    ({"attendees": "nonempty"}, {"attendees": []}, False),
    ({"attendees": "nonempty"}, {}, False),
    ({"url": "*bank*"}, {"url": "https://my.BANK.example"}, True),         # case-insensitive glob
    ({"to": "*@work.com"}, {"to": ["me@home.com", "boss@work.com"]}, True),   # any item of a list
    ({"amount": 0}, {"amount": 0}, True),
    ({"amount": 0}, {"amount": 5}, False),
])
def test_conditions(tmp_path, when, args, hit):
    p = rules(tmp_path, {"tool": "*", "when": when, "action": "deny"})
    assert (p.decide("workspace__write_file", args).action == "deny") is hit


def test_defaults_by_kind(tmp_path):
    p = rules(tmp_path, {"tool": "*", "action": "default"})
    assert [p.decide(t, {}).action for t in ("workspace__read_file", "workspace__write_file", "email__send",
                                             "mystery__tool", "never__seen")] == \
        ["allow", "allow", "ask", "ask", "ask"]
    assert p.decide("approve", {}).action == "allow"                 # the gate's own tools are never gated


@pytest.mark.parametrize("content, problem", [
    ("{not json", "isn't valid JSON"),
    (json.dumps({"rules": [{"tool": "*", "action": "alow"}]}), "rule 1: action must be one of"),
    (json.dumps({"rules": [{"action": "allow"}]}), "rule 1: needs a \"tool\" pattern"),
    (json.dumps({"rules": [{"tool": "*", "when": ["x"], "action": "allow"}]}), "\"when\" must be an object"),
])
def test_a_broken_file_fails_closed(tmp_path, content, problem):
    path = tmp_path / "permissions.json"
    path.write_text(content)
    p = Permissions(path, {"workspace__read_file": "read", "workspace__write_file": "local"})
    assert problem in p.problem
    assert p.decide("workspace__read_file", {}).action == "allow"
    assert p.decide("workspace__write_file", {}).action == "ask"      # even an undoable change asks now
    assert "broken" in p.decide("workspace__write_file", {}).why


def test_a_starter_file_is_written_on_first_run(tmp_path):
    p = Permissions(tmp_path / "permissions.json")
    assert json.loads((tmp_path / "permissions.json").read_text()) == STARTER and not p.problem
    assert p.decide("gcal__create_event", {"attendees": ["a@x.com"]}).rule == 1


def test_mcp_annotations_become_kinds():
    assert kind_from_annotations(None) == "unknown"
    assert kind_from_annotations(NS(read_only_hint=True)) == "read"
    assert kind_from_annotations(NS(read_only_hint=False, destructive_hint=True, open_world_hint=None)) == "external"
    assert kind_from_annotations(NS(read_only_hint=False, destructive_hint=None, open_world_hint=True)) == "external"
    assert kind_from_annotations(NS(read_only_hint=False, destructive_hint=False, open_world_hint=False)) == "local"
    assert kind_from_annotations(NS(read_only_hint=None, destructive_hint=None, open_world_hint=None)) == "unknown"


# ---- the gate carrying decisions out

def gate(tmp_path, *items):
    calls, seen = [], []
    registry = ToolRegistry()
    for name, kind in (("workspace__write_file", "local"), ("email__send", "external"), ("new__thing", "")):
        registry.register(Tool(name, "", {"type": "object", "properties": {}},
                               lambda _n=name, **a: calls.append((_n, a)) or "done", kind=kind))
    context = TurnContext("s1", "t1")
    path = tmp_path / "permissions.json"
    path.write_text(json.dumps({"rules": list(items) or [{"tool": "*", "action": "default"}]}))
    approvals = Approvals(context, Permissions(path))
    approvals.notify = lambda kind, **d: seen.append((kind, d))
    approvals.guard(registry)
    for tool in approvals.tools():
        registry.register(tool)
    return registry, approvals, context, calls, seen


def test_allow_ask_and_deny_through_the_gate(tmp_path):
    registry, approvals, context, calls, seen = gate(
        tmp_path, {"tool": "workspace__write_file", "when": {"path": "finance/*"}, "action": "deny", "why": "money"},
        {"tool": "*", "action": "default"})
    assert registry.execute("workspace__write_file", {"path": "a.md"}) == ("done", False)          # local: allow
    out, error = registry.execute("workspace__write_file", {"path": "finance/x.md"})
    assert error and "rule 1: money" in out and "Don't try another way" in out
    out, error = registry.execute("email__send", {"to": "boss@work.com"})                          # external: ask
    assert not error and out.startswith("NOT DONE YET") and "default for its kind" in out
    out, _ = registry.execute("new__thing", {})                                                   # no kind: ask
    assert out.startswith("NOT DONE YET")
    assert [c[0] for c in calls] == ["workspace__write_file"]
    assert [(d["action"], d["tool"]) for k, d in seen if k == "permission"] == \
        [("deny", "workspace__write_file"), ("ask", "email__send"), ("ask", "new__thing")]
    draft = next(d for d in approvals.drafts.values() if d.tool == "email__send")
    context.turn_id = "t2"                                                                         # the user replied
    assert registry.execute("approve", {"draft_id": draft.id}) == ("done", False)
    assert calls[-1] == ("email__send", {"to": "boss@work.com"})


def test_the_agent_gates_everything_and_reports_a_broken_file(tmp_path):
    from fake import FakeClient
    from harness.app import Settings, Sora
    (tmp_path / "permissions.json").write_text('{"rules": [{"tool": "*", "action": "sometimes"}]}')
    agent = Sora(Settings(home=tmp_path, memory_mirror=False), FakeClient([]), connect_mcp=False)
    assert "rule 1: action must be one of" in agent.status()["permissions"]["problem"]
    assert "remember" in agent.approvals.originals and "approve" not in agent.approvals.originals
    out, _ = agent.registry.execute("remember", {"fact": "x"})
    assert out.startswith("NOT DONE YET")                       # broken rules: even memory writes ask


def test_cli_lists_and_tests_calls(tmp_path, monkeypatch):
    import io
    from fake import FakeClient
    from harness import cli
    from harness.app import Settings, Sora
    monkeypatch.setattr(cli, "setup_logging", lambda *a: None)
    agent = Sora(Settings(home=tmp_path, memory_mirror=False), FakeClient([]), connect_mcp=False)
    out = io.StringIO()
    assert cli.main(["permissions"], create=lambda home: agent, out=out) == 0
    assert "remember" in out.getvalue() and "local" in out.getvalue()
    out = io.StringIO()
    cli.main(["permissions", "test", "gcal__create_event", '{"attendees": ["a@x.com"]}'], create=lambda h: agent, out=out)
    assert "ask (rule 1:" in out.getvalue()


def test_browser_calls_decide_their_kind_per_call(tmp_path):
    p = Permissions(tmp_path / "permissions.json")
    assert p.decide("browser__click", {"ref": "e3"}).action == "allow"                   # an ordinary click
    assert p.decide("browser__click", {"ref": "e9", "confirm": True}).action == "ask"    # a commit
    assert p.decide("browser__type", {"ref": "e2", "text": "hi"}).action == "allow"
    assert p.decide("browser__type", {"ref": "e2", "text": "hi", "submit": True}).action == "ask"
    assert p.decide("browser__press", {"key": "Enter", "confirm": True}).action == "ask"
