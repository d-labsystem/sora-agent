"""Permission rules: what sora may do on its own, what waits for you, what never runs.

The rules live in <home>/permissions.json, which you edit. The first rule
that matches a tool call decides:

    {"rules": [
      {"tool": "browser__*", "action": "ask", "why": "acts on real websites"},
      {"tool": "gcal__create_event", "when": {"attendees": "nonempty"}, "action": "ask"},
      {"tool": "workspace__write_file", "when": {"path": "finance/**"}, "action": "deny"},
      {"tool": "*", "action": "default"}
    ]}

    allow     runs now
    ask       held as a draft until you reply in a later message (tools/approvals.py)
    deny      doesn't run; the model is told which rule stopped it
    default   decided by what kind of tool it is (below)

`tool` is a glob over names (workspace__*). `when` tests arguments: "nonempty",
"empty", a glob for strings ("finance/**", "*bank*"), or an exact value.

Kinds, and their defaults when no rule says otherwise:
    read       only looks (read, search, list)                       allow
    local      changes your workspace or memory, and can be undone   allow
    external   acts outside your Mac, or can't be undone              ask
    unknown    a tool that says nothing about itself (a new MCP
               server without annotations): it must earn trust       ask
MCP tools say what they are with annotations (readOnlyHint,
destructiveHint, openWorldHint); sora's own tools are listed in KINDS.

A broken rules file fails closed: everything except reading is "ask", and
the status line says what's wrong. A typo never quietly allows everything.
"""

from __future__ import annotations

import fnmatch
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

ACTIONS = ("allow", "ask", "deny", "default")
DEFAULTS = {"read": "allow", "local": "allow", "external": "ask", "unknown": "ask"}
CONTROL = {"approve", "discard_draft"}         # the gate's own tools: never gated

# sora's built-in tools and what they do (MCP tools describe themselves).
KINDS = {
    "search_memory": "read", "load_skill": "read", "update_plan": "read", "current_time": "read",
    "show_chart": "read", "show_table": "read", "show_diagram": "read",
    "delegate": "read",
    "remember": "local", "forget": "local", "save_skill": "local", "run_code": "local",
}

# Tools whose kind depends on the call. The browser refuses commits (a "Place order"
# click, Enter in a form) unless confirm=true, and confirm=true is external: it asks.
CALL_KINDS = {
    "browser__click": lambda a: "external" if a.get("confirm") else "read",
    "browser__press": lambda a: "external" if a.get("confirm") else "read",
    "browser__type": lambda a: "external" if a.get("submit") or a.get("confirm") else "local",
}

STARTER = {"rules": [
    {"tool": "gcal__create_event", "when": {"attendees": "nonempty"}, "action": "ask",
     "why": "Google emails an invitation to every attendee, in your name"},
    {"tool": "*", "action": "default", "why": "by kind: reads and undoable changes run, the rest asks"},
]}


def kind_from_annotations(annotations) -> str:
    """MCP's hints as one of our kinds."""
    if annotations is None:
        return "unknown"
    if getattr(annotations, "read_only_hint", None):
        return "read"
    if getattr(annotations, "destructive_hint", None) or getattr(annotations, "open_world_hint", None):
        return "external"
    if getattr(annotations, "destructive_hint", None) is False and getattr(annotations, "open_world_hint", None) is False:
        return "local"
    return "unknown"


@dataclass
class Rule:
    tool: str
    action: str
    when: dict = field(default_factory=dict)
    why: str = ""


@dataclass
class Decision:
    action: str            # allow | ask | deny
    rule: int | None       # 1-based rule number, None when decided by kind
    why: str

    def describe(self) -> str:
        source = f"rule {self.rule}" if self.rule else "default for its kind"
        return f"{self.action} ({source}{': ' + self.why if self.why else ''})"


def matches_value(want, got) -> bool:
    if want == "nonempty":
        return bool(got)
    if want == "empty":
        return not got
    if isinstance(want, str) and isinstance(got, str):
        return fnmatch.fnmatch(got, want) or fnmatch.fnmatch(got.lower(), want.lower())
    if isinstance(want, str) and isinstance(got, list):
        return any(isinstance(g, str) and fnmatch.fnmatch(g, want) for g in got)
    return want == got


class Permissions:
    def __init__(self, path: Path | None, kinds: dict[str, str] | None = None):
        self.path = path
        self.kinds = dict(KINDS, **(kinds or {}))
        self.rules: list[Rule] = []
        self.problem = ""              # set when the file is broken: then we fail closed
        self.load()

    def load(self) -> None:
        self.rules, self.problem = [], ""
        if self.path is None:
            raw = STARTER
        elif not self.path.is_file():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(STARTER, indent=2) + "\n", encoding="utf-8")
            raw = STARTER
        else:
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                self.problem = f"{self.path.name} isn't valid JSON ({exc})"
                log.error("permissions: %s; failing closed", self.problem)
                return
        try:
            for n, r in enumerate(raw.get("rules", []), 1):
                if not isinstance(r, dict) or not isinstance(r.get("tool"), str):
                    raise ValueError(f"rule {n}: needs a \"tool\" pattern")
                if r.get("action") not in ACTIONS:
                    raise ValueError(f"rule {n}: action must be one of {', '.join(ACTIONS)}, not {r.get('action')!r}")
                if not isinstance(r.get("when", {}), dict):
                    raise ValueError(f"rule {n}: \"when\" must be an object of argument conditions")
                self.rules.append(Rule(r["tool"], r["action"], r.get("when", {}), r.get("why", "")))
        except (ValueError, AttributeError) as exc:
            self.rules, self.problem = [], f"{self.path.name if self.path else 'rules'}: {exc}"
            log.error("permissions: %s; failing closed", self.problem)

    def kind(self, tool: str, args: dict | None = None) -> str:
        if tool in CALL_KINDS:
            return CALL_KINDS[tool](args or {})
        return self.kinds.get(tool, "unknown")

    def decide(self, tool: str, args: dict) -> Decision:
        if tool in CONTROL:
            return Decision("allow", None, "the approval tools themselves")
        kind = self.kind(tool, args)
        if self.problem:               # fail closed
            return Decision("allow" if kind == "read" else "ask", None,
                            f"the rules file is broken ({self.problem}), so only reading runs without asking")
        for n, rule in enumerate(self.rules, 1):
            if not fnmatch.fnmatch(tool, rule.tool):
                continue
            if any(not matches_value(want, args.get(key)) for key, want in rule.when.items()):
                continue
            if rule.action == "default":
                return Decision(DEFAULTS[kind], None, f"{kind} tool")
            return Decision(rule.action, n, rule.why)
        return Decision(DEFAULTS[kind], None, f"{kind} tool")

    def table(self, tools: list[str]) -> list[tuple[str, str, str]]:
        """(tool, kind, what happens with no special arguments) for the CLI."""
        return [(t, self.kind(t), self.decide(t, {}).describe()) for t in sorted(tools)]
