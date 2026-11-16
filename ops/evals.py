"""Evals: is the agent actually good, and did my change make it better?

Three layers, from cheapest to most expensive:

    turn checks   code rules run on every real turn, for free
                  (did it reply, any tool errors, did it hit the step limit)
    case checks   expectations written for a test case: which tools to use,
                  what the reply must contain, what memory must end up holding
    LLM judge     a small model grades the reply against criteria, 1-5,
                  for qualities code can't check ("is this a useful plan?")

Datasets are JSONL files in evals/, one case per line:

    {"id": "remembers-name",
     "setup_facts": ["The user's name is Aryan"],
     "message": "What's my name?",
     "expect": {"reply_contains": ["Aryan"], "no_tool_errors": true,
                "gate": {"retrieve": true}},
     "criteria": "Answers with the name, briefly."}

A case may use "messages": [...] for a multi-turn conversation; checks apply
to the last turn. "setup_files": {"name.md": "text"} puts files in the case's
workspace before it runs. Expectation keys are listed in EXPECT_KEYS below.

The runner gives every case a fresh, empty home folder, so cases can't leak
memory into each other or into your real memory. Their traces go into the
real traces.db marked source="eval", so you can open any failed case in the
dashboard and see exactly what happened.

The release gate: `python -m ops.evals evals/smoke.jsonl --min-pass 0.8`
exits with an error if the pass rate is below 0.8. Run it before you change
a prompt or a model, and compare.

The flywheel: any real turn can become a case (trace_to_case). A thumbs-down
today becomes a regression test tomorrow.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from loop.llm import ask_json, schema
from ops.tracing import Tracer, now_iso
from tools.visuals import VISUAL_TOOLS

EXPECT_KEYS = {
    "files":              "object: {workspace file: [texts it must contain]} after the case",
    "plan_closed":        "bool: the last plan had no open steps (all done or blocked)",
    "tools":              "list: each of these tools was called",
    "no_tools":           "list: none of these tools was called",
    "reply_contains":     "list: each appears in the reply (case-insensitive)",
    "reply_not_contains": "list: none appears in the reply",
    "no_tool_errors":     "bool: no tool call failed",
    "gate":               'object: e.g. {"retrieve": true} or {"correction": true}',
    "recalls":            "list: each appears in a fact the gate put in the prompt (retrieval, before the model runs)",
    "facts_contain":      "list: each appears in some saved fact after the turn",
    "facts_not_contain":  "list: none appears in any saved fact (forgotten or replaced)",
    "max_facts":          "number: at most this many facts saved in total (catches duplicates)",
    "max_cost_usd":       "number: the turn cost at most this",
    "judge_min":          "number 0-1: the LLM judge's score is at least this",
}


class Agent(Protocol):
    """What the eval runner needs from the harness (harness/app.py builds it)."""
    store: Any                         # MemoryStore

    def respond(self, message: str, session_id: str | None = None, source: str = "chat",
                eval_run: str | None = None) -> dict: ...  # the finished turn, with spans


AgentFactory = Callable[[Path, Tracer], Agent]     # (fresh home folder, shared tracer)


# ---- checks

def turn_checks(turn: dict) -> dict[str, bool]:
    """Free checks for every real turn. Stored with the trace."""
    return {
        "replied": turn.get("status") == "ok" and bool((turn.get("reply") or "").strip()),
        "no_tool_errors": not turn.get("tool_errors"),
        "within_step_limit": turn.get("status") != "limit",
        "within_budget": turn.get("status") != "budget",
    }


def _called(turn: dict) -> set[str]:
    return {s["name"] for s in turn.get("spans", []) if s["kind"] in ("tool", "server_tool")}


def _gate(turn: dict) -> dict:
    return next((s["data"] for s in turn.get("spans", []) if s["kind"] == "gate"), {})


def case_checks(expect: dict, turn: dict, facts: list[str], workspace: Path | None = None) -> dict[str, bool]:
    """Apply a case's expectations to its last turn. Returns {check: passed}."""
    unknown = set(expect) - set(EXPECT_KEYS)
    if unknown:
        raise ValueError(f"unknown expectations: {', '.join(sorted(unknown))}")
    reply = (turn.get("reply") or "").lower()
    called = _called(turn)
    out: dict[str, bool] = {}
    for tool in expect.get("tools", []):
        out[f"called {tool}"] = tool in called
    for tool in expect.get("no_tools", []):
        out[f"did not call {tool}"] = tool not in called
    for text in expect.get("reply_contains", []):
        out[f"reply contains {text!r}"] = text.lower() in reply
    for text in expect.get("reply_not_contains", []):
        out[f"reply lacks {text!r}"] = text.lower() not in reply
    if expect.get("no_tool_errors"):
        out["no tool errors"] = not turn.get("tool_errors")
    for key, want in (expect.get("gate") or {}).items():
        out[f"gate {key}={want}"] = _gate(turn).get(key) == want
    recalled = [f.lower() for f in _gate(turn).get("facts") or []]
    for text in expect.get("recalls", []):
        out[f"recalled {text!r}"] = any(text.lower() in f for f in recalled)
    lowered = [f.lower() for f in facts]
    for text in expect.get("facts_contain", []):
        out[f"remembered {text!r}"] = any(text.lower() in f for f in lowered)
    for text in expect.get("facts_not_contain", []):
        out[f"no fact has {text!r}"] = not any(text.lower() in f for f in lowered)
    if "max_facts" in expect:
        out[f"at most {expect['max_facts']} facts"] = len(facts) <= expect["max_facts"]
    if "max_cost_usd" in expect:
        out[f"cost <= ${expect['max_cost_usd']}"] = (turn.get("cost_usd") or 0) <= expect["max_cost_usd"]
    for name, texts in (expect.get("files") or {}).items():
        path = (workspace / name) if workspace else None
        content = path.read_text(encoding="utf-8").lower() if path and path.is_file() else None
        if content is not None:     # 1,550 and 2,88,000 count as 1550 and 288000
            content = re.sub(r"(?<=\d)[,\u00a0 ](?=\d)", "", content)
        out[f"wrote {name}"] = content is not None
        for text in texts:
            out[f"{name} has {text!r}"] = content is not None and text.lower() in content
    if "plan_closed" in expect:
        plans = [s["data"] for s in turn.get("spans", []) if s["kind"] == "plan" and s.get("name") == "plan"]
        closed = bool(plans) and not any(st["status"] in ("pending", "in_progress") for st in plans[-1]["steps"])
        out["plan closed"] = closed == bool(expect["plan_closed"])
    return out


# ---- LLM judge

JUDGE_FORMAT = schema({
    "score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
    "reason": {"type": "string"},
})

JUDGE_PROMPT = """Grade a personal assistant's reply. Be strict and specific.

5 = fully meets the criteria, correct, nothing important missing
4 = meets them with minor issues
3 = partly meets them, a noticeable gap or error
2 = mostly misses them
1 = wrong, harmful, or ignores the request

Criteria: {criteria}

User message:
{message}

Tools used: {tools}

What those tools returned (the assistant read this; a detail in the reply that
appears here is supported by evidence, not invented):
{evidence}

Visuals shown to the user next to the reply (charts, tables and diagrams
render beside the text, so a reply may point to them instead of repeating them):
{visuals}

Reply:
{reply}

Give the score and one or two sentences of reason, naming the main issue if any."""

DEFAULT_CRITERIA = "Correct, directly answers the request, appropriately brief, no invented facts."


def visual_summary(turn: dict, max_rows: int = 12) -> str:
    """What the user saw beside the reply, as text the judge can read: each
    chart's points, each table's rows, each diagram's labels. Built from the
    visual tools' arguments in the turn's spans."""
    lines = []
    for span in turn.get("spans", []):
        if span["kind"] != "tool" or not span.get("ok", True):
            continue
        args = (span.get("data") or {}).get("args") or {}
        title = args.get("title", "untitled")
        if span["name"] == "show_chart":
            lines.append(f"- {args.get('type', '')} chart '{title}'"
                         + (f" (y: {args['y_label']})" if args.get("y_label") else ""))
            labels = args.get("labels") or []
            for series in args.get("series") or []:
                points = ", ".join(f"{l}={v}" for l, v in zip(labels, series.get("values") or []))
                lines.append(f"  {series.get('name') or 'values'}: {points}")
        elif span["name"] == "show_table":
            rows = args.get("rows") or []
            lines.append(f"- table '{title}': " + " | ".join(map(str, args.get("columns") or [])))
            lines += ["  " + " | ".join(map(str, row)) for row in rows[:max_rows]]
            if len(rows) > max_rows:
                lines.append(f"  ... {len(rows) - max_rows} more rows")
        elif span["name"] == "show_diagram":
            labels = [t.strip() for t in re.findall(r"<text[^>]*>([^<]+)</text>", args.get("svg", ""))]
            lines.append(f"- diagram '{title}'" + (f", labels: {', '.join(labels[:30])}" if labels else ""))
    return "\n".join(lines) or "none"


def tool_evidence(turn: dict, per_call: int = 1500, total: int = 12000) -> str:
    """What each tool returned in this turn, clipped, so the judge can check
    the reply against the same data the assistant saw. Without it, facts read
    from a file look "unverified" and get marked down (complex-tasks eval,
    2026-11-16: three correct replies failed that way)."""
    lines = []
    for span in turn.get("spans", []):
        data = span.get("data") or {}
        if span["kind"] == "tool" and span["name"] not in VISUAL_TOOLS:
            args = json.dumps(data.get("args", {}), ensure_ascii=False)[:300]
            output = str(data.get("output", ""))
            clipped = output[:per_call] + (f" …[{len(output) - per_call} more chars]" if len(output) > per_call else "")
            lines.append(f"- {span['name']}({args}) ->\n{clipped}")
        elif span["kind"] == "server_tool":
            what = (data.get("input") or {}).get("query") or (data.get("input") or {}).get("url") or ""
            lines.append(f"- {span['name']}: {what} (results not recorded)")
    text = "\n".join(lines)
    return (text[:total] + "\n…[more tool output not shown]") if len(text) > total else (text or "none")


def judge(client, model: str, message: str, reply: str, criteria: str | None = None,
          tools: list[str] | None = None, notify=None, visuals: str = "none", evidence: str = "none") -> dict:
    """Score 0-1 (from a 1-5 grade) with a reason. Raises on failure."""
    answer = ask_json(client, model, JUDGE_PROMPT.format(
        criteria=criteria or DEFAULT_CRITERIA, message=message[:4000],
        tools=", ".join(tools or []) or "none", visuals=visuals[:4000], evidence=evidence,
        reply=(reply or "(no reply)")[:8000]),
        JUDGE_FORMAT, notify, name="judge", max_tokens=4096)   # long replies get long reasons
    return {"score": (answer["score"] - 1) / 4, "grade": answer["score"], "reason": answer["reason"]}


# ---- datasets

def load_cases(path: Path) -> list[dict]:
    cases, seen = [], set()
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("//"):
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{n}: not valid JSON ({exc.msg})") from None
        case.setdefault("id", f"case-{n}")
        if case["id"] in seen:
            raise ValueError(f"{path}:{n}: duplicate id {case['id']!r}")
        seen.add(case["id"])
        if not case.get("message") and not case.get("messages"):
            raise ValueError(f"{path}:{n}: needs 'message' or 'messages'")
        unknown = set(case.get("expect", {})) - set(EXPECT_KEYS)
        if unknown:
            raise ValueError(f"{path}:{n}: unknown expectations {sorted(unknown)}")
        cases.append(case)
    return cases


def repeat(cases: list[dict], times: int) -> list[dict]:
    """Each case `times` times, ids suffixed #1, #2...; one run then shows the spread."""
    if times <= 1:
        return cases
    return [{**c, "id": f"{c['id']}#{i}"} for c in cases for i in range(1, times + 1)]


def trace_to_case(turn: dict) -> dict:
    """Draft a regression case from a real turn. Edit it before relying on it:
    for a thumbs-down turn, the draft asks for better than what happened."""
    good = (turn.get("feedback") or 0) >= 0 and not turn.get("tool_errors")
    expect: dict[str, Any] = {"no_tool_errors": True}
    if good:
        tools = sorted({s["name"] for s in turn.get("spans", []) if s["kind"] == "tool"})
        if tools:
            expect["tools"] = tools
    expect["judge_min"] = 0.75
    criteria = DEFAULT_CRITERIA
    if turn.get("feedback_note"):
        criteria += f" The user said about an earlier answer: {turn['feedback_note']}"
    return {"id": f"from-{turn['id']}", "message": turn["message"], "expect": expect,
            "criteria": criteria, "source_turn": turn["id"]}


def append_case(path: Path, case: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = {c["id"] for c in load_cases(path)} if path.exists() else set()
    if case["id"] in existing:
        raise ValueError(f"{path.name} already has a case {case['id']!r}")
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(case, ensure_ascii=False) + "\n")


# ---- runner

def run_case(case: dict, factory: AgentFactory, tracer: Tracer, run_id: str,
             client=None, judge_model: str | None = None, grade=None) -> dict:
    """Run one case in a fresh home. `grade(case, turn)` (benchmarks) replaces
    the case's checks with its own: {"checks": {...}, "judge": {...} | None}."""
    with tempfile.TemporaryDirectory(prefix="sora-eval-") as home:
        agent = None
        spent = 0.0          # every turn of the case; kept even if a later step fails
        try:
            agent = factory(Path(home), tracer)
            for fact in case.get("setup_facts", []):
                agent.store.add_fact(fact, source="eval-setup")
            for name, content in (case.get("setup_files") or {}).items():
                workspace = (Path(home) / "workspace").resolve()
                file = (workspace / name).resolve()
                if workspace not in file.parents:
                    raise ValueError(f"setup file {name!r} is outside the workspace")
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_text(content, encoding="utf-8")
            session = None
            turn: dict = {}
            for message in case.get("messages") or [case["message"]]:
                extra = {"long": True} if case.get("long") else {}   # long task: more steps, a cap
                turn = agent.respond(message, session_id=session, source="eval", eval_run=run_id, **extra)
                session = turn.get("session_id")
                spent += turn.get("cost_usd") or 0
            facts = [f["content"] for f in agent.store.list_facts()]
            checks = case_checks(case.get("expect", {}), turn, facts, Path(home) / "workspace")
            verdict = None
            if "judge_min" in case.get("expect", {}):
                verdict = judge(client, judge_model, turn["message"], turn.get("reply") or "",
                                case.get("criteria"), sorted(_called(turn)),
                                visuals=visual_summary(turn), evidence=tool_evidence(turn))
                tracer.set_judge(turn["id"], verdict["score"], verdict["reason"])
                checks[f"judge >= {case['expect']['judge_min']}"] = \
                    verdict["score"] >= case["expect"]["judge_min"]
            if grade is not None:
                # A benchmark grades the answer by its own rules. A turn that stopped at
                # its step or dollar limit still answered, so only an error fails it outright.
                graded = grade(case, turn)
                checks, verdict = graded["checks"], graded.get("judge")
                if verdict:
                    tracer.set_judge(turn["id"], verdict["score"], verdict["reason"])
                if turn.get("status") == "error":
                    checks["turn finished"] = False
            elif turn.get("status") != "ok":
                checks["turn finished"] = False
            return {"id": case["id"], "turn_id": turn.get("id"), "passed": all(checks.values()),
                    "checks": checks, "judge": verdict, "reply": (turn.get("reply") or "")[:500],
                    "cost_usd": spent}
        except Exception as exc:
            return {"id": case["id"], "turn_id": None, "passed": False, "checks": {},
                    "judge": None, "error": f"{type(exc).__name__}: {exc}",
                    "cost_usd": spent}   # the turns ran and were billed; budgets must see it
        finally:
            semantic = getattr(getattr(agent, "store", None), "semantic", None)
            if semantic is not None and semantic.name != "sqlite":
                semantic.clear()        # this case's scope in Postgres; a SQLite home is just deleted
            if agent is not None and hasattr(agent, "close"):
                agent.close()           # e.g. stop the case's MCP servers


def run_eval(cases: list[dict], factory: AgentFactory, tracer: Tracer, dataset: str,
             model: str, client=None, judge_model: str | None = None,
             progress: Callable[[dict], None] | None = None) -> dict:
    run_id = "eval-" + uuid.uuid4().hex[:8]
    started = now_iso()
    results = []
    for case in cases:
        result = run_case(case, factory, tracer, run_id, client, judge_model)
        results.append(result)
        if progress:
            progress(result)
    run = {"id": run_id, "dataset": dataset, "model": model, "cases": len(results),
           "passed": sum(r["passed"] for r in results),
           "cost_usd": round(sum(r["cost_usd"] for r in results), 6),
           "results": results, "started_at": started, "ended_at": now_iso()}
    tracer.save_eval_run(run)
    run["pass_rate"] = run["passed"] / run["cases"] if run["cases"] else 0.0
    return run


def rejudge(run_id: str, cases: list[dict], tracer: Tracer, client, judge_model: str) -> list[dict]:
    """Grade an old run's replies again with the current judge, without rerunning
    the agent: the replies and visuals are already in the traces. Read-only, so
    the stored scores stay as they were and the two can be compared."""
    run = tracer.eval_run(run_id)
    if run is None:
        raise ValueError(f"no eval run {run_id!r}")
    by_id = {c["id"]: c for c in cases}
    out = []
    for result in run["results"]:
        case = by_id.get(result["id"].split("#")[0])
        turn = tracer.turn(result["turn_id"]) if result.get("turn_id") else None
        if case is None or turn is None or "judge_min" not in case.get("expect", {}):
            continue
        old = (result.get("judge") or {}).get("grade")
        try:
            verdict = judge(client, judge_model, turn["message"], turn.get("reply") or "",
                            case.get("criteria"), sorted(_called(turn)), visuals=visual_summary(turn),
                            evidence=tool_evidence(turn))
        except Exception as exc:          # one failed grade shouldn't lose the others
            out.append({"id": result["id"], "old": old, "new": None, "reason": f"judge failed: {exc}"})
            continue
        out.append({"id": result["id"], "old": old, "new": verdict["grade"], "reason": verdict["reason"]})
    return out


def main(argv: list[str] | None = None, factory: AgentFactory | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ops.evals", description="Run an eval dataset.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--min-pass", type=float, default=0.0,
                        help="exit with an error if the pass rate is below this (0-1)")
    parser.add_argument("--repeat", type=int, default=1,
                        help="run every case N times: the agent isn't deterministic, so one run can mislead")
    parser.add_argument("--label", default="",
                        help="shown next to the dataset name, e.g. 'embeddings off', to tell runs apart")
    parser.add_argument("--rejudge", metavar="RUN_ID",
                        help="grade that run's replies again with the current judge; the agent doesn't run")
    args = parser.parse_args(argv)

    from harness.app import eval_setup          # imported late: the harness imports this module
    tracer, client, model, judge_model, default_factory = eval_setup()
    if args.rejudge:
        for r in rejudge(args.rejudge, load_cases(args.dataset), tracer, client, judge_model):
            print(f"{r['id']}: {r['old']} -> {r['new']}  {r['reason']}")
        return 0
    cases = repeat(load_cases(args.dataset), args.repeat)

    def show(r: dict) -> None:
        failed = [name for name, ok in r["checks"].items() if not ok]
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"{mark}  {r['id']}" + (f"  ({r.get('error') or ', '.join(failed)})" if not r["passed"] else ""))

    name = f"{args.dataset.name} ({args.label})" if args.label else args.dataset.name
    run = run_eval(cases, factory or default_factory, tracer, name, model,
                   client, judge_model, show)
    print(f"\n{run['passed']}/{run['cases']} passed ({run['pass_rate']:.0%}), "
          f"cost ${run['cost_usd']:.4f}, run {run['id']}")
    if run["pass_rate"] < args.min_pass:
        print(f"Below the --min-pass of {args.min_pass:.0%}.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
