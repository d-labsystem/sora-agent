"""Does sora learn from its mistakes? A controlled learning-curve experiment.

sora's self-learning (memory/learning.py) reflects on turns that went wrong and puts
the lessons in later prompts. This measures whether that actually helps.

Quirky tools: small scripted tools, each with one hidden rule ("dates as DD/MM/YYYY").
Breaking the rule returns an error that explains it, so the rule can be learned; the
tool's description doesn't mention it, so it can't be known in advance. Nothing real
happens: the tools only record their calls.

A run is one home, with ~10 tasks per quirk in shuffled order, each a new conversation,
so the only thing that carries from task to task is learning. Two arms:

    on    sora as it is: reflection on tool errors, lessons in later prompts
    off   the same, with SORA_LEARNING=0

Fact memory, skills and the remember tool are off in both arms, so nothing else can
carry the rule. Every result is graded in code from the tool calls: did the first call
break the rule, how many calls broke it, did the task get done.

Quirks are split before any run: "dev" ones to build and debug with, "test" ones run
once at the end. The bar, set beforehand: on the test quirks, learning cuts repeat
errors (rule breaks after a quirk's first task) by at least half, every quirk the same way.
"""

from __future__ import annotations

import json
import random
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Callable

from tools.registry import Tool, ToolError

NOTHING_REAL = " (a test system: nothing real happens)"


def _schema(props: dict) -> dict:
    return {"type": "object", "properties": props, "required": list(props)}


@dataclass
class Quirk:
    name: str
    split: str                                     # dev | test
    tool: str                                      # the tool whose rule is measured
    tasks: list[str]
    make: Callable[[list], list[Tool]]             # (call log) -> the tools to register


def _log(log: list, tool: str, args: dict, ok: bool, broke: bool) -> None:
    log.append({"tool": tool, "args": args, "ok": ok, "broke": broke})


# ---- dev quirks: used to build and debug the experiment

def _book_room(log):
    def book_room(room: str, date: str) -> str:
        ok = bool(re.fullmatch(r"\d\d/\d\d/\d{4}", date.strip()))
        if ok:
            try:
                datetime.strptime(date.strip(), "%d/%m/%Y")
            except ValueError:
                ok = False
        _log(log, "book_room", {"room": room, "date": date}, ok, not ok)
        if not ok:
            raise ToolError(f"Booking failed: {date!r} is not a valid date. Dates must be written "
                            "DD/MM/YYYY, e.g. 18/11/2026.")
        return f"Booked {room} on {date.strip()}. Reference RB-{abs(hash((room, date))) % 9000 + 1000}."
    return [Tool("book_room", "Book a meeting room for a day" + NOTHING_REAL + ".",
                 _schema({"room": {"type": "string"}, "date": {"type": "string"}}), book_room, kind="read")]


def _send_invoice(log):
    def send_invoice(customer: str, amount: str) -> str:
        ok = bool(re.fullmatch(r"[A-Z]{3} \d+(\.\d\d)?", str(amount).strip()))
        _log(log, "send_invoice", {"customer": customer, "amount": amount}, ok, not ok)
        if not ok:
            raise ToolError(f"Invoice rejected: amount {amount!r} is invalid. Write the currency code, "
                            "a space, then the number, e.g. 'USD 45.50'.")
        return f"Invoice for {str(amount).strip()} sent to {customer}."
    return [Tool("send_invoice", "Send an invoice to a customer" + NOTHING_REAL + ".",
                 _schema({"customer": {"type": "string"}, "amount": {"type": "string"}}), send_invoice,
                 kind="read")]


# ---- test quirks: run once, at the end

CUSTOMERS = {"priya shah": "priya@shahdesign.in", "tom becker": "tom.becker@example.de",
             "ana lima": "ana@limaeventos.br", "kenji mori": "kenji@moricraft.jp",
             "sara okafor": "sara.okafor@example.ng", "leo martin": "leo@martinbikes.fr",
             "maya cohen": "maya@cohenlaw.il", "ravi nair": "ravi@nairfarms.in",
             "elena rossi": "elena@rossicafe.it", "omar haddad": "omar@haddadtrade.ae"}


def _find_customer(log):
    def find_customer(name: str) -> str:
        ok = name == name.lower()
        _log(log, "find_customer", {"name": name}, ok, not ok)
        if not ok:
            raise ToolError(f"No customer called {name!r}. Customer names are stored in lowercase.")
        email = CUSTOMERS.get(name.strip())
        return f"{name.strip()}: {email}" if email else f"No customer called {name!r}."
    return [Tool("find_customer", "Look up a customer's email by name" + NOTHING_REAL + ".",
                 _schema({"name": {"type": "string"}}), find_customer, kind="read")]


def _export_report(log):
    def export_report(month: str, format: str) -> str:
        ok = format.strip().lower() in ("csv", "tsv")
        _log(log, "export_report", {"month": month, "format": format}, ok, not ok)
        if not ok:
            raise ToolError(f"Export failed: format {format!r} isn't supported. Supported formats: csv, tsv.")
        return f"Exported the {month} report as report-{month.strip().replace(' ', '-').lower()}.{format.lower()}."
    return [Tool("export_report", "Export the monthly sales report as a file" + NOTHING_REAL + ".",
                 _schema({"month": {"type": "string"}, "format": {"type": "string"}}), export_report, kind="read")]


def _delete_record(log):
    issued: set[str] = set()

    def confirm_token(record_id: str) -> str:
        token = f"tok-{abs(hash(record_id)) % 90000 + 10000}"
        issued.add(f"{record_id.strip()}:{token}")
        return f"Token for record {record_id.strip()}: {token}"

    def delete_record(record_id: str, token: str = "") -> str:
        ok = f"{record_id.strip()}:{token.strip()}" in issued
        _log(log, "delete_record", {"record_id": record_id, "token": token}, ok, not ok)
        if not ok:
            raise ToolError("Delete refused: missing or wrong token. Call confirm_token(record_id) first "
                            "and pass the token it returns.")
        return f"Deleted record {record_id.strip()}."
    return [Tool("confirm_token", "Get a confirmation token for a record" + NOTHING_REAL + ".",
                 _schema({"record_id": {"type": "string"}}), confirm_token, kind="read"),
            Tool("delete_record", "Delete a record from the CRM" + NOTHING_REAL + ".",
                 {"type": "object", "properties": {"record_id": {"type": "string"}, "token": {"type": "string"}},
                  "required": ["record_id"]}, delete_record, kind="read")]


QUIRKS = [
    Quirk("book_room", "dev", "book_room", [
        "Book the Orion room for 3 December 2026.", "Please reserve room B on 2026-12-14.",
        "Can you book Atlas for Dec 9th, 2026?", "I need the Lotus room on the 21st of January 2027.",
        "Book room 4 for 2027-02-02, please.", "Reserve the Cedar room on February 17, 2027.",
        "Get me the Harbor room on 5 March 2027.", "Book Studio 2 for 2027/03/19.",
        "Reserve Orion again, this time for April 8th 2027.", "Book the boardroom on 30 April 2027."], _book_room),
    Quirk("send_invoice", "dev", "send_invoice", [
        "Invoice Priya Shah for $45.50.", "Send Tom Becker an invoice for 120 dollars.",
        "Bill Ana Lima $9.99 for the template.", "Invoice Kenji Mori for $1,250.00.",
        "Send an invoice of $310.75 to Sara Okafor.", "Bill Leo Martin 64 dollars and 20 cents.",
        "Invoice Maya Cohen $18.", "Send Ravi Nair an invoice for $2,040.40.",
        "Bill Elena Rossi for $75.25, please.", "Invoice Omar Haddad for $500."], _send_invoice),
    Quirk("find_customer", "test", "find_customer", [
        "What's Priya Shah's email?", "Find Tom Becker's email address.", "I need Ana Lima's email.",
        "Look up Kenji Mori's email for me.", "What email do we have for Sara Okafor?",
        "Get me Leo Martin's email.", "Can you find Maya Cohen's email?", "What's the email for Ravi Nair?",
        "Look up Elena Rossi.", "Find Omar Haddad's email, please."], _find_customer),
    Quirk("export_report", "test", "export_report", [
        "Export the March 2026 sales report as an Excel file.", "I need the April 2026 report as a PDF.",
        "Export May 2026's sales report so I can open it in Excel.", "Give me the June 2026 report as a spreadsheet.",
        "Export the July 2026 report in JSON.", "Can I get August 2026's report as a Word document?",
        "Export the September 2026 sales report.", "I want the October 2026 report as an xlsx.",
        "Export November 2026's report for Google Sheets.", "Send me the December 2026 report as a PDF file."], _export_report),
    Quirk("delete_record", "test", "delete_record", [
        "Delete CRM record 1042.", "Remove record 2210 from the CRM.", "Please delete record A-77.",
        "Get rid of CRM record 5531.", "Delete record 0099, it's a duplicate.", "Remove record 7300.",
        "Delete the CRM record with id 8812.", "Record 3003 is spam, delete it.", "Please remove record B-12.",
        "Delete record 6400 from the CRM."], _delete_record),
]


@dataclass
class RunResult:
    arm: str
    repeat: int
    tasks: list[dict] = field(default_factory=list)       # quirk, k, message, first_broke, broke, ok_calls, called
    lessons: list[dict] = field(default_factory=list)
    cost_usd: float = 0.0
    error: str = ""


def _agent(home: Path, base, learning: bool, quirks: list[Quirk], log: list, client=None):
    from harness.app import Sora
    settings = replace(base, home=home, learning=learning, consolidate_every=10 ** 6, web_tools=False,
                       code_execution=False, workspace=False, notes=False, browser=False, helpers=False,
                       sandbox="off", google_calendar="0", memory_mirror=False, auto_judge=False,
                       claim_check=False, phoenix_endpoint="", langfuse_public_key="")
    agent = Sora(settings, client, connect_mcp=False)
    for name in ("remember", "forget", "save_skill", "search_memory", "search_conversations", "load_skill"):
        agent.registry.unregister(name)         # nothing but lessons may carry a rule between tasks
    for q in quirks:
        for tool in q.make(log):
            agent.registry.register(tool)
    return agent


def run_one(base, quirks: list[Quirk], arm: str, repeat: int, n_tasks: int, seed: int = 0,
            client=None) -> RunResult:
    """One home, every quirk's tasks in a seeded shuffled order, a new conversation each."""
    order = [(q, i) for q in quirks for i in range(min(n_tasks, len(q.tasks)))]
    random.Random(seed * 1000 + repeat).shuffle(order)
    out = RunResult(arm, repeat)
    seen: dict[str, int] = {}
    with tempfile.TemporaryDirectory(prefix="sora-learning-") as home:
        log: list = []
        try:
            agent = _agent(Path(home), base, arm == "on", quirks, log, client)
            for q, i in order:
                seen[q.name] = seen.get(q.name, 0) + 1
                start = len(log)
                turn = agent.respond(q.tasks[i], source="eval")
                calls = [c for c in log[start:] if c["tool"] == q.tool]
                out.tasks.append({"quirk": q.name, "k": seen[q.name], "message": q.tasks[i],
                                  "called": bool(calls), "first_broke": calls[0]["broke"] if calls else None,
                                  "broke": sum(c["broke"] for c in calls), "done": any(c["ok"] for c in calls),
                                  "status": turn.get("status")})
                out.cost_usd += turn.get("cost_usd") or 0.0
            out.lessons = agent.lessons.list()
        except Exception as exc:                  # report it; don't lose the other runs
            out.error = f"{type(exc).__name__}: {exc}"
    return out


def run(base, split: str, repeats: int = 3, n_tasks: int = 10, seed: int = 0,
        arms: tuple[str, ...] = ("on", "off"), parallel: int = 6) -> list[RunResult]:
    quirks = [q for q in QUIRKS if q.split == split]
    if not quirks:
        raise ValueError(f"no quirks in split {split!r}")
    jobs = [(arm, r) for r in range(repeats) for arm in arms]
    with ThreadPoolExecutor(parallel) as ex:
        return list(ex.map(lambda j: run_one(base, quirks, j[0], j[1], n_tasks, seed), jobs))


def summarize(results: list[RunResult]) -> dict:
    """Per arm: the learning curve (share of tasks whose first call broke the rule, by
    encounter k), repeat errors (rule breaks at k >= 2), and the same per quirk."""
    out = {}
    for arm in sorted({r.arm for r in results}):
        tasks = [t for r in results if r.arm == arm for t in r.tasks]
        ks = sorted({t["k"] for t in tasks})
        called = lambda ts: [t for t in ts if t["called"]]
        curve = {k: (sum(t["first_broke"] for t in called([t for t in tasks if t["k"] == k])),
                     len(called([t for t in tasks if t["k"] == k]))) for k in ks}
        per_quirk = {}
        for q in sorted({t["quirk"] for t in tasks}):
            qt = [t for t in tasks if t["quirk"] == q]
            per_quirk[q] = {"repeat_errors": sum(t["broke"] for t in qt if t["k"] >= 2),
                            "first_errors": sum(t["broke"] for t in qt if t["k"] == 1),
                            "done": sum(t["done"] for t in qt), "tasks": len(qt)}
        out[arm] = {"curve": curve, "repeat_errors": sum(t["broke"] for t in tasks if t["k"] >= 2),
                    "done": sum(t["done"] for t in tasks), "tasks": len(tasks), "per_quirk": per_quirk,
                    "cost_usd": sum(r.cost_usd for r in results if r.arm == arm),
                    "errors": [r.error for r in results if r.arm == arm and r.error],
                    "lessons": sum(len(r.lessons) for r in results if r.arm == arm)}
    if "on" in out and "off" in out and out["off"]["repeat_errors"]:
        out["reduction"] = 1 - out["on"]["repeat_errors"] / out["off"]["repeat_errors"]
        out["quirk_reduction"] = {
            q: (1 - out["on"]["per_quirk"][q]["repeat_errors"] / v["repeat_errors"]) if v["repeat_errors"] else None
            for q, v in out["off"]["per_quirk"].items() if q in out["on"]["per_quirk"]}
    return out


def format_summary(s: dict) -> str:
    lines = []
    for arm in ("on", "off"):
        if arm not in s:
            continue
        a = s[arm]
        curve = "  ".join(f"{k}:{b}/{n}" for k, (b, n) in a["curve"].items())
        lines += [f"learning {arm}: {a['done']}/{a['tasks']} tasks done, {a['repeat_errors']} repeat errors, "
                  f"{a['lessons']} lessons saved, ${a['cost_usd']:.3f}",
                  f"  first call broke the rule, by encounter (k: broke/tasks): {curve}"]
        for q, v in a["per_quirk"].items():
            lines.append(f"  {q:15} first errors {v['first_errors']:2}  repeat errors {v['repeat_errors']:3}  "
                         f"done {v['done']}/{v['tasks']}")
        lines += [f"  RUN FAILED: {e}" for e in a["errors"]]
    if "reduction" in s:
        lines.append(f"\nrepeat errors, learning on vs off: {100 * s['reduction']:.0f}% fewer")
        lines += [f"  {q:15} " + ("no repeat errors without learning" if r is None else f"{100 * r:.0f}% fewer")
                  for q, r in s["quirk_reduction"].items()]
    return "\n".join(lines)


def main(split: str, repeats: int, n_tasks: int, seed: int, out_path: Path | None, out) -> int:
    from harness.app import load_settings
    results = run(load_settings(), split, repeats, n_tasks, seed)
    summary = summarize(results)
    out.write(format_summary(summary) + "\n")
    if out_path:
        out_path.write_text(json.dumps({"summary": summary, "runs": [r.__dict__ for r in results]},
                                       indent=1, default=str))
        out.write(f"details: {out_path}\n")
    return 0
