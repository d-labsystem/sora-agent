"""The terminal gateway: chat with sora, and run the ops commands.

    python main.py                       chat (the default)
    python main.py ask "what's on my list?"
    python main.py serve                 the web dashboard
    python main.py eval evals/smoke.jsonl --min-pass 0.8
    python main.py export -o traces.jsonl [--source chat]
    python main.py status                model, tools, MCP servers, skills
    python main.py service install       run the dashboard in the background (macOS)

Like every gateway, it only calls Sora.respond() and friends: the CLI owns
input and display, nothing else. Streaming works through the same listener
the dashboard uses, so text appears as it's generated and tool calls show up
as they happen.

Inside chat, slash commands reach the parts behind the agent:
/new /sessions /facts /lessons /skills /tools /good /bad /trace /help /quit
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import IO

DIM, BOLD, RED, GREEN, RESET = "\033[2m", "\033[1m", "\033[31m", "\033[32m", "\033[0m"

HELP = """Commands:
  /new             start a new conversation
  /sessions        recent conversations (/open <id> to continue one)
  /facts [query]   what sora remembers (all, or matching a search)
  /lessons         lessons learned from past mistakes
  /skills          available skills
  /tools           tools and MCP servers
  /good  /bad [note]   rate the last reply (👎 makes sora reflect on it)
  /trace           steps, time and cost of the last turn
  /quit            exit"""


class Printer:
    """Turns trace events into terminal output, live."""

    def __init__(self, out: IO[str], color: bool):
        self.out, self.color = out, color
        self.mid_line = False

    def c(self, code: str, text: str) -> str:
        return f"{code}{text}{RESET}" if self.color else text

    def line(self, text: str) -> None:
        if self.mid_line:
            self.out.write("\n")
            self.mid_line = False
        self.out.write(text + "\n")
        self.out.flush()

    def __call__(self, kind: str, data: dict) -> None:
        if kind == "text":
            self.out.write(data.get("delta", ""))
            self.out.flush()
            self.mid_line = True
        elif kind == "tool":
            mark = self.c(GREEN, "✓") if data.get("ok", True) else self.c(RED, "✗")
            self.line(self.c(DIM, f"  ↳ {data.get('name')} ") + mark + self.c(DIM, f" {data.get('ms', 0)}ms"))
        elif kind == "server_tool":
            args = data.get("input") or {}
            # search -> query, fetch -> url, code -> its first line (Anthropic's code
            # tools send `code` or `command`); nothing reported -> just the name
            what = args.get("query") or args.get("url") or \
                (args.get("code") or args.get("command") or "").strip().split("\n")[0][:80]
            self.line(self.c(DIM, f"  ↳ {data.get('name')}" + (f": {what}" if what else "")))
        elif kind == "gate" and data.get("facts"):
            self.line(self.c(DIM, f"  ↳ recalled {len(data['facts'])} fact(s)"))
        elif kind == "learning" and data.get("lessons"):
            for lesson in data["lessons"]:
                self.line(self.c(DIM, f"  ↳ learned: {lesson['lesson']}"))
        elif kind == "memory" and data.get("facts"):
            self.line(self.c(DIM, f"  ↳ saved {len(data['facts'])} fact(s) to long-term memory"))
        elif kind == "plan" and data.get("name") == "plan":
            marks = {"pending": "○", "in_progress": "◐", "done": "●", "blocked": "✗"}
            self.line(self.c(DIM, "  ↳ plan: " + "  ".join(f"{marks[s['status']]} {s['title']}"
                                                         for s in data.get("steps", []))))
        elif kind == "plan":
            self.line(self.c(DIM, f"  ↳ {data.get('open')} plan step(s) still open: sent back to finish them"))

    def finish(self) -> None:
        if self.mid_line:
            self.out.write("\n")
            self.mid_line = False


def describe_turn(turn: dict) -> str:
    rows = [f"turn {turn['id']}  {turn['status']}  {turn.get('latency_ms') or 0}ms  "
            f"${turn.get('cost_usd') or 0:.4f}  {turn.get('input_tokens')} in + "
            f"{turn.get('cache_read_tokens') or 0} cached / "
            f"{turn.get('output_tokens')} out"]
    for s in turn.get("spans", []):
        mark = "✓" if s["ok"] else "✗"
        cost = s["data"].get("cost_usd")
        extra = f"  ${cost:.4f}" if cost else ""
        rows.append(f"  {s['start_ms']:>6}ms  {mark} {s['kind']:<12} {s['name']}"
                    f"{f'  ({s['duration_ms']}ms)' if s['duration_ms'] else ''}{extra}")
    return "\n".join(rows)


class Chat:
    """The interactive session. Separated from I/O so tests can drive it."""

    def __init__(self, agent, out: IO[str], color: bool = False):
        self.agent, self.out = agent, out
        self.printer = Printer(out, color)
        self.session_id: str | None = None
        self.last_turn: dict | None = None

    def say(self, text: str) -> None:
        self.printer.line(text)

    def send(self, message: str, long: bool = False) -> dict:
        turn = self.agent.respond(message, self.session_id, source="cli", listener=self.printer,
                                  long=long)
        self.printer.finish()
        if turn["status"] == "error":
            self.say(self.printer.c(RED, f"Error: {turn['error']}"))
        elif turn["status"] == "limit":
            self.say(self.printer.c(DIM, "(stopped at the step limit)"))
        elif turn["status"] == "budget":
            self.say(self.printer.c(DIM, "(stopped at the long-task spending cap)"))
        self.session_id = turn.get("session_id") or self.session_id
        self.last_turn = turn
        return turn

    def command(self, line: str) -> bool:
        """Run a slash command. Returns False when the chat should end."""
        name, _, arg = line[1:].partition(" ")
        arg = arg.strip()
        a = self.agent
        if name in ("quit", "exit", "q"):
            return False
        if name == "help":
            self.say(HELP)
        elif name == "new":
            self.session_id, self.last_turn = None, None
            self.say("Started a new conversation.")
        elif name == "sessions":
            for s in a.store.list_sessions(15):
                self.say(f"  {s['id']}  {s['updated_at'][:16]}  {s['title']}")
        elif name == "open":
            ids = {s["id"] for s in a.store.list_sessions(500)}
            if arg in ids:
                self.session_id = arg
                for m in a.store.history(arg, 3):
                    self.say(f"  {m['role']}: {m['content'][:200]}")
            else:
                self.say(f"No conversation {arg!r}. Try /sessions.")
        elif name == "facts":
            facts = a.store.search_facts(arg, 20) if arg else [f["content"] for f in a.store.list_facts()]
            self.say("\n".join(f"  - {f}" for f in facts) or "  (nothing yet)")
        elif name == "lessons":
            rows = a.lessons.list()
            self.say("\n".join(f"  - [{r['tool'] or 'general'}] {r['lesson']}  (used {r['uses']}x)"
                               for r in rows) or "  (no lessons yet)")
        elif name == "skills":
            for k in a.skills.all().values():
                self.say(f"  {k.name}{' (bundled)' if k.bundled else ''}: {k.description}")
        elif name == "tools":
            self.say(format_status(a.status()))
        elif name in ("good", "bad"):
            if not self.last_turn:
                self.say("Nothing to rate yet.")
            else:
                result = a.feedback(self.last_turn["id"], 1 if name == "good" else -1, arg)
                self.say("Thanks." + "".join(f"\n  learned: {l['lesson']}" for l in result["lessons"]))
        elif name == "trace":
            if not self.last_turn:
                self.say("No turn yet.")
            else:
                self.say(describe_turn(a.tracer.turn(self.last_turn["id"])))
        else:
            self.say(f"Unknown command /{name}. Type /help.")
        return True

    def run(self, read: Callable[[str], str] = input) -> None:
        self.say(f"sora · {self.agent.settings.model} · /help for commands, /quit to leave")
        while True:
            try:
                line = read(self.printer.c(BOLD, "you › ")).strip()
            except (EOFError, KeyboardInterrupt):
                self.say("")
                return
            if not line:
                continue
            if line.startswith("/"):
                if not self.command(line):
                    return
                continue
            self.out.write(self.printer.c(BOLD, "sora › "))
            self.send(line)


def format_status(status: dict) -> str:
    prov = status.get("providers", {})
    rows = [f"model {status['model']} (small: {status['small_model']}, effort {status['effort']})",
            f"via   {prov.get('main', '?')} / {prov.get('small', '?')}, web tools "
            + ("on" if prov.get("web_tools") else "off") + ", server tools: "
            + (", ".join(t["name"] for t in status["tools"] if t["source"] == prov.get("main")) or "none"),
            *[f"limit {line}" for line in prov.get("limits", [])],
            f"home  {status['home']}"]
    if status.get("sandbox", {}).get("offered"):
        problem = status["sandbox"].get("problem")
        rows.append("code  run_code in Docker" + (f" (not usable: {problem})" if problem else ""))
    if status.get("otel"):
        rows.append("otel  " + ", ".join(f"{t['name']} {t['endpoint']}"
                                         + ("" if t["content"] else " (no text)") for t in status["otel"]))
    if status.get("memory"):
        m = status["memory"]
        rows.append(f"memory {m['backend']}, {status.get('memory_search', '')}, consolidated by "
                    f"{m['consolidator']}" + (f", mirrored to {m['mirror']}" if m.get("mirror") else ""))
    rows.append("tools:")
    rows += [f"  {t['name']:<28} {t['source']}" for t in status["tools"]]
    if status["mcp"]:
        rows.append("MCP servers:")
        rows += [f"  {m['name']:<12} {m['status']:<10} {len(m['tools'])} tools"
                 + (f"  {m['error']}" if m["error"] else "") for m in status["mcp"]]
    if status["skill_errors"]:
        rows.append("skill errors:")
        rows += [f"  {e}" for e in status["skill_errors"]]
    return "\n".join(rows)


def browser_command(args, out: IO[str]) -> int:
    import asyncio
    from harness.app import load_settings
    from tools import browser_server
    settings = load_settings(args.home)
    if args.action == "login":
        if not args.url:
            out.write("usage: browser login https://site.example\n")
            return 1
        out.write("A browser window opens on sora's own profile (not your Chrome profile). Sign in there, "
                  "then close the window; sora will reuse that sign-in.\n")
        out.flush()
        asyncio.run(browser_server.login(settings.home, args.url, settings.browser_engine))
        out.write("Saved. Close any other windows of that browser before sora uses it.\n")
        return 0

    async def check():
        b = browser_server.Browser(settings.home, settings.browser_engine)
        try:
            await b.start()
            return f"browser OK: {b.context.browser.version if b.context.browser else 'Chromium'} " \
                   f"(engine {settings.browser_engine}), profile {settings.home / 'browser' / 'profile'}"
        finally:
            await b.stop()
    try:
        out.write(asyncio.run(check()) + "\n")
        return 0
    except Exception as exc:
        out.write(f"browser can't start: {exc}\n")
        return 1


def permissions_command(agent, args, out: IO[str]) -> int:
    """Rules apply to the live tool list, MCP tools included, so this builds the agent."""
    import json
    perms = agent.permissions
    out.write(f"rules: {agent.settings.home / 'permissions.json'} ({len(perms.rules)} rules)\n")
    if perms.problem:
        out.write(f"BROKEN: {perms.problem}. Failing closed: only reading runs without asking.\n")
    tools = [t["name"] for t in agent.registry.describe() if t["source"] != agent.registry.server_source]
    if args.action == "test":
        if not args.tool:
            out.write("usage: permissions test TOOL '{\"arg\": \"value\"}'\n")
            return 1
        try:
            call = json.loads(args.args)
        except ValueError as exc:
            out.write(f"arguments must be JSON: {exc}\n")
            return 1
        out.write(f"{args.tool} ({perms.kind(args.tool)}): {perms.decide(args.tool, call).describe()}\n")
        return 0
    for tool, kind, what in perms.table(tools):
        out.write(f"  {tool:34} {kind:9} {what}\n")
    return 0


def google_command(args, out: IO[str]) -> int:
    from harness.app import load_settings
    from tools import google_auth
    home = load_settings(args.home).home
    try:
        if args.action == "login":
            scopes = google_auth.login(home)
            out.write(f"Signed in to Google. Granted: {scopes}\nToken saved to "
                      f"{home / google_auth.TOKEN_FILE} (only you can read it). "
                      f"Calendar tools turn on the next time sora starts.\n")
        elif args.action == "logout":
            done = google_auth.logout(home)
            out.write("Signed out: access revoked at Google and the local token deleted.\n" if done
                      else "Not signed in.\n")
        else:
            token = google_auth.load_token(home)
            out.write(("Signed in to Google" if token else "Not signed in") + "\n")
            if token:
                google_auth.access_token(home)            # proves the refresh token still works
                out.write("Token works.\n")
        return 0
    except google_auth.AuthError as exc:
        out.write(f"{exc}\n")
        return 1


def whatsapp_command(args, out: IO[str]) -> int:
    """The webhook alone, in the foreground (the dashboard isn't started)."""
    import threading
    from harness import whatsapp
    from harness.app import create, load_settings
    settings = load_settings(args.home)
    setup_logging(settings.home / "logs")
    provider, why = whatsapp.provider_from(settings)
    if provider is None or not settings.whatsapp_allowed:
        out.write(f"WhatsApp isn't set up: {why or 'set SORA_WHATSAPP_ALLOWED to your number'}\n")
        return 1
    agent = create(args.home)
    server, gateway = whatsapp.start(agent, settings)
    out.write(f"sora WhatsApp webhook: http://127.0.0.1:{settings.whatsapp_port}{whatsapp.PATH} via "
              f"{provider.name}. Expose it with: cloudflared tunnel --url http://localhost:{settings.whatsapp_port}"
              f"  (Ctrl+C to stop)\n")
    out.flush()
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        gateway.stop()
        agent.close()
    return 0


def otel_backfill(settings, tracer, source, since, out: IO[str]) -> int:
    """Send finished turns from traces.db to the OTel destinations. Each turn
    keeps its trace id, so sending one twice updates it rather than copying."""
    from ops import otel
    targets = otel.destinations(settings)
    if not targets:
        out.write("No OTel destination set: add SORA_PHOENIX_ENDPOINT, or LANGFUSE_PUBLIC_KEY "
                  "and LANGFUSE_SECRET_KEY, to .env\n")
        return 1
    try:
        exporter = otel.OTelExporter(targets)
    except ImportError as exc:
        out.write(f"OTel packages missing: uv sync --extra otel ({exc})\n")
        return 1
    sent = 0
    for turn_id in tracer.finished_ids(source=source, since=since):
        exporter.export(tracer.turn(turn_id))
        sent += 1
    exporter.close()                         # waits for the queue to drain
    delivery = exporter.delivery()
    out.write(f"exported {sent} turns\n")
    failed = False
    for t in targets:
        d = delivery[t.name]
        failed |= d["rejected"] > 0
        out.write(f"  {t.name} ({'with' if t.content else 'without'} text): {d['accepted']} spans accepted"
                  + (f", {d['rejected']} REJECTED (see logs/sora.log; HTTP 401 means it needs a key)"
                     if d["rejected"] else "") + "\n")
    return 1 if failed else 0


def memory_command(args, out: IO[str]) -> int:
    from harness.app import load_settings
    from memory import mirror
    from memory.backends import open_memory
    from memory.embeddings import Embedder
    settings = load_settings(args.home)
    embedder = Embedder(settings.models_dir, settings.embedding_model) if settings.embeddings else None
    if args.action == "mirror":
        store = open_memory(settings)
        try:
            mirror.write(settings.home / "MEMORY.md", store.semantic, store.semantic.name)
        finally:
            store.close()
        out.write(f"wrote {settings.home / 'MEMORY.md'}\n")
        return 0
    if args.source == args.to:
        out.write("--from and --to are the same backend; nothing to copy\n")
        return 1
    try:
        source = open_memory(settings, kind=args.source)
        target = open_memory(settings, embedder, kind=args.to)
    except ValueError as exc:
        out.write(f"{exc}\n")
        return 1
    try:
        copied = target.semantic.import_from(source.semantic)
    finally:
        source.close()
        target.close()
    out.write(f"copied {copied['facts']} facts and {copied['episodes']} episodes "
              f"from {args.source} to {args.to}. Set SORA_MEMORY={args.to} to use it.\n")
    return 0


def keep_awake() -> None:
    """Keep the Mac from idle-sleeping while this process runs (a benchmark can
    take an hour): `caffeinate -i -s -w <pid>` ends by itself when we exit.
    A LongMemEval run stalled overnight when the Mac slept (2026-11-16). It
    can't stop sleep from closing the lid on battery."""
    import os
    import shutil
    import subprocess
    if sys.platform == "darwin" and shutil.which("caffeinate"):
        subprocess.Popen(["caffeinate", "-i", "-s", "-w", str(os.getpid())],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def judge_command(args, out: IO[str]) -> int:
    from harness.app import load_settings
    from ops.judge_check import JudgeCheck, benchmark_references, format_report
    from ops.tracing import Tracer
    settings = load_settings(args.home)
    check = JudgeCheck(Tracer(settings.home / "traces.db"))
    if args.action == "sample":
        runs = [r.strip() for r in args.runs.split(",") if r.strip()]
        if not runs:
            out.write("give the runs to sample from: --runs bench-…,bench-… (see the Evals tab)\n")
            return 1
        try:
            added = check.sample(runs, args.n, benchmark_references(settings.home), args.seed)
        except (KeyError, ValueError) as exc:
            out.write(f"{exc.args[0]}\n")
            return 1
        p = check.progress()
        out.write(f"added {added} verdicts ({p['total']} in the set, {p['done']} labelled). "
                  "Label them in the dashboard: Evals → Judge check.\n")
        return 0
    if args.action == "status":
        p = check.progress()
        out.write(f"{p['done']}/{p['total']} labelled, {p['unsure']} unsure\n")
        return 0
    out.write(format_report(check.report()) + "\n")
    return 0


def setup_logging(folder: Path) -> None:
    """Details (tracebacks, MCP failures) go to a log file, not over the chat."""
    folder.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=folder / "sora.log", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main(argv: list[str] | None = None, create=None, out: IO[str] = sys.stdout,
         read: Callable[[str], str] = input) -> int:
    parser = argparse.ArgumentParser(prog="sora", description="A local personal AI assistant.")
    parser.add_argument("--home", type=Path, help="data folder (default ~/.sora or SORA_HOME)")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("chat", help="chat in the terminal (default)")
    ask = sub.add_parser("ask", help="ask one question and print the reply")
    ask.add_argument("message", nargs="+")
    ask.add_argument("--long", action="store_true",
                     help="run as a long task: more steps, a dollar cap, compaction")
    sub.add_parser("serve", help="run the web dashboard (and WhatsApp, when configured)")
    sub.add_parser("whatsapp", help="run only the WhatsApp webhook")
    ev = sub.add_parser("eval", help="run an eval dataset")
    ev.add_argument("dataset")
    ev.add_argument("--min-pass", default="0")
    ev.add_argument("--label", default="")
    ev.add_argument("--repeat", default="1")
    ev.add_argument("--rejudge", metavar="RUN_ID", help="grade an old run again; the agent doesn't run")
    bn = sub.add_parser("bench", help="run a public benchmark (frames) on a sample, with a spending cap")
    bn.add_argument("name", choices=["frames", "gaia", "browsecomp", "longmemeval"])
    bn.add_argument("--bare", action="store_true",
                    help="the same model and tools without sora's harness, to measure what the harness adds")
    bn.add_argument("--full-context", action="store_true",
                    help="longmemeval: the whole chat history in the prompt and no memory system (the paper's baseline)")
    bn.add_argument("--closed-book", action="store_true",
                    help="the model alone, no tools: a high score means the answers were memorised")
    bn.add_argument("--n", type=int, default=50, help="questions to sample (0 = all)")
    bn.add_argument("--budget", type=float, default=12.0, help="stop once this many dollars are spent")
    bn.add_argument("--seed", type=int, default=0)
    bn.add_argument("--label", default="")
    bn.add_argument("--ids", default="", help="comma-separated question ids to run instead of a sample")
    jc = sub.add_parser("judge", help="check the benchmark judge: sample its verdicts, you label them blind, see agreement")
    jc.add_argument("action", choices=["sample", "status", "report"])
    jc.add_argument("--runs", default="", help="sample: comma-separated eval run ids to draw verdicts from")
    jc.add_argument("--n", type=int, default=60, help="sample: how many verdicts to add")
    jc.add_argument("--seed", type=int, default=0)
    ex = sub.add_parser("export", help="export traces as JSONL")
    ex.add_argument("-o", "--output", type=Path)
    ex.add_argument("--source", choices=["chat", "cli", "eval"])
    ex.add_argument("--otel", action="store_true",
                    help="send turns to the configured Phoenix/Langfuse instead of writing JSONL")
    ex.add_argument("--since", help="only turns started on or after this date (YYYY-MM-DD)")
    sub.add_parser("status", help="model, tools, MCP servers and skills")
    prm = sub.add_parser("permissions", help="what each tool may do on its own; test a call against the rules")
    prm.add_argument("action", nargs="?", default="list", choices=["list", "test"])
    prm.add_argument("tool", nargs="?")
    prm.add_argument("args", nargs="?", default="{}", help="the call's arguments as JSON")
    brw = sub.add_parser("browser", help="sign in to a site in sora's browser, or check the browser starts")
    brw.add_argument("action", choices=["login", "check"])
    brw.add_argument("url", nargs="?")
    sbx = sub.add_parser("sandbox", help="build the code sandbox image, or check it works")
    sbx.add_argument("action", choices=["build", "check"])
    cal = sub.add_parser("calendar", help="allow sora to use the Mac's Calendar, and list what it sees")
    cal.add_argument("action", choices=["check"])
    goo = sub.add_parser("google", help="sign in to Google Calendar, or out")
    goo.add_argument("action", choices=["login", "logout", "status"])
    mem = sub.add_parser("memory", help="move facts between backends, or rewrite MEMORY.md")
    mem_sub = mem.add_subparsers(dest="action", required=True)
    mig = mem_sub.add_parser("migrate", help="copy facts and episodes into another backend")
    mig.add_argument("--to", required=True, choices=["sqlite", "pgvector", "langgraph"])
    mig.add_argument("--from", dest="source", default="sqlite", choices=["sqlite", "pgvector", "langgraph"])
    mem_sub.add_parser("mirror", help="rewrite MEMORY.md from the configured backend now")
    svc = sub.add_parser("service", help="run the dashboard in the background (macOS)")
    svc.add_argument("action", choices=["plist", "install", "status", "restart", "uninstall"])
    args = parser.parse_args(argv)
    command = args.command or "chat"

    if command == "eval":       # builds its own agents, one per case
        from ops import evals
        return evals.main([args.dataset, "--min-pass", args.min_pass, "--label", args.label,
                           "--repeat", args.repeat]
                          + (["--rejudge", args.rejudge] if args.rejudge else []))

    if command == "bench":      # builds its own agents, one per question
        from ops import bench
        keep_awake()
        return bench.main(args.name, args.n, args.budget, args.seed, args.label, out, bare=args.bare,
                          closed_book=args.closed_book, ids=args.ids, full_context=args.full_context)

    if command == "judge":      # reads runs and labels in traces.db; no model calls
        return judge_command(args, out)

    if command == "export":     # needs only the traces, not the agent
        from harness.app import load_settings
        from ops.tracing import Tracer
        settings = load_settings(args.home)
        tracer = Tracer(settings.home / "traces.db")
        if args.otel:
            return otel_backfill(settings, tracer, args.source, args.since, out)
        target = args.output.open("w", encoding="utf-8") if args.output else out
        try:
            count = tracer.export_jsonl(target, source=args.source, since=args.since)
        finally:
            if args.output:
                target.close()
        print(f"Exported {count} turns" + (f" to {args.output}" if args.output else ""), file=sys.stderr)
        return 0

    if command == "service":    # manages a separate process; doesn't start an agent here
        from harness.app import load_settings
        from ops import service
        settings = load_settings(args.home)
        return service.main(args.action, settings.home, settings.port,
                            out=lambda line: out.write(line + "\n"))

    if command == "memory":     # works on the stores directly; no agent, no model calls
        return memory_command(args, out)

    if command == "browser":     # a visible window on sora's own profile; the agent isn't needed
        return browser_command(args, out)

    if command == "sandbox":
        from tools import sandbox
        if args.action == "build":
            return sandbox.build_image(lambda line: out.write(line + "\n"))
        from harness.app import load_settings
        box = sandbox.Sandbox(load_settings(args.home).home / "workspace")
        problem = box.available()
        out.write((f"sandbox: {problem}" if problem else box.run_code("import sys; print('ok', sys.version.split()[0])")) + "\n")
        return 1 if problem else 0

    if command == "calendar":   # run it in Terminal: macOS asks the app that runs it
        from tools import calendar_server
        return calendar_server.check(lambda line: out.write(line + "\n"))

    if command == "google":     # the OAuth sign-in runs here, in your browser; no agent needed
        return google_command(args, out)

    if command == "whatsapp":
        return whatsapp_command(args, out)

    if command == "serve":
        from harness import web
        return web.main(home=args.home)

    if create is None:
        from harness.app import create, load_settings
        setup_logging(load_settings(args.home).home / "logs")
    agent = create(args.home)
    try:
        if command == "permissions":
            return permissions_command(agent, args, out)
        if command == "status":
            out.write(format_status(agent.status()) + "\n")
            return 0
        chat = Chat(agent, out, color=out.isatty())
        if command == "ask":
            turn = chat.send(" ".join(args.message), long=args.long)
            return 0 if turn["status"] in ("ok", "limit", "budget") else 1
        chat.run(read)
        return 0
    finally:
        agent.close()


if __name__ == "__main__":
    sys.exit(main())
