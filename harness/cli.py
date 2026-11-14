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
            query = (data.get("input") or {}).get("query") or (data.get("input") or {}).get("url", "")
            self.line(self.c(DIM, f"  ↳ {data.get('name')}: {query}"))
        elif kind == "gate" and data.get("facts"):
            self.line(self.c(DIM, f"  ↳ recalled {len(data['facts'])} fact(s)"))
        elif kind == "learning" and data.get("lessons"):
            for lesson in data["lessons"]:
                self.line(self.c(DIM, f"  ↳ learned: {lesson['lesson']}"))
        elif kind == "memory" and data.get("facts"):
            self.line(self.c(DIM, f"  ↳ saved {len(data['facts'])} fact(s) to long-term memory"))

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

    def send(self, message: str) -> dict:
        turn = self.agent.respond(message, self.session_id, source="cli", listener=self.printer)
        self.printer.finish()
        if turn["status"] == "error":
            self.say(self.printer.c(RED, f"Error: {turn['error']}"))
        elif turn["status"] == "limit":
            self.say(self.printer.c(DIM, "(stopped at the step limit)"))
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
    rows = [f"model {status['model']} (small: {status['small_model']}, effort {status['effort']})",
            f"home  {status['home']}", "tools:"]
    rows += [f"  {t['name']:<28} {t['source']}" for t in status["tools"]]
    if status["mcp"]:
        rows.append("MCP servers:")
        rows += [f"  {m['name']:<12} {m['status']:<10} {len(m['tools'])} tools"
                 + (f"  {m['error']}" if m["error"] else "") for m in status["mcp"]]
    if status["skill_errors"]:
        rows.append("skill errors:")
        rows += [f"  {e}" for e in status["skill_errors"]]
    return "\n".join(rows)


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
    sub.add_parser("serve", help="run the web dashboard")
    ev = sub.add_parser("eval", help="run an eval dataset")
    ev.add_argument("dataset")
    ev.add_argument("--min-pass", default="0")
    ev.add_argument("--label", default="")
    ev.add_argument("--repeat", default="1")
    ev.add_argument("--rejudge", metavar="RUN_ID", help="grade an old run again; the agent doesn't run")
    ex = sub.add_parser("export", help="export traces as JSONL")
    ex.add_argument("-o", "--output", type=Path)
    ex.add_argument("--source", choices=["chat", "cli", "eval"])
    sub.add_parser("status", help="model, tools, MCP servers and skills")
    svc = sub.add_parser("service", help="run the dashboard in the background (macOS)")
    svc.add_argument("action", choices=["plist", "install", "status", "restart", "uninstall"])
    args = parser.parse_args(argv)
    command = args.command or "chat"

    if command == "eval":       # builds its own agents, one per case
        from ops import evals
        return evals.main([args.dataset, "--min-pass", args.min_pass, "--label", args.label,
                           "--repeat", args.repeat]
                          + (["--rejudge", args.rejudge] if args.rejudge else []))

    if command == "export":     # needs only the traces, not the agent
        from harness.app import load_settings
        from ops.tracing import Tracer
        tracer = Tracer(load_settings(args.home).home / "traces.db")
        target = args.output.open("w", encoding="utf-8") if args.output else out
        try:
            count = tracer.export_jsonl(target, source=args.source)
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

    if command == "serve":
        from harness import web
        return web.main(home=args.home)

    if create is None:
        from harness.app import create, load_settings
        setup_logging(load_settings(args.home).home / "logs")
    agent = create(args.home)
    try:
        if command == "status":
            out.write(format_status(agent.status()) + "\n")
            return 0
        chat = Chat(agent, out, color=out.isatty())
        if command == "ask":
            turn = chat.send(" ".join(args.message))
            return 0 if turn["status"] in ("ok", "limit") else 1
        chat.run(read)
        return 0
    finally:
        agent.close()


if __name__ == "__main__":
    sys.exit(main())
