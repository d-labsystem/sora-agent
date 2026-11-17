"""The web gateway: a local dashboard to chat with sora and watch it work.

Standard-library HTTP server, no framework: the whole API fits on one screen,
and there's nothing to install. The page is static HTML + JS (harness/static).

    GET  /                          the dashboard
    POST /api/chat                  {message, session_id, long?} → server-sent events
    GET  /api/sessions[/<id>]       conversations, and one conversation's messages
    POST /api/feedback              {turn_id, value: 1|-1|0, note}
    POST /api/judge | /api/to-eval  {turn_id}
    GET  /api/turns[/<id>]          traces (filters: source, status, limit, offset)
    GET  /api/stats                 headline numbers (days, source)
    GET  /api/export                all turns as JSONL
    GET  /api/memory                facts, episodes, lessons
    DELETE /api/memory/<kind>/<id>  kind: fact | episode | lesson
    DELETE /api/skills/<name>       a user skill
    GET  /api/evals[/<run_id>]      datasets and runs, or one run's results
    POST /api/evals/run             {dataset} → runs in the background
    GET  /api/status                model, tools, MCP servers, skills
    GET  /api/events                server-sent events: {"topic": "sessions" | "turns" |
                                    "memory" | "evals"} whenever that data changes

Security. The server listens on 127.0.0.1 only, so other machines can't reach
it. Two more checks stop *websites* you visit from using it through your own
browser: the Host header must be localhost (blocks DNS rebinding), and any
Origin must be localhost (blocks cross-site requests). POSTs must be JSON,
which a plain HTML form on another site can't send without an Origin.

Streaming. /api/chat answers with server-sent events: one `data: {json}` line
per trace event, so the page shows text and tool calls as they happen. If the
tab closes, the turn still finishes and is saved; only the display is lost.
"""

from __future__ import annotations

import json
import mimetypes
import queue
import re
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from harness.app import PROJECT, Sora, create, load_settings
from harness.events import KEEPALIVE_SECONDS, Hub, Watcher, sse
from memory.learning import LessonStore
from memory import mirror
from memory.backends import open_memory
from ops.evals import load_cases, run_eval
from ops.judge_check import JudgeCheck, format_report

STATIC = Path(__file__).parent / "static"
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
MAX_BODY = 1_000_000


def _hostname(value: str) -> str:
    try:
        return urlsplit(f"//{value}").hostname or ""
    except ValueError:
        return ""


def _summary(turn: dict) -> dict:
    return {k: v for k, v in turn.items() if k != "spans"}


class Dashboard:
    """Everything the handlers need. Reads use their own database connections,
    so the dashboard stays responsive while a turn holds the agent's lock."""

    def __init__(self, agent: Sora, watch_interval: float = 1.0):
        self.agent = agent
        path = agent.settings.home / "memory.db"
        # Its own connection (and, for Postgres, its own backend connection);
        # deletes made here rewrite MEMORY.md too.
        self.memory = open_memory(agent.settings)
        if agent.settings.memory_mirror:
            mirror.attach(self.memory, agent.settings.home / "MEMORY.md")
        self.lessons = LessonStore(path)
        self.judge = JudgeCheck(agent.tracer)
        self.read_lock = threading.Lock()
        self.eval_lock = threading.Lock()
        self.eval_running: str | None = None
        self.hub = Hub()
        # eval_running lives in memory, not a table, so the watcher is told about it directly
        self.watcher = Watcher(self.hub, path, agent.tracer.path, watch_interval,
                               extra=lambda: {"evals": self.eval_running}).start()

    def close(self) -> None:
        self.watcher.stop()
        self.hub.close()
        self.memory.close()

    # ---- evals

    def datasets(self) -> list[dict]:
        found = []
        for folder, kind in ((PROJECT / "evals", "bundled"), (self.agent.settings.home / "evals", "yours")):
            for path in sorted(folder.glob("*.jsonl")) if folder.is_dir() else []:
                try:
                    count = len(load_cases(path))
                    error = ""
                except (ValueError, OSError) as exc:
                    count, error = 0, str(exc)
                found.append({"name": path.name, "path": str(path), "kind": kind,
                              "cases": count, "error": error})
        return found

    def start_eval(self, name: str) -> None:
        match = next((d for d in self.datasets() if d["name"] == name and not d["error"]), None)
        if match is None:
            raise KeyError(f"no valid dataset {name!r}")
        with self.eval_lock:
            if self.eval_running:
                raise RuntimeError(f"an eval is already running ({self.eval_running})")
            self.eval_running = name
        agent = self.agent
        settings = agent.settings

        def factory(home: Path, tracer):
            return Sora(replace(settings, home=home, auto_judge=False), agent.providers.anthropic_client,
                        tracer=tracer,
                        embedder=agent.embedder)

        def work():
            try:
                run_eval(load_cases(Path(match["path"])), factory, agent.tracer, name,
                         settings.model, agent.small, agent.small_id)
            finally:
                with self.eval_lock:
                    self.eval_running = None

        threading.Thread(target=work, name="eval", daemon=True).start()


def make_handler(dash: Dashboard):
    agent = dash.agent

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args) -> None:
            pass

        # ---- plumbing

        def _workspace_image(self, relative: str) -> None:
            """Charts that run_code saved. Images only, and only from the workspace:
            the path is resolved and must stay inside it."""
            root = (agent.settings.home / "workspace").resolve()
            target = (root / relative).resolve()
            kinds = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif",
                     ".webp": "image/webp", ".svg": "image/svg+xml"}
            if root not in target.parents or target.suffix.lower() not in kinds or not target.is_file():
                return self._error(404, "not found")
            headers = {"Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'"}   # an SVG can't run script
            return self._send(200, target.read_bytes(), kinds[target.suffix.lower()], headers)

        def _send(self, status: int, body: bytes, content_type: str, extra: dict | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data, status: int = 200) -> None:
            self._send(status, json.dumps(data, default=str).encode(), "application/json")

        def _error(self, status: int, message: str) -> None:
            self._json({"error": message}, status)

        def _allowed(self) -> bool:
            host_ok = _hostname(self.headers.get("Host", "")) in LOCAL_HOSTS
            origin = self.headers.get("Origin")
            origin_ok = origin is None or _hostname(urlsplit(origin).netloc) in LOCAL_HOSTS
            if host_ok and origin_ok:
                return True
            self._send(403, b"forbidden", "text/plain")
            return False

        def _body(self) -> dict | None:
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                self._error(415, "send JSON")
                return None
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                self._error(413, "request too large")
                return None
            try:
                data = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._error(400, "invalid JSON")
                return None
            if not isinstance(data, dict):
                self._error(400, "send a JSON object")
                return None
            return data

        def _route(self):
            url = urlsplit(self.path)
            parts = [p for p in url.path.split("/") if p]
            query = {k: v[-1] for k, v in parse_qs(url.query).items()}
            return parts, query

        # ---- GET

        def do_GET(self) -> None:
            if not self._allowed():
                return
            parts, q = self._route()
            try:
                self._get(parts, q)
            except Exception as exc:
                self._error(500, f"{type(exc).__name__}: {exc}")

        def _get(self, parts: list[str], q: dict) -> None:
            if parts[:2] == ["api", "workspace"] and len(parts) > 2:
                return self._workspace_image("/".join(parts[2:]))
            if not parts:
                return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            if parts[0] == "static" and len(parts) == 2:
                file = (STATIC / parts[1]).resolve()
                if file.parent != STATIC.resolve() or not file.is_file():
                    return self._error(404, "not found")
                kind = mimetypes.guess_type(file.name)[0] or "application/octet-stream"
                return self._send(200, file.read_bytes(), kind)
            if parts[0] != "api" or len(parts) < 2:
                return self._error(404, "not found")
            name, rest = parts[1], parts[2:]
            tracer = agent.tracer
            if name == "status":
                return self._json(agent.status())
            if name == "events":
                return self._events()
            if name == "sessions":
                with dash.read_lock:
                    if rest:
                        return self._json(dash.memory.messages(rest[0]))
                    return self._json(dash.memory.list_sessions(int(q.get("limit", 50))))
            if name == "turns":
                if rest:
                    turn = tracer.turn(rest[0])
                    return self._json(turn) if turn else self._error(404, "no such turn")
                return self._json(tracer.turns(int(q.get("limit", 50)), int(q.get("offset", 0)),
                                               q.get("source"), q.get("session_id"),
                                               q.get("status"), q.get("eval_run")))
            if name == "stats":
                return self._json(tracer.stats(int(q.get("days", 7)), q.get("source", "chat")))
            if name == "export":
                import io
                buffer = io.StringIO()
                tracer.export_jsonl(buffer, source=q.get("source"))
                return self._send(200, buffer.getvalue().encode(), "application/x-ndjson",
                                  {"Content-Disposition": 'attachment; filename="sora-traces.jsonl"'})
            if name == "memory":
                with dash.read_lock:
                    return self._json({"facts": dash.memory.list_facts(),
                                       "episodes": dash.memory.list_episodes(),
                                       "lessons": dash.lessons.list()})
            if name == "judge-check":
                if rest == ["report"]:
                    report = dash.judge.report()
                    return self._json({**report, "text": format_report(report)})
                label_id = int(q["id"]) if q.get("id", "").isdigit() else None
                return self._json({"progress": dash.judge.progress(), "item": dash.judge.item(label_id)})
            if name == "evals":
                if rest:
                    run = tracer.eval_run(rest[0])
                    return self._json(run) if run else self._error(404, "no such run")
                return self._json({"datasets": dash.datasets(), "runs": tracer.eval_runs(),
                                   "running": dash.eval_running})
            return self._error(404, "not found")

        def _events(self) -> None:
            """Hold the connection open and write each notice as it arrives.
            Runs on this request's own thread (ThreadingHTTPServer), so a tab
            waiting here never blocks other requests."""
            q = dash.hub.subscribe()
            self.close_connection = True
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                self.wfile.write(b"retry: 3000\n" + sse({"topic": "hello"}))   # reconnect after 3s if dropped
                self.wfile.flush()
                while True:
                    try:
                        notice = q.get(timeout=KEEPALIVE_SECONDS)
                    except queue.Empty:
                        self.wfile.write(sse(comment="keepalive"))
                    else:
                        if notice is None:           # server shutting down
                            break
                        self.wfile.write(sse(notice))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass                                 # the tab closed or reloaded
            finally:
                dash.hub.unsubscribe(q)

        # ---- POST

        def do_POST(self) -> None:
            if not self._allowed():
                return
            parts, _ = self._route()
            body = self._body()
            if body is None:
                return
            if parts == ["api", "chat"]:
                return self._chat(body)
            try:
                if parts == ["api", "feedback"]:
                    value = int(body.get("value", 0))
                    return self._json(agent.feedback(str(body.get("turn_id")), value,
                                                     str(body.get("note") or "")[:2000]))
                if parts == ["api", "judge"]:
                    return self._json(agent.judge_turn(str(body.get("turn_id"))))
                if parts == ["api", "to-eval"]:
                    return self._json(agent.add_to_evals(str(body.get("turn_id"))))
                if parts == ["api", "judge-check"]:
                    return self._json(dash.judge.label(int(body.get("id", -1)), str(body.get("human")),
                                                       str(body.get("note") or "")))
                if parts == ["api", "evals", "run"]:
                    dash.start_eval(str(body.get("dataset")))
                    return self._json({"started": body.get("dataset")}, 202)
                return self._error(404, "not found")
            except KeyError as exc:
                return self._error(404, str(exc).strip("'\""))
            except (ValueError, RuntimeError) as exc:
                return self._error(400, str(exc))
            except Exception as exc:
                return self._error(500, f"{type(exc).__name__}: {exc}")

        def _chat(self, body: dict) -> None:
            message = str(body.get("message") or "").strip()
            if not message:
                return self._error(400, "empty message")
            session_id = body.get("session_id") or None

            self.close_connection = True
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()

            def emit(kind: str, data: dict) -> None:
                line = json.dumps({"kind": kind, **data}, default=str)
                try:
                    self.wfile.write(f"data: {line}\n\n".encode())
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass   # the tab closed; the turn still finishes and is saved

            turn = agent.respond(message, session_id, source="chat", listener=emit,
                                 long=bool(body.get("long")))
            emit("done", _summary(turn))

        # ---- DELETE

        def do_DELETE(self) -> None:
            if not self._allowed():
                return
            parts, _ = self._route()
            with dash.read_lock:
                # ids are numbers in SQLite and Postgres, UUIDs in LangGraph's store
                if len(parts) == 4 and parts[:2] == ["api", "memory"] and re.fullmatch(r"[\w-]{1,64}", parts[3]):
                    kind, row = parts[2], int(parts[3]) if parts[3].isdigit() else parts[3]
                    delete = {"fact": dash.memory.delete_fact, "episode": dash.memory.delete_episode,
                              "lesson": dash.lessons.delete}.get(kind)
                    if delete:
                        delete(row)
                        return self._json({"deleted": kind, "id": row})
                if len(parts) == 3 and parts[:2] == ["api", "skills"]:
                    if agent.skills.delete(parts[2]):
                        return self._json({"deleted": parts[2]})
                    return self._error(404, "no user skill by that name (bundled skills can't be deleted)")
                if len(parts) == 3 and parts[:2] == ["api", "turns"]:
                    agent.tracer.delete(parts[2])
                    return self._json({"deleted": parts[2]})
            return self._error(404, "not found")

    return Handler


def make_server(agent: Sora, port: int, watch_interval: float = 1.0) -> ThreadingHTTPServer:
    dashboard = Dashboard(agent, watch_interval)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(dashboard))
    server.daemon_threads = True
    server.dashboard = dashboard          # so whoever stops the server can stop the watcher
    return server


def serve(agent: Sora, port: int) -> None:
    server = make_server(agent, port)
    print(f"sora dashboard: http://127.0.0.1:{port}  (Ctrl+C to stop)", flush=True)
    # WhatsApp, when configured, runs in this same process: one agent, one memory.
    from harness import whatsapp
    started = whatsapp.start(agent, agent.settings)
    if started:
        print(f"sora WhatsApp webhook: http://127.0.0.1:{agent.settings.whatsapp_port}{whatsapp.PATH} "
              f"via {started[1].provider.name} (expose it with a tunnel)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.dashboard.close()        # ends open event streams and the watcher
        server.server_close()
        agent.close()


def main(home: Path | None = None, port: int | None = None) -> int:
    settings = load_settings(home)
    from harness.cli import setup_logging
    setup_logging(settings.home / "logs")
    serve(create(home), port or settings.port)
    return 0
