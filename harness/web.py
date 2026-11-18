"""The web gateway: a local dashboard to chat with sora and watch it work.

Standard-library HTTP server, no framework: the whole API fits on one screen,
and there's nothing to install. The page is static HTML + JS (harness/static).

    GET  /                          the dashboard
    POST /api/chat                  {message, session_id, long?} → server-sent events
    GET  /api/sessions[/<id>]       conversations, and one conversation's messages
    POST /api/feedback              {turn_id, value: 1|-1|0, note}
    POST /api/judge | /api/to-eval  {turn_id}
    GET  /api/claim-check[?id=N] | /api/claim-check/report;  POST /api/claim-check {id, human, note}
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
it (SORA_BIND changes the address for a container, which then publishes the port
on the host's 127.0.0.1 only: see compose.yaml). Two more checks stop *websites* you visit from using it through your own
browser: the Host header must be localhost (blocks DNS rebinding), and any
Origin must be localhost (blocks cross-site requests). POSTs must be JSON,
which a plain HTML form on another site can't send without an Origin.

Streaming. /api/chat answers with server-sent events: one `data: {json}` line
per trace event, so the page shows text and tool calls as they happen. If the
tab closes, the turn still finishes and is saved; only the display is lost.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import queue
import re
import threading
from dataclasses import replace
from datetime import datetime, timezone
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from harness.app import PROJECT, Settings, Sora, create, load_settings
from harness.events import KEEPALIVE_SECONDS, Hub, Watcher, sse
from memory.learning import LessonStore
from memory import mirror
from memory.backends import open_memory
from ops.evals import load_cases, run_eval
from ops.judge_check import JudgeCheck, format_report
from ops import claim_labels

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
        self.claims = claim_labels.ClaimLabels(agent.tracer)
        self.read_lock = threading.Lock()
        self.busy = threading.Lock()          # with logins on: one model-spending request at a time
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


COOKIE = "sora_session"
log = logging.getLogger(__name__)


class Site:
    """Whose sora a request is for. Auth off: the one dashboard, as before. Auth on: the
    logged-in user's, each with its own home (memory, traces, lessons, workspace...),
    built on first use and sharing the heavy parts (embedding model, API clients)."""

    def __init__(self, single: "Dashboard | None" = None, auth=None, base: Settings | None = None,
                 make_agent=None, watch_interval: float = 1.0):
        self.single, self.auth, self.base = single, auth, base
        self.make_agent = make_agent             # (name, home) -> Sora
        self.watch_interval = watch_interval
        self.dashboards: dict[str, Dashboard] = {}
        self.lock = threading.Lock()
        self.building: dict[str, threading.Lock] = {}      # one per name: a slow first build blocks only that user
        settings = base or (single.agent.settings if single else None)
        self.proxies = {p.strip() for p in (settings.trusted_proxies if settings else "").split(",") if p.strip()}
        extra = {h.strip().lower() for h in (settings.allowed_hosts if settings else "").split(",") if h.strip()}
        self.hosts = LOCAL_HOSTS | extra
        self.secure = bool(settings and settings.public_url.lower().startswith("https://"))

    def dashboard(self, name: str) -> "Dashboard":
        with self.lock:
            if name in self.dashboards:
                return self.dashboards[name]
            building = self.building.setdefault(name, threading.Lock())
        with building:
            with self.lock:
                if name in self.dashboards:
                    return self.dashboards[name]
            made = Dashboard(self.make_agent(name, self.auth.home(name)), self.watch_interval)
            with self.lock:
                self.dashboards[name] = made
            return made

    def close(self) -> None:
        for d in ([self.single] if self.single else []) + list(self.dashboards.values()):
            d.close()
            if d is not self.single:
                d.agent.close()


def make_handler(site: Site):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        dash: Dashboard          # set per request: the logged-in user's (or the only) dashboard
        user: str | None = None
        timeout = 30             # a client that stops sending can't hold a thread forever

        def log_message(self, *args) -> None:
            pass

        @property
        def agent(self) -> Sora:
            return self.dash.agent

        # ---- who is asking

        def _token(self) -> str | None:
            cookie = SimpleCookie()
            try:
                cookie.load(self.headers.get("Cookie", ""))
            except Exception:
                return None
            return cookie[COOKIE].value if COOKIE in cookie else None

        def _who(self, method: str, parts: list[str]) -> bool:
            """Set self.dash (and self.user). False when the request was answered here:
            not logged in. The login page and its files are public."""
            if site.auth is None:
                self.dash = site.single
                return True
            self.user = site.auth.session(self._token())
            if self.user:
                self.dash = site.dashboard(self.user)
                return True
            if method == "GET" and (not parts or parts[0] == "static"):
                return True                      # the login page and its files
            if parts and parts[0] == "api":
                self._error(401, "log in first")
            else:
                self._send(302, b"", "text/plain", {"Location": "/"})
            return False

        def _cookie(self, token: str, max_age: int) -> str:
            return (f"{COOKIE}={token}; Path=/; HttpOnly; SameSite=Strict; Max-Age={max_age}"
                    + ("; Secure" if site.secure else ""))

        def _client_ip(self) -> str:
            """The caller's address; behind a proxy listed in SORA_TRUSTED_PROXIES, the
            address that proxy saw (the last X-Forwarded-For entry). Anyone else's
            X-Forwarded-For is ignored: it's just a header they wrote."""
            peer = self.client_address[0]
            forwarded = self.headers.get("X-Forwarded-For", "")
            if peer in site.proxies and forwarded:
                return forwarded.split(",")[-1].strip() or peer
            return peer

        def _metered(self, run, long: bool = False):
            """Model calls a user starts: with logins on, one at a time per user, and only
            within today's cap (checked once the slot is theirs, so parallel requests
            can't all pass the check first). Long tasks are for admins."""
            if site.auth is None:
                return run()
            if long and not site.auth.is_admin(self.user):
                return self._error(403, "long tasks are for admins (they can spend more per turn)")
            if not self.dash.busy.acquire(blocking=False):
                return self._error(429, "one at a time: your last request is still running")
            try:
                cap = self.agent.settings.user_daily_usd
                today = datetime.now(timezone.utc).strftime("%Y-%m-%dT00:00:00")
                if cap > 0 and self.agent.tracer.spent_since(today) >= cap:
                    return self._error(429, f"today's limit of ${cap:.2f} is used up; it resets at 00:00 UTC")
                return run()
            finally:
                self.dash.busy.release()

        def _login(self, body: dict) -> None:
            from ops.auth import AuthError, SESSION_DAYS
            try:
                token = site.auth.login(str(body.get("name") or ""), str(body.get("password") or ""),
                                        self._client_ip())
            except AuthError as exc:
                return self._error(401, str(exc))
            self._send(200, json.dumps({"name": body.get("name")}).encode(), "application/json",
                       {"Set-Cookie": self._cookie(token, SESSION_DAYS * 86400)})

        # ---- plumbing

        def _workspace_image(self, relative: str) -> None:
            """Charts that run_code saved. Images only, and only from the workspace:
            the path is resolved and must stay inside it."""
            root = (self.agent.settings.home / "workspace").resolve()
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
            host_ok = _hostname(self.headers.get("Host", "")) in site.hosts
            origin = self.headers.get("Origin")
            origin_ok = origin is None or _hostname(urlsplit(origin).netloc) in site.hosts
            if host_ok and origin_ok:
                return True
            self._send(403, b"forbidden", "text/plain")
            return False

        def _body(self) -> dict | None:
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                self._error(415, "send JSON")
                return None
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = -1
            if length < 0:                       # rfile.read(-1) would read until the client stops
                self._error(400, "bad Content-Length")
                return None
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
            if parts == ["healthz"]:             # the container's health check: says nothing else
                return self._send(200, b"ok", "text/plain")
            if not self._who("GET", parts):
                return
            try:
                self._get(parts, q)
            except Exception:
                log.exception("GET %s failed", self.path)
                self._error(500, "something went wrong on the server (details are in its log)")

        def _get(self, parts: list[str], q: dict) -> None:
            if parts[:2] == ["api", "workspace"] and len(parts) > 2:
                return self._workspace_image("/".join(parts[2:]))
            if not parts:
                page = "index.html" if site.auth is None or self.user else "login.html"
                return self._send(200, (STATIC / page).read_bytes(), "text/html; charset=utf-8")
            if parts[0] == "static" and len(parts) == 2:
                file = (STATIC / parts[1]).resolve()
                if file.parent != STATIC.resolve() or not file.is_file():
                    return self._error(404, "not found")
                kind = mimetypes.guess_type(file.name)[0] or "application/octet-stream"
                return self._send(200, file.read_bytes(), kind)
            if parts[0] != "api" or len(parts) < 2:
                return self._error(404, "not found")
            name, rest = parts[1], parts[2:]
            tracer = self.agent.tracer
            if name == "me":
                return self._json({"auth": site.auth is not None, "name": self.user,
                                   "admin": bool(self.user and site.auth.is_admin(self.user))})
            if name == "status":
                return self._json(self.agent.status())
            if name == "events":
                return self._events()
            if name == "sessions":
                with self.dash.read_lock:
                    if rest:
                        return self._json(self.dash.memory.messages(rest[0]))
                    return self._json(self.dash.memory.list_sessions(int(q.get("limit", 50))))
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
                with self.dash.read_lock:
                    return self._json({"facts": self.dash.memory.list_facts(),
                                       "episodes": self.dash.memory.list_episodes(),
                                       "lessons": self.dash.lessons.list()})
            if name == "judge-check":
                if rest == ["report"]:
                    report = self.dash.judge.report()
                    return self._json({**report, "text": format_report(report)})
                label_id = int(q["id"]) if q.get("id", "").isdigit() else None
                return self._json({"progress": self.dash.judge.progress(), "item": self.dash.judge.item(label_id)})
            if name == "claim-check":
                if rest == ["report"]:
                    report = self.dash.claims.report()
                    return self._json({**report, "text": claim_labels.format_report(report)})
                label_id = int(q["id"]) if q.get("id", "").isdigit() else None
                return self._json({"progress": self.dash.claims.progress(), "item": self.dash.claims.item(label_id)})
            if name == "evals":
                if rest:
                    run = tracer.eval_run(rest[0])
                    return self._json(run) if run else self._error(404, "no such run")
                return self._json({"datasets": [{k: v for k, v in d.items() if k != "path"}
                                                 for d in self.dash.datasets()], "runs": tracer.eval_runs(),
                                   "running": self.dash.eval_running})
            return self._error(404, "not found")

        def _events(self) -> None:
            """Hold the connection open and write each notice as it arrives.
            Runs on this request's own thread (ThreadingHTTPServer), so a tab
            waiting here never blocks other requests."""
            q = self.dash.hub.subscribe()
            self.close_connection = True
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                self.wfile.write(b"retry: 3000\n" + sse({"topic": "hello"}))   # reconnect after 3s if dropped
                self.wfile.flush()
                token = self._token()
                while True:
                    if site.auth is not None and site.auth.session(token) != self.user:
                        break                        # logged out, password changed or account removed
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
                self.dash.hub.unsubscribe(q)

        # ---- POST

        def do_POST(self) -> None:
            if not self._allowed():
                return
            parts, _ = self._route()
            body = self._body()
            if body is None:
                return
            if site.auth is not None and parts == ["api", "login"]:
                return self._login(body)
            if not self._who("POST", parts):
                return
            if site.auth is not None and parts == ["api", "logout"]:
                site.auth.logout(self._token())
                return self._send(200, b'{"logged_out": true}', "application/json",
                                  {"Set-Cookie": self._cookie("", 0)})
            if parts == ["api", "chat"]:
                return self._metered(lambda: self._chat(body), long=bool(body.get("long")))
            try:
                if parts == ["api", "feedback"]:
                    value = int(body.get("value", 0))
                    run = lambda: self._json(self.agent.feedback(str(body.get("turn_id")), value,
                                                                 str(body.get("note") or "")[:2000]))
                    return self._metered(run) if value == -1 else run()     # a 👎 runs reflection
                if parts == ["api", "judge"]:
                    return self._metered(lambda: self._json(self.agent.judge_turn(str(body.get("turn_id")))))
                if parts == ["api", "to-eval"]:
                    return self._json(self.agent.add_to_evals(str(body.get("turn_id"))))
                if parts == ["api", "judge-check"]:
                    return self._json(self.dash.judge.label(int(body.get("id", -1)), str(body.get("human")),
                                                       str(body.get("note") or "")))
                if parts == ["api", "claim-check"]:
                    return self._json(self.dash.claims.label(int(body.get("id", -1)), str(body.get("human")),
                                                        str(body.get("note") or "")))
                if parts == ["api", "evals", "run"]:
                    if site.auth is not None and not site.auth.is_admin(self.user):
                        return self._error(403, "only an admin can run evals (they spend the shared API keys)")
                    self.dash.start_eval(str(body.get("dataset")))
                    return self._json({"started": body.get("dataset")}, 202)
                return self._error(404, "not found")
            except KeyError as exc:
                return self._error(404, str(exc).strip("'\""))
            except (ValueError, RuntimeError) as exc:
                return self._error(400, str(exc))
            except Exception:
                log.exception("POST %s failed", self.path)
                return self._error(500, "something went wrong on the server (details are in its log)")

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

            turn = self.agent.respond(message, session_id, source="chat", listener=emit,
                                 long=bool(body.get("long")))
            emit("done", _summary(turn))

        # ---- DELETE

        def do_DELETE(self) -> None:
            if not self._allowed():
                return
            parts, _ = self._route()
            if not self._who("DELETE", parts):
                return
            with self.dash.read_lock:
                # ids are numbers in SQLite and Postgres, UUIDs in LangGraph's store
                if len(parts) == 4 and parts[:2] == ["api", "memory"] and re.fullmatch(r"[\w-]{1,64}", parts[3]):
                    kind, row = parts[2], int(parts[3]) if parts[3].isdigit() else parts[3]
                    delete = {"fact": self.dash.memory.delete_fact, "episode": self.dash.memory.delete_episode,
                              "lesson": self.dash.lessons.delete}.get(kind)
                    if delete:
                        delete(row)
                        return self._json({"deleted": kind, "id": row})
                if len(parts) == 3 and parts[:2] == ["api", "skills"]:
                    if self.agent.skills.delete(parts[2]):
                        return self._json({"deleted": parts[2]})
                    return self._error(404, "no user skill by that name (bundled skills can't be deleted)")
                if len(parts) == 3 and parts[:2] == ["api", "turns"]:
                    self.agent.tracer.delete(parts[2])
                    return self._json({"deleted": parts[2]})
            return self._error(404, "not found")

    return Handler


LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def check_bind(settings: Settings) -> None:
    """Listening beyond this computer needs logins: anyone who can reach the port could
    otherwise read memory and spend the API keys. compose.yaml says, with
    SORA_ALLOW_UNAUTHENTICATED_BIND, that the port is only published on the host's 127.0.0.1."""
    if settings.bind not in LOOPBACK and not settings.auth and not settings.allow_unauthenticated_bind:
        raise SystemExit(f"refusing to listen on {settings.bind} without logins: set SORA_AUTH=1 and add "
                         "users (sora users add NAME), or keep SORA_BIND=127.0.0.1")


def make_server(agent: Sora, port: int, watch_interval: float = 1.0) -> ThreadingHTTPServer:
    """One user, as before."""
    check_bind(agent.settings)
    site = Site(Dashboard(agent, watch_interval))
    server = ThreadingHTTPServer((agent.settings.bind, port), make_handler(site))
    server.daemon_threads = True
    server.site = server.dashboard = site      # so whoever stops the server can stop the watchers
    return server


def user_agents(settings: Settings, client=None):
    """(name, home) -> that user's Sora. Off for everyone: what needs the Mac, starts
    programs (their own mcp.json) or the host's Docker."""
    from loop.providers import Providers
    from memory.embeddings import Embedder
    shared = Providers(client)
    embedder = Embedder(settings.models_dir, settings.embedding_model) if settings.embeddings else None

    def make(name: str, home: Path) -> Sora:
        user = replace(settings, home=home, notes=False, calendar=False, browser=False, sandbox="off",
                       user_mcp=False, auth=False)
        return Sora(user, shared.anthropic_client, embedder=embedder)
    return make


def make_auth_server(settings: Settings, port: int, watch_interval: float = 1.0, client=None,
                     make_agent=None, require_users: bool = False) -> ThreadingHTTPServer:
    """Many users: logins (ops/auth.py), one home per user under <home>/users/."""
    from ops.auth import Auth
    check_bind(settings)
    auth = Auth(settings.home)
    if require_users and not auth.users():
        raise SystemExit("logins are on but there are no users yet: add one with  sora users add NAME --admin")
    if settings.bind not in LOOPBACK and not settings.public_url.lower().startswith("https://"):
        log.warning("logins are on and sora listens on %s, but SORA_PUBLIC_URL isn't https: the session "
                    "cookie won't be marked Secure. Serve it through an https proxy and set SORA_PUBLIC_URL.",
                    settings.bind)
    site = Site(auth=auth, base=settings, watch_interval=watch_interval,
                make_agent=make_agent or user_agents(settings, client))
    server = ThreadingHTTPServer((settings.bind, port), make_handler(site))
    server.daemon_threads = True
    server.site = server.dashboard = site
    return server


def serve(agent: Sora | None, port: int, settings: Settings | None = None) -> None:
    """One user (agent given), or many (settings.auth: logins, a home per user)."""
    settings = settings or agent.settings
    server = make_server(agent, port) if not settings.auth else make_auth_server(settings, port, require_users=True)
    where = "127.0.0.1" if settings.bind in LOOPBACK or settings.bind == "0.0.0.0" else settings.bind
    print(f"sora dashboard: http://{where}:{port}" + ("  (logins on)" if settings.auth else "")
          + "  (Ctrl+C to stop)", flush=True)
    # WhatsApp, when configured, runs in this same process. With logins on it talks to
    # one user's sora (SORA_WHATSAPP_USER), or is off.
    from harness import whatsapp
    wa_agent = agent
    if settings.auth:
        known = {u["name"] for u in server.site.auth.users()}
        wa_agent = server.site.dashboard(settings.whatsapp_user).agent if settings.whatsapp_user in known else None
    started = whatsapp.start(wa_agent, wa_agent.settings) if wa_agent else None
    if started:
        print(f"sora WhatsApp webhook: http://127.0.0.1:{settings.whatsapp_port}{whatsapp.PATH} "
              f"via {started[1].provider.name} (expose it with a tunnel)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.site.close()             # ends open event streams and watchers (and users' agents)
        server.server_close()
        if agent:
            agent.close()


def main(home: Path | None = None, port: int | None = None) -> int:
    settings = load_settings(home)
    from harness.cli import setup_logging
    setup_logging(settings.home / "logs")
    check_bind(settings)                 # before loading anything
    if settings.auth:
        serve(None, port or settings.port, settings)
    else:
        serve(create(home), port or settings.port)
    return 0
