"""The WhatsApp gateway: message sora from your phone.

    phone ─► provider (Twilio or Meta) ─► public tunnel ─► this server :8766
                                                              │ verify, allowlist
                                                              ▼
    phone ◄── provider's send API ◄── format + split ◄── Sora.respond()

Why a separate server. The dashboard is built to be reachable only from this
Mac: it refuses any Host or Origin that isn't localhost, and its API can
delete memory and read every conversation. A webhook must be reachable from
the internet (through a tunnel), so it gets its own tiny server that answers
one path, /whatsapp, and nothing else. The dashboard is never exposed.

Who may talk to sora. Two checks on every request:
  1. it really comes from the provider: Twilio signs requests with your auth
     token (HMAC-SHA1), Meta with your app secret (HMAC-SHA256); a request
     that fails the check gets 403
  2. it comes from a number in SORA_WHATSAPP_ALLOWED; anyone else is ignored,
     because sora can read your files and memory

Answer now, work later. Providers wait ~15 seconds for the webhook to answer
and retry if it doesn't; a turn with web research takes longer. So the
webhook answers at once, a single worker thread runs the turn (one at a time,
in arrival order), and the reply goes out through the provider's send API.
Retried deliveries are recognised by message id and dropped.

One conversation per phone number, kept in <home>/whatsapp_sessions.json;
send /new to start a fresh one.

Providers are small adapters (verify, parse, send) so switching is one
setting, SORA_WHATSAPP_PROVIDER = twilio | meta.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import queue
import re
import threading
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from tools.visuals import VISUAL_TOOLS

log = logging.getLogger(__name__)

PATH = "/whatsapp"
MAX_BODY = 256 * 1024
SEEN = 1000                 # message ids remembered to drop retried deliveries
NEW = {"/new", "/reset"}

MEDIA_REPLY = "I can only read text messages for now. Send it as text and I'll help."
ERROR_REPLY = "Sorry, something went wrong on my side ({error}). Try again in a moment."
NEW_REPLY = "Started a new conversation. What's up?"
CHART_NOTE = "(There's a chart for this in the dashboard.)"


# ---- the message, whichever provider it came from

@dataclass
class Incoming:
    sender: str                 # digits only, with country code: "919800000000"
    text: str
    message_id: str
    name: str = ""
    media: bool = False         # an image, voice note, file...


@dataclass
class Parsed:
    messages: list[Incoming] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)    # delivery failures the provider reported
    reply: tuple[int, str, bytes] = (200, "text/plain", b"ok")


def digits(number: str) -> str:
    return re.sub(r"\D", "", number or "")


class SendError(Exception):
    pass


def http_post(url: str, data: bytes, headers: dict, timeout: float = 20) -> dict:
    """POST and return the JSON answer; SendError carries the provider's own
    message (e.g. Twilio 63015 'not in sandbox', Meta 131031 'account locked')."""
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        try:
            detail = json.loads(body)
            detail = detail.get("error", detail)
            message = detail.get("message") or body
            code = detail.get("code")
        except ValueError:
            message, code = body, None
        raise SendError(f"HTTP {exc.code}{f' code {code}' if code else ''}: {message[:300]}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise SendError(f"network: {exc}") from exc


# ---- providers

class Twilio:
    """Twilio's WhatsApp API (the sandbox while testing). Incoming messages are
    form posts; replies go out through the Messages API."""
    name = "twilio"
    limit = 1600                # characters per WhatsApp message on Twilio

    def __init__(self, account_sid: str, auth_token: str, sender: str = "whatsapp:+14155238886",
                 post=http_post):
        self.sid, self.token, self.sender, self.post = account_sid, auth_token, sender, post

    def signature(self, url: str, params: dict) -> str:
        """Twilio's scheme: HMAC-SHA1 over the full URL followed by every form
        field, sorted by name, as name+value; base64."""
        payload = url + "".join(k + params[k] for k in sorted(params))
        return base64.b64encode(hmac.new(self.token.encode(), payload.encode(), hashlib.sha1).digest()).decode()

    def verify_get(self, query: dict) -> tuple[int, str, bytes]:
        return 405, "text/plain", b"POST only"

    def parse(self, url: str, headers, body: bytes) -> Parsed | None:
        params = {k: v[0] for k, v in urllib.parse.parse_qs(body.decode("utf-8", "replace"),
                                                            keep_blank_values=True).items()}
        if not hmac.compare_digest(self.signature(url, params), headers.get("X-Twilio-Signature", "")):
            return None
        empty = (200, "text/xml", b'<?xml version="1.0" encoding="UTF-8"?><Response></Response>')
        if "MessageStatus" in params and "Body" not in params:     # a delivery status callback
            problem = ([f"Twilio {params.get('ErrorCode')}: delivery {params['MessageStatus']}"]
                       if params.get("ErrorCode") else [])
            return Parsed([], problem, empty)
        message = Incoming(digits(params.get("From", "")), params.get("Body", ""),
                           params.get("MessageSid", ""), params.get("ProfileName", ""),
                           int(params.get("NumMedia", "0") or 0) > 0)
        return Parsed([message], [], empty)

    def send(self, to: str, text: str) -> str:
        url = f"https://api.twilio.com/2010-04-01/Accounts/{self.sid}/Messages.json"
        auth = base64.b64encode(f"{self.sid}:{self.token}".encode()).decode()
        data = urllib.parse.urlencode({"From": self.sender, "To": f"whatsapp:+{to}", "Body": text}).encode()
        answer = self.post(url, data, {"Authorization": f"Basic {auth}",
                                       "Content-Type": "application/x-www-form-urlencoded"})
        return answer.get("sid", "")

    def mark_read(self, message_id: str) -> None:
        pass                    # Twilio's sandbox has no read receipts


class Meta:
    """Meta's WhatsApp Cloud API. Meta first checks the webhook with a GET
    (hub.challenge); messages then arrive as signed JSON."""
    name = "meta"
    limit = 4096

    LOCKED = 131031             # "Business Account locked": every outgoing message fails

    def __init__(self, access_token: str, phone_number_id: str, app_secret: str, verify_token: str,
                 version: str = "v23.0", post=http_post):
        self.token, self.phone_id, self.secret, self.verify_token = access_token, phone_number_id, app_secret, verify_token
        self.version, self.post = version, post

    def signature(self, body: bytes) -> str:
        return "sha256=" + hmac.new(self.secret.encode(), body, hashlib.sha256).hexdigest()

    def verify_get(self, query: dict) -> tuple[int, str, bytes]:
        if (query.get("hub.mode") == "subscribe" and self.verify_token
                and hmac.compare_digest(query.get("hub.verify_token", ""), self.verify_token)):
            return 200, "text/plain", query.get("hub.challenge", "").encode()
        return 403, "text/plain", b"forbidden"

    def parse(self, url: str, headers, body: bytes) -> Parsed | None:
        if not hmac.compare_digest(self.signature(body), headers.get("X-Hub-Signature-256", "")):
            return None
        parsed = Parsed()
        try:
            payload = json.loads(body)
        except ValueError:
            return parsed
        for entry in payload.get("entry", []):
            for change in entry.get("changes", []):
                value = change.get("value", {})
                names = {c.get("wa_id"): (c.get("profile") or {}).get("name", "") for c in value.get("contacts", [])}
                for m in value.get("messages", []):
                    kind = m.get("type")
                    parsed.messages.append(Incoming(
                        digits(m.get("from", "")), (m.get("text") or {}).get("body", "") if kind == "text" else "",
                        m.get("id", ""), names.get(m.get("from"), ""), kind != "text"))
                for status in value.get("statuses", []):
                    for error in status.get("errors", []):
                        hint = (" (Meta has locked the WhatsApp Business Account; see Business Support Home)"
                                if error.get("code") == self.LOCKED else "")
                        parsed.problems.append(f"Meta {error.get('code')}: {error.get('title', '')}{hint}")
        return parsed

    def _post(self, payload: dict) -> dict:
        return self.post(f"https://graph.facebook.com/{self.version}/{self.phone_id}/messages",
                         json.dumps(payload).encode(),
                         {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"})

    def send(self, to: str, text: str) -> str:
        answer = self._post({"messaging_product": "whatsapp", "recipient_type": "individual", "to": to,
                             "type": "text", "text": {"body": text, "preview_url": False}})
        return ((answer.get("messages") or [{}])[0]).get("id", "")

    def mark_read(self, message_id: str) -> None:
        self._post({"messaging_product": "whatsapp", "status": "read", "message_id": message_id})


# ---- formatting: Markdown in, WhatsApp out

def to_whatsapp(text: str) -> str:
    """WhatsApp knows *bold*, _italic_, ~strike~ and ```mono```, not Markdown.
    Headings become bold lines, links become 'text (url)', tables plain rows."""
    out, in_code = [], False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
            out.append("```")
            continue
        if in_code:
            out.append(line)
            continue
        if re.fullmatch(r"\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*", line):
            continue                                         # a table's |---|---| row
        if line.strip().startswith("|"):
            line = "  ".join(cell.strip() for cell in line.strip().strip("|").split("|"))
        heading = re.match(r"\s*#{1,6}\s+(.*)", line)
        if heading:
            line = f"**{heading.group(1).strip()}**"
        line = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", r"\1 (\2)", line)
        line = re.sub(r"(?<![*\w])\*(?!\*)([^*\n]+?)(?<!\*)\*(?![*\w])", r"_\1_", line)   # *it* -> _it_
        line = re.sub(r"\*\*(.+?)\*\*", r"*\1*", line)                                   # **b** -> *b*
        line = re.sub(r"__(.+?)__", r"*\1*", line)
        line = re.sub(r"~~(.+?)~~", r"~\1~", line)
        line = re.sub(r"^(\s*)[-+] ", r"\1• ", line)
        out.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def split(text: str, limit: int) -> list[str]:
    """Pieces of at most `limit` characters, numbered (1/3) when there is more
    than one. Whole paragraphs are kept together where they fit; a paragraph
    too long on its own is cut at lines, then words, then characters."""
    room = limit - 8                                         # space for " (12/12)"

    def units(block: str, seps=("\n\n", "\n", " ")) -> list[str]:
        """`block` as parts that each fit, cut at the coarsest separator possible."""
        if len(block) <= room:
            return [block]
        if not seps:
            return [block[i:i + room] for i in range(0, len(block), room)]
        sep, parts, out = seps[0], block.split(seps[0]), []
        for i, part in enumerate(parts):
            sub = units(part, seps[1:])
            if i < len(parts) - 1:
                sub[-1] += sep                               # the separator ends its own part only
            out.extend(sub)
        return out

    pieces, current = [], ""
    for unit in units(text.strip()):
        if current and len(current) + len(unit) > room:
            pieces.append(current.strip())
            current = ""
        current += unit
    if current.strip():
        pieces.append(current.strip())
    if len(pieces) > 1:
        pieces = [f"{p} ({i}/{len(pieces)})" for i, p in enumerate(pieces, 1)]
    return pieces or [""]


# ---- the gateway: who may talk, which conversation, one turn at a time

class Gateway:
    def __init__(self, agent, provider, allowed: list[str], sessions_path: Path):
        self.agent, self.provider = agent, provider
        self.allowed = {digits(n) for n in allowed if digits(n)}
        self.sessions_path = sessions_path
        self.sessions: dict[str, str] = self._load()
        self.seen: OrderedDict[str, None] = OrderedDict()
        self.queue: queue.Queue = queue.Queue()
        self.lock = threading.Lock()
        self.worker = threading.Thread(target=self._work, name="whatsapp-worker", daemon=True)
        self.worker.start()

    def _load(self) -> dict:
        try:
            return json.loads(self.sessions_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _save(self) -> None:
        tmp = self.sessions_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.sessions, indent=1), encoding="utf-8")
        tmp.replace(self.sessions_path)

    def accept(self, message: Incoming) -> str:
        """Decide what to do with one incoming message (fast: the webhook is
        waiting). Returns what happened, for logs and tests."""
        with self.lock:
            if message.message_id in self.seen:
                return "duplicate"
            self.seen[message.message_id] = None
            while len(self.seen) > SEEN:
                self.seen.popitem(last=False)
        if message.sender not in self.allowed:
            log.warning("WhatsApp message from a number not in SORA_WHATSAPP_ALLOWED (…%s); ignored",
                        message.sender[-4:])
            return "not allowed"
        self.queue.put(message)
        return "queued"

    def _work(self) -> None:
        while True:
            message = self.queue.get()
            if message is None:
                return
            try:
                self.handle(message)
            except Exception:
                log.exception("WhatsApp message %s failed", message.message_id)
            finally:
                self.queue.task_done()

    def _send(self, to: str, text: str) -> None:
        for piece in split(to_whatsapp(text), self.provider.limit):
            try:
                self.provider.send(to, piece)
            except SendError as exc:
                log.error("couldn't send a WhatsApp reply via %s: %s", self.provider.name, exc)
                return

    def handle(self, message: Incoming) -> None:
        try:
            self.provider.mark_read(message.message_id)
        except SendError as exc:
            log.warning("couldn't mark a WhatsApp message read: %s", exc)
        text = message.text.strip()
        if message.media and not text:
            return self._send(message.sender, MEDIA_REPLY)
        if text.lower() in NEW:
            self.sessions.pop(message.sender, None)
            self._save()
            return self._send(message.sender, NEW_REPLY)
        if not text:
            return
        turn = self.agent.respond(text, session_id=self.sessions.get(message.sender), source="whatsapp")
        if turn.get("session_id") and self.sessions.get(message.sender) != turn["session_id"]:
            self.sessions[message.sender] = turn["session_id"]
            self._save()
        reply = turn.get("reply") or ERROR_REPLY.format(error=turn.get("error") or turn.get("status"))
        if any(s.get("kind") == "tool" and s.get("name") in VISUAL_TOOLS for s in turn.get("spans", [])):
            reply += "\n\n" + CHART_NOTE
        self._send(message.sender, reply)

    def drain(self, timeout: float = 30) -> None:
        """Wait for queued messages to finish (tests, shutdown)."""
        done = threading.Event()
        threading.Thread(target=lambda: (self.queue.join(), done.set()), daemon=True).start()
        done.wait(timeout)

    def stop(self) -> None:
        self.queue.put(None)


# ---- the webhook server

def make_handler(gateway: Gateway, public_url: str = ""):
    provider = gateway.provider

    class Handler(BaseHTTPRequestHandler):
        server_version = "sora-whatsapp"

        def log_message(self, fmt, *args):          # quiet: no phone numbers in the console
            pass

        def _send(self, code: int, kind: str, body: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _url(self) -> str:
            """The URL the provider signed: the public one, not localhost.
            Behind a tunnel, Host and X-Forwarded-Proto say what it was."""
            if public_url:
                return public_url.rstrip("/") + PATH
            proto = self.headers.get("X-Forwarded-Proto", "https")
            return f"{proto}://{self.headers.get('Host', '')}{self.path}"

        def do_GET(self) -> None:
            path, _, query = self.path.partition("?")
            if path != PATH:
                return self._send(404, "text/plain", b"not found")
            self._send(*provider.verify_get({k: v[0] for k, v in urllib.parse.parse_qs(query).items()}))

        def do_POST(self) -> None:
            if self.path.partition("?")[0] != PATH:
                return self._send(404, "text/plain", b"not found")
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                return self._send(413, "text/plain", b"too large")
            body = self.rfile.read(length)
            parsed = provider.parse(self._url(), self.headers, body)
            if parsed is None:
                log.warning("WhatsApp webhook: bad signature; rejected (check the public URL and secret)")
                return self._send(403, "text/plain", b"bad signature")
            for problem in parsed.problems:
                log.error("WhatsApp delivery problem: %s", problem)
            for message in parsed.messages:
                gateway.accept(message)
            self._send(*parsed.reply)

    return Handler


def provider_from(settings, post=http_post):
    """The configured provider, or None with the reason it isn't set up."""
    kind = settings.whatsapp_provider or (
        "twilio" if settings.twilio_account_sid else "meta" if settings.whatsapp_access_token else "")
    if kind == "twilio":
        if not (settings.twilio_account_sid and settings.twilio_auth_token):
            return None, "set TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN"
        return Twilio(settings.twilio_account_sid, settings.twilio_auth_token, settings.whatsapp_from, post), ""
    if kind == "meta":
        missing = [n for n, v in (("WHATSAPP_ACCESS_TOKEN", settings.whatsapp_access_token),
                                  ("WHATSAPP_PHONE_NUMBER_ID", settings.whatsapp_phone_number_id),
                                  ("WHATSAPP_APP_SECRET", settings.whatsapp_app_secret),
                                  ("SORA_WHATSAPP_VERIFY_TOKEN", settings.whatsapp_verify_token)) if not v]
        if missing:
            return None, "set " + ", ".join(missing)
        return Meta(settings.whatsapp_access_token, settings.whatsapp_phone_number_id, settings.whatsapp_app_secret,
                    settings.whatsapp_verify_token, settings.whatsapp_graph_version, post), ""
    if kind:
        return None, f"SORA_WHATSAPP_PROVIDER must be twilio or meta, not {kind!r}"
    return None, "no WhatsApp provider configured"


def start(agent, settings) -> tuple[ThreadingHTTPServer, Gateway] | None:
    """Start the webhook server in the background; None (and why, in the log)
    when WhatsApp isn't configured."""
    provider, why = provider_from(settings)
    if provider is None:
        log.info("WhatsApp off: %s", why)
        return None
    if not settings.whatsapp_allowed:
        log.error("WhatsApp off: set SORA_WHATSAPP_ALLOWED to your number (e.g. +919800000000)")
        return None
    gateway = Gateway(agent, provider, settings.whatsapp_allowed.split(","), settings.home / "whatsapp_sessions.json")
    # 127.0.0.1: the tunnel connects from this Mac; nothing else on the network can.
    server = ThreadingHTTPServer(("127.0.0.1", settings.whatsapp_port), make_handler(gateway, settings.whatsapp_public_url))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, name="whatsapp-server", daemon=True).start()
    return server, gateway
