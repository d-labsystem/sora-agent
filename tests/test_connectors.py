"""Apple Notes, Google Calendar, Google sign-in and approvals, all offline."""

import base64
import hashlib
import json
import stat
import threading
import time
import urllib.parse
import urllib.request
from types import SimpleNamespace as NS

import pytest
from mcp.server.mcpserver.exceptions import ToolError as ServerToolError

from tools import google_auth, notes_server
from tools.approvals import Approvals
from tools.builtin import TurnContext
from tools.gcal_server import Calendar
from tools.registry import Tool, ToolRegistry

# ---- Apple Notes

NOTES = [
    {"id": "x-coredata://A/ICNote/p1", "title": "Trip to Pune", "folder": "Notes", "modified": "2026-11-10T09:00:00Z",
     "locked": False, "text": "Trip to Pune\nBook train with Priya. Pack chargers."},
    {"id": "x-coredata://A/ICNote/p2", "title": "Groceries", "folder": "Home", "modified": "2026-11-15T09:00:00Z",
     "locked": False, "text": "Groceries\nmilk, coffee, peanuts"},
    {"id": "x-coredata://A/ICNote/p3", "title": "Bank", "folder": "Notes", "modified": "2026-11-12T09:00:00Z",
     "locked": True, "text": None},
]


class FakeOsascript:
    def __init__(self):
        self.calls = []

    def __call__(self, script, *args):
        self.calls.append((script, args))
        if script == notes_server.ALL_NOTES:
            return json.dumps([{**n, "text": n["text"] if args[0] == "text" else None} for n in NOTES])
        if script == notes_server.ONE_NOTE:
            return json.dumps(next(n for n in NOTES if n["id"] == args[0]))
        if script == notes_server.CREATE:
            return json.dumps({"id": "x-coredata://A/ICNote/new", "folder": args[0]})
        raise AssertionError("unknown script")


def test_notes_list_search_and_read():
    run = FakeOsascript()
    n = notes_server.Notes(run)
    listed = n.list_notes().splitlines()
    assert listed[0].startswith("- Groceries  [Home, 2026-11-15]") and "(locked)" in listed[1]
    assert n.list_notes(folder="home").count("\n") == 0
    found = n.search_notes("priya train")
    assert "Trip to Pune" in found and "Book train with Priya" in found and "Groceries" not in found
    assert "No note contains" in n.search_notes("zebra")
    assert n.read_note("x-coredata://A/ICNote/p1").startswith("# Trip to Pune")
    assert "locked with a password" in n.read_note("x-coredata://A/ICNote/p3")
    with pytest.raises(ServerToolError):
        n.read_note("p1")


def test_creating_a_note_never_puts_text_inside_the_script():
    run = FakeOsascript()
    title, body = 'Plan"); Application("Finder").delete()', "line <b>1</b>\nline 2"
    out = notes_server.Notes(run).create_note(title, body)
    script, args = run.calls[-1]
    assert script == notes_server.CREATE                       # a fixed script, untouched
    assert args[0] == "Sora" and args[2] == title               # user text only as argv
    assert "&lt;b&gt;1&lt;/b&gt;" in args[1] and "<div>line 2</div>" in args[1]
    assert "in the 'Sora' folder" in out


def test_osascript_errors_become_clear_messages(monkeypatch):
    def fake_run(cmd, **kw):
        assert cmd[:4] == ["osascript", "-l", "JavaScript", "-e"] and cmd[5:] == ["arg"]
        return NS(returncode=1, stdout="", stderr="execution error: Not authorized to send Apple events to Notes. (-1743)")
    monkeypatch.setattr(notes_server.subprocess, "run", fake_run)
    with pytest.raises(ServerToolError, match="Privacy & Security > Automation"):
        notes_server.osascript("script", "arg")


# ---- Google Calendar

class FakeGoogle:
    def __init__(self, items=()):
        self.calls, self.items = [], list(items)

    def __call__(self, method, url, token, body=None):
        self.calls.append((method, url, token, body))
        if method == "POST":
            return {**body, "id": "ev1", "htmlLink": "https://calendar.google.com/ev1"}
        if "calendarList" in url:
            return {"items": [{"id": "me@x.com", "summary": "Aryan", "primary": True}]}
        return {"items": self.items}


def cal(google):
    return Calendar(NS(), zone="Asia/Kolkata", request=google, token=lambda: "tok")


def test_listing_shows_local_times_and_all_day_ranges():
    google = FakeGoogle([
        {"id": "a", "summary": "Standup", "start": {"dateTime": "2026-11-20T04:00:00Z"},
         "end": {"dateTime": "2026-11-20T04:30:00Z"}, "location": "Meet"},
        {"id": "b", "summary": "Goa", "start": {"date": "2026-11-21"}, "end": {"date": "2026-11-24"}}])
    out = cal(google).events("2026-11-20", "2026-11-23")
    assert "Fri 20 Nov 09:30–10:00  Standup (at Meet)" in out                  # UTC 04:00 = 09:30 IST
    assert "Sat 21 Nov – Mon 23 Nov, all day  Goa" in out                       # end date is exclusive
    params = urllib.parse.parse_qs(urllib.parse.urlsplit(google.calls[0][1]).query)
    assert params["timeMin"] == ["2026-11-20T00:00:00+05:30"]
    assert params["timeMax"] == ["2026-11-24T00:00:00+05:30"]                   # covers the whole last day
    assert params["singleEvents"] == ["true"] and google.calls[0][2] == "tok"


def test_creating_events_with_and_without_invitations():
    google = FakeGoogle()
    c = cal(google)
    c.create("Focus", "2026-11-20T15:00", "2026-11-20T16:00")
    method, url, _, body = google.calls[-1]
    assert "sendUpdates=none" in url and body["start"] == {"dateTime": "2026-11-20T15:00:00+05:30",
                                                           "timeZone": "Asia/Kolkata"}
    out = c.create("Dinner", "2026-11-21T20:00", "2026-11-21T22:00", attendees=["priya@x.com"])
    assert "sendUpdates=all" in google.calls[-1][1] and "Invitations emailed to priya@x.com" in out
    c.create("Trip", "2026-11-21", "2026-11-23")
    assert google.calls[-1][3]["end"] == {"date": "2026-11-24"}
    for bad in [("x", "2026-11-20T16:00", "2026-11-20T15:00"), ("x", "2026-11-20", "2026-11-20T15:00"),
                ("x", "tomorrow", "2026-11-20"), ("", "2026-11-20", "2026-11-20")]:
        with pytest.raises(ServerToolError):
            c.create(*bad)
    with pytest.raises(ServerToolError, match="email addresses"):
        c.create("x", "2026-11-20", "2026-11-20", attendees=["Priya"])


# ---- Google sign-in

def client_file(home):
    (home / google_auth.CLIENT_FILE).write_text(json.dumps({"installed": {"client_id": "cid", "client_secret": "cs"}}))


def test_login_uses_state_and_pkce_and_saves_a_private_token(tmp_path):
    client_file(tmp_path)
    posted = {}

    def browser(url):
        q = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))
        posted["challenge"] = q["code_challenge"]
        assert q["code_challenge_method"] == "S256" and q["access_type"] == "offline"
        assert "calendar.events" in q["scope"] and q["redirect_uri"].startswith("http://127.0.0.1:")
        # the browser comes back to sora's local server with the code
        threading.Thread(target=lambda: urllib.request.urlopen(
            f"{q['redirect_uri']}/?code=abc&state={q['state']}").read(), daemon=True).start()

    def post(url, fields):
        posted.update(fields)
        return {"access_token": "at", "refresh_token": "rt", "expires_in": 3600, "scope": "calendar"}

    assert google_auth.login(tmp_path, open_browser=browser, post=post, timeout=10) == "calendar"
    verifier_hash = base64.urlsafe_b64encode(hashlib.sha256(posted["code_verifier"].encode()).digest()).rstrip(b"=")
    assert verifier_hash.decode() == posted["challenge"] and posted["code"] == "abc"
    token_path = tmp_path / google_auth.TOKEN_FILE
    assert stat.S_IMODE(token_path.stat().st_mode) == 0o600
    assert google_auth.load_token(tmp_path)["refresh_token"] == "rt"


def test_login_rejects_an_answer_for_someone_elses_attempt(tmp_path):
    client_file(tmp_path)

    def browser(url):
        q = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))
        threading.Thread(target=lambda: urllib.request.urlopen(
            f"{q['redirect_uri']}/?code=evil&state=forged").read(), daemon=True).start()

    with pytest.raises(google_auth.AuthError, match="state mismatch"):
        google_auth.login(tmp_path, open_browser=browser, post=lambda *a: pytest.fail("must not trade the code"),
                          timeout=10)
    assert not (tmp_path / google_auth.TOKEN_FILE).exists()


def test_expired_tokens_are_refreshed_and_saved(tmp_path):
    client_file(tmp_path)
    google_auth.save_token(tmp_path, {"access_token": "old", "refresh_token": "rt", "expires_at": time.time() - 1})
    sent = []
    fresh = google_auth.access_token(tmp_path, post=lambda url, f: sent.append(f) or {"access_token": "new", "expires_in": 3600})
    assert fresh == "new" and sent[0]["grant_type"] == "refresh_token" and sent[0]["client_id"] == "cid"
    assert google_auth.access_token(tmp_path, post=lambda *a: pytest.fail("still valid")) == "new"
    with pytest.raises(google_auth.AuthError, match="google login"):
        google_auth.access_token(tmp_path / "nobody")


# ---- approvals

def guarded():
    calls = []
    registry = ToolRegistry()
    registry.register(Tool("gcal__create_event", "", {"type": "object", "properties": {}},
                           lambda **a: calls.append(a) or "created"))
    context = TurnContext("s1", "t1")
    clock = [1000.0]
    approvals = Approvals(context, clock=lambda: clock[0])
    approvals.guard(registry)
    for tool in approvals.tools():
        registry.register(tool)
    return registry, context, calls, clock, approvals


def test_events_without_attendees_run_at_once():
    registry, _, calls, _, _ = guarded()
    assert registry.execute("gcal__create_event", {"title": "Focus", "attendees": []}) == ("created", False)
    assert len(calls) == 1


def test_invites_wait_for_a_later_turn_and_run_exactly_as_drafted():
    registry, context, calls, _, approvals = guarded()
    out, _ = registry.execute("gcal__create_event",
                              {"title": "Dinner", "start": "a", "end": "b", "attendees": ["priya@x.com"]})
    assert out.startswith("NOT DONE YET") and calls == []
    draft_id = next(iter(approvals.drafts))
    out, error = registry.execute("approve", {"draft_id": draft_id})          # same turn: refused
    assert error and "hasn't replied" in out and calls == []
    context.turn_id = "t2"                                                    # the user answered
    assert registry.execute("approve", {"draft_id": draft_id}) == ("created", False)
    assert calls == [{"title": "Dinner", "start": "a", "end": "b", "attendees": ["priya@x.com"]}]
    out, error = registry.execute("approve", {"draft_id": draft_id})          # runs once only
    assert error and "No pending draft" in out


def test_drafts_stay_in_their_conversation_and_expire():
    registry, context, calls, clock, approvals = guarded()
    registry.execute("gcal__create_event", {"title": "x", "attendees": ["a@x.com"]})
    draft_id = next(iter(approvals.drafts))
    context.session_id, context.turn_id = "other", "t9"
    assert "another conversation" in registry.execute("approve", {"draft_id": draft_id})[0]
    context.session_id = "s1"
    clock[0] += 601
    assert "expired" in registry.execute("approve", {"draft_id": draft_id})[0] and calls == []


def test_the_agent_wires_the_gate_and_reports_connectors(tmp_path):
    from fake import FakeClient
    from harness.app import Settings, Sora
    agent = Sora(Settings(home=tmp_path, memory_mirror=False), FakeClient([]), connect_mcp=False)
    assert agent.registry.get("approve") is not None
    status = agent.status()["connectors"]
    assert status == {"notes": False, "calendar": False, "google_calendar": False, "google_signed_in": False,
                      "approval_needed_for": []}
    google_auth.save_token(tmp_path, {"refresh_token": "rt"})
    assert agent.calendar_on()                                                # on once signed in


# ---- the Mac's Calendar (EventKit faked)

class FakeEventKit:
    def __init__(self, status=3, events=()):
        self._status, self._events, self.saved = status, list(events), []

    def status(self):
        return self._status

    def calendars(self):
        return [{"id": "c1", "title": "Home", "account": "iCloud", "writable": True, "default": True},
                {"id": "c2", "title": "Holidays", "account": "Other", "writable": False, "default": False}]

    def events(self, start, end, ids=None):
        self.asked = (start, end, ids)
        return self._events

    def save(self, title, start, end, all_day, location, notes, calendar_id):
        self.saved.append((title, start, end, all_day, calendar_id))
        return {"id": "e1", "calendar": "Home"}


def mac(fake):
    from tools.calendar_server import MacCalendar
    return MacCalendar(fake, zone="Asia/Kolkata")


def ts(text):
    from datetime import datetime
    return datetime.fromisoformat(text).timestamp()


def test_mac_calendar_lists_every_occurrence_in_local_time():
    fake = FakeEventKit(events=[
        {"id": "w", "title": "Gym", "start": ts("2026-11-20T07:00+05:30"), "end": ts("2026-11-20T08:00+05:30"),
         "all_day": False, "location": "", "notes": "", "calendar": "Home", "recurring": True},
        {"id": "h", "title": "Guru Nanak Jayanti", "start": ts("2026-11-19T00:00+05:30"),
         "end": ts("2026-11-20T00:00+05:30"), "all_day": True, "location": "", "notes": "", "calendar": "Holidays",
         "recurring": False}])
    out = mac(fake).events("2026-11-19", "2026-11-20")
    assert "Thu 19 Nov, all day  Guru Nanak Jayanti (Holidays)" in out
    assert "Fri 20 Nov 07:00–08:00  Gym (Home, repeats)" in out
    assert out.index("Guru Nanak") < out.index("Gym")                          # in time order
    assert fake.asked[:2] == (ts("2026-11-19T00:00+05:30"), ts("2026-11-21T00:00+05:30"))
    assert "default for new events" in mac(fake).calendars() and "(read only)" in mac(fake).calendars()


def test_mac_calendar_creates_timed_and_all_day_events():
    fake = FakeEventKit()
    out = mac(fake).create("Focus", "2026-11-20T15:00", "2026-11-20T16:00")
    assert fake.saved[-1] == ("Focus", ts("2026-11-20T15:00+05:30"), ts("2026-11-20T16:00+05:30"), False, None)
    assert out == "Created: Fri 20 Nov 15:00–16:00  Focus (Home)"
    out = mac(fake).create("Goa", "2026-11-21", "2026-11-23")
    assert fake.saved[-1][1:4] == (ts("2026-11-21T00:00+05:30"), ts("2026-11-24T00:00+05:30"), True)
    assert "Sat 21 Nov – Mon 23 Nov, all day  Goa" in out
    with pytest.raises(ServerToolError):
        mac(fake).create("x", "2026-11-20T16:00", "2026-11-20T15:00")


def test_mac_calendar_explains_missing_permission():
    with pytest.raises(ServerToolError, match="calendar check"):
        mac(FakeEventKit(status=0)).events("2026-11-20", "2026-11-20")
