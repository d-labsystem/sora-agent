"""Multi-user sora: accounts, sessions, and that users can't see each other's data."""

import json
import re
import threading
import time
import urllib.request
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

from fake import FakeClient, message, text, tool_use
from harness.app import Settings, Sora
from harness.web import check_bind, make_auth_server
from ops.auth import Auth, AuthError, hash_password, verify_password
from test_app import SmallModel

PW = "correct horse battery"


# ---- accounts and sessions

def test_passwords_are_salted_hashes_never_plain_text(tmp_path):
    a, b = hash_password(PW), hash_password(PW)
    assert a != b and a.startswith("scrypt$") and PW not in a
    assert verify_password(PW, a) and not verify_password("wrong password", a)
    auth = Auth(tmp_path)
    auth.add_user("aryan", PW)
    assert PW.encode() not in (tmp_path / "auth.db").read_bytes()
    assert oct((tmp_path / "auth.db").stat().st_mode)[-3:] == "600"


def test_names_and_passwords_are_checked(tmp_path):
    auth = Auth(tmp_path)
    for bad in ("A", "aryan!", "../etc", "", "x" * 40):
        with pytest.raises(ValueError):
            auth.add_user(bad, PW)
    with pytest.raises(ValueError, match="at least 10"):
        auth.add_user("aryan", "short")
    assert auth.add_user("aryan", PW) == tmp_path / "users" / "aryan"
    with pytest.raises(ValueError, match="already"):
        auth.add_user("aryan", PW)


def test_sessions_are_stored_hashed_and_end_on_logout_or_password_change(tmp_path):
    auth = Auth(tmp_path)
    auth.add_user("aryan", PW)
    token = auth.login("aryan", PW)
    assert auth.session(token) == "aryan" and auth.session("made-up") is None and auth.session(None) is None
    assert token.encode() not in (tmp_path / "auth.db").read_bytes()
    auth.logout(token)
    assert auth.session(token) is None
    token = auth.login("aryan", PW)
    auth.set_password("aryan", "another long password")
    assert auth.session(token) is None
    auth.conn.execute("UPDATE sessions SET expires_at = 0")
    assert auth.session(auth.login("aryan", "another long password")) == "aryan"


def test_expired_sessions_are_refused(tmp_path):
    auth = Auth(tmp_path)
    auth.add_user("aryan", PW)
    token = auth.login("aryan", PW)
    auth.conn.execute("UPDATE sessions SET expires_at = ?", (time.time() - 1,))
    assert auth.session(token) is None


def test_wrong_passwords_slow_down_and_say_nothing_about_which_part_was_wrong(tmp_path):
    auth = Auth(tmp_path)
    auth.add_user("aryan", PW)
    for who in ("aryan", "nobody"):
        with pytest.raises(AuthError, match="^wrong name or password$"):
            auth.login(who, "nope", "1.2.3.4")
    for _ in range(5):
        with pytest.raises(AuthError, match="wrong"):
            auth.login("aryan", "nope", "5.6.7.8")
    with pytest.raises(AuthError, match="too many wrong tries"):
        auth.login("aryan", PW, "5.6.7.8")              # even the right password waits, from that IP
    # ...but a guesser can't lock the owner out: from their own IP it still works
    assert auth.session(auth.login("aryan", PW, "9.9.9.9")) == "aryan"
    auth.conn.execute("UPDATE failures SET last_at = 0")
    assert auth.session(auth.login("aryan", PW, "5.6.7.8")) == "aryan"
    assert auth.conn.execute("SELECT COUNT(*) FROM failures WHERE key LIKE 'pair:aryan|5.6.7.8'").fetchone()[0] == 0


def test_parallel_guesses_are_all_counted(tmp_path):
    """The review reproduced 60 parallel wrong guesses all reaching the password check."""
    auth = Auth(tmp_path)
    auth.add_user("aryan", PW)
    results = []

    def guess():
        try:
            auth.login("aryan", "nope", "6.6.6.6")
        except AuthError as exc:
            results.append(str(exc))
    threads = [threading.Thread(target=guess) for _ in range(30)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    count = auth.conn.execute("SELECT count FROM failures WHERE key = 'pair:aryan|6.6.6.6'").fetchone()[0]
    checked = sum(r == "wrong name or password" for r in results)
    assert checked <= 6 and count == checked            # the rest were turned away before scrypt


def test_one_ip_trying_many_names_is_slowed_too(tmp_path):
    auth = Auth(tmp_path)
    for i in range(30):
        with pytest.raises(AuthError, match="wrong"):
            auth.login(f"user{i}", "nope", "7.7.7.7")
    with pytest.raises(AuthError, match="too many"):
        auth.login("user99", "nope", "7.7.7.7")


# ---- the server

def script(request):
    last = request["messages"][-1]["content"]
    if isinstance(last, list):
        return message(text("Saved it."))
    if "remember" in last:
        return message(tool_use("remember", {"fact": f"secret fact: {last[-12:]}"}), stop_reason="tool_use")
    return message(text("Hello"))


@pytest.fixture
def site(tmp_path):
    settings = Settings(home=tmp_path, auth=True, user_daily_usd=0, trusted_proxies="127.0.0.1")
    agents = {}

    def make_agent(name, home):
        agents[name] = Sora(Settings(home=home), FakeClient(script, small=SmallModel()), connect_mcp=False)
        return agents[name]
    srv = make_auth_server(settings, 0, watch_interval=0.1, make_agent=make_agent)
    auth = srv.site.auth
    auth.add_user("alice", PW, admin=True)
    auth.add_user("bob", PW)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield SimpleNamespace(base=f"http://127.0.0.1:{srv.server_address[1]}", srv=srv, agents=agents, auth=auth)
    srv.site.close()
    srv.shutdown()
    srv.server_close()


def call(base, path, body=None, method=None, cookie=None, headers=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method=method, headers={
        **({"Content-Type": "application/json"} if data is not None else {}),
        **({"Cookie": f"sora_session={cookie}"} if cookie else {}), **(headers or {})})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, r.read(), dict(r.headers)
    except HTTPError as e:
        return e.code, e.read(), dict(e.headers)


def login(base, name):
    status, _, headers = call(base, "/api/login", {"name": name, "password": PW})
    assert status == 200
    cookie = headers["Set-Cookie"]
    return re.match(r"sora_session=([^;]+)", cookie).group(1), cookie


def chat(base, cookie, words):
    status, body, _ = call(base, "/api/chat", {"message": words}, cookie=cookie)
    assert status == 200
    done = [json.loads(l[6:]) for l in body.decode().splitlines() if l.startswith("data: ")][-1]
    return done


def test_logged_out_you_get_the_login_page_and_nothing_else(site):
    status, page, _ = call(site.base, "/")
    assert status == 200 and b"/static/login.js" in page
    assert call(site.base, "/static/login.js")[0] == 200
    for path in ("/api/status", "/api/turns", "/api/memory", "/api/me", "/api/events", "/api/sessions"):
        assert call(site.base, path)[0] == 401, path
    assert call(site.base, "/api/chat", {"message": "hi"})[0] == 401
    assert call(site.base, "/api/turns/x", method="DELETE")[0] == 401
    assert call(site.base, "/api/me", cookie="forged-token")[0] == 401


def test_the_session_cookie_is_httponly_and_samesite_strict(site):
    token, cookie = login(site.base, "alice")
    assert "HttpOnly" in cookie and "SameSite=Strict" in cookie and "Path=/" in cookie
    assert "Secure" not in cookie                       # plain http on localhost
    me = json.loads(call(site.base, "/api/me", cookie=token)[1])
    assert me == {"auth": True, "name": "alice", "admin": True}
    status, _, headers = call(site.base, "/api/logout", {}, cookie=token)
    assert status == 200 and "Max-Age=0" in headers["Set-Cookie"]
    assert call(site.base, "/api/me", cookie=token)[0] == 401          # ended on the server too
    assert call(site.base, "/api/login", {"name": "alice", "password": "nope"})[0] == 401


def test_other_sites_still_cant_use_a_logged_in_browser(site):
    token, _ = login(site.base, "alice")
    assert call(site.base, "/api/me", cookie=token, headers={"Origin": "https://evil.example"})[0] == 403
    assert call(site.base, "/api/me", cookie=token, headers={"Host": "evil.example"})[0] == 403


def get_routes() -> list[str]:
    """Every GET /api/<name> the server answers, read from its source, so a new route
    can't be left out of the isolation test."""
    source = (Path(__file__).parent.parent / "harness" / "web.py").read_text()
    return sorted(set(re.findall(r'if name == "([\w-]+)"', source)))


def test_users_see_only_their_own_data_on_every_route(site):
    a, _ = login(site.base, "alice")
    b, _ = login(site.base, "bob")
    turn = chat(site.base, a, "please remember alice-only-xyz")
    chat(site.base, b, "hello")
    alice = site.agents["alice"]
    alice.lessons.add("alice situation", "alice-lesson-xyz")
    (alice.settings.home / "workspace").mkdir(exist_ok=True)
    (alice.settings.home / "workspace" / "chart.png").write_bytes(b"\x89PNG alice")
    session = turn["session_id"]
    fact_id = alice.store.list_facts()[0]["id"]

    routes = get_routes()
    assert {"status", "turns", "memory", "sessions", "stats", "export", "evals", "me"} <= set(routes)
    paths = [f"/api/{r}" for r in routes if r != "events"] + [
        f"/api/turns/{turn['id']}", f"/api/sessions/{session}", "/api/workspace/chart.png",
        "/api/turns?limit=500", "/api/export?source=chat"]
    for path in paths:
        status, body, _ = call(site.base, path, cookie=b)
        assert b"only-xyz" not in body and b"alice-lesson-xyz" not in body and b"PNG alice" not in body, path
        assert turn["id"].encode() not in body and session.encode() not in body, path
    assert call(site.base, f"/api/turns/{turn['id']}", cookie=b)[0] == 404
    # deleting by alice's ids, as bob, touches only bob's own (empty) data
    call(site.base, f"/api/turns/{turn['id']}", method="DELETE", cookie=b)
    call(site.base, f"/api/memory/fact/{fact_id}", method="DELETE", cookie=b)
    assert alice.tracer.turn(turn["id"]) is not None and alice.store.list_facts()
    # ...and alice still sees all of hers
    assert b"only-xyz" in call(site.base, "/api/memory", cookie=a)[1]
    assert call(site.base, f"/api/turns/{turn['id']}", cookie=a)[0] == 200
    assert alice.settings.home != site.agents["bob"].settings.home


def test_only_admins_run_evals_and_long_tasks(site):
    b, _ = login(site.base, "bob")
    status, body, _ = call(site.base, "/api/evals/run", {"dataset": "smoke.jsonl"}, cookie=b)
    assert status == 403 and b"admin" in body
    status, body, _ = call(site.base, "/api/chat", {"message": "hi", "long": True}, cookie=b)
    assert status == 403 and b"long tasks" in body


def test_the_daily_cap_counts_deleted_turns_judging_and_thumbs_down(site):
    """The review: deleting today's turns gave the money back; judge and 👎 cost nothing."""
    a, _ = login(site.base, "alice")
    b, _ = login(site.base, "bob")
    turn = chat(site.base, b, "hello")
    bob = site.agents["bob"]
    bob.settings.__dict__["user_daily_usd"] = 0.01      # read per request
    bob.tracer.add_spend(0.02, "test")
    call(site.base, f"/api/turns/{turn['id']}", method="DELETE", cookie=b)
    for path, body in (("/api/chat", {"message": "hi"}), ("/api/judge", {"turn_id": turn["id"]}),
                       ("/api/feedback", {"turn_id": turn["id"], "value": -1})):
        status, text_, _ = call(site.base, path, body, cookie=b)
        assert status == 429 and b"limit" in text_, path
    assert call(site.base, "/api/chat", {"message": "hi"}, cookie=a)[0] == 200    # alice isn't capped


def test_judge_and_thumbs_down_go_on_the_ledger(tmp_path):
    agent = Sora(Settings(home=tmp_path), FakeClient(script, small=SmallModel()), connect_mcp=False)
    turn = agent.respond("hello")
    before = agent.tracer.spent_since("2000-01-01")
    agent.judge_turn(turn["id"])
    agent.feedback(turn["id"], -1, "meh")
    rows = agent.tracer.conn.execute("SELECT what FROM spend").fetchall()
    assert [r[0].split()[0] for r in rows].count("turn") == 1
    assert agent.tracer.spent_since("2000-01-01") >= before
    agent.tracer.delete(turn["id"])
    assert agent.tracer.spent_since("2000-01-01") >= before        # deleting the turn changes nothing
    agent.close()


def test_one_request_at_a_time_per_user(site):
    b, _ = login(site.base, "bob")
    chat(site.base, b, "hello")
    dash = site.srv.site.dashboard("bob")
    dash.busy.acquire()                                  # as if bob's last turn were still running
    try:
        status, body, _ = call(site.base, "/api/chat", {"message": "again"}, cookie=b)
        assert status == 429 and b"one at a time" in body
    finally:
        dash.busy.release()
    a, _ = login(site.base, "alice")
    assert call(site.base, "/api/chat", {"message": "hi"}, cookie=a)[0] == 200


def test_requests_before_login_are_bounded(site):
    req = urllib.request.Request(site.base + "/api/login", data=b"{}", method="POST",
                                 headers={"Content-Type": "application/json", "Content-Length": "-1"})
    with pytest.raises(HTTPError) as err:
        urllib.request.urlopen(req)
    assert err.value.code == 400
    assert call(site.base, "/healthz")[:2] == (200, b"ok")                 # public, and says nothing else
    assert site.srv.RequestHandlerClass.timeout == 30


def test_a_trusted_proxy_passes_on_the_real_address(site):
    for _ in range(5):
        call(site.base, "/api/login", {"name": "alice", "password": "nope"}, headers={"X-Forwarded-For": "8.8.8.8"})
    status, body, _ = call(site.base, "/api/login", {"name": "alice", "password": PW},
                           headers={"X-Forwarded-For": "8.8.8.8"})
    assert status == 401 and b"too many" in body                         # 8.8.8.8 is slowed...
    assert call(site.base, "/api/login", {"name": "alice", "password": PW},
                headers={"X-Forwarded-For": "4.4.4.4"})[0] == 200          # ...a different visitor isn't


def test_errors_and_eval_lists_show_no_server_details(site):
    a, _ = login(site.base, "alice")
    evals = json.loads(call(site.base, "/api/evals", cookie=a)[1])
    assert evals["datasets"] and all("path" not in d for d in evals["datasets"])
    site.agents["alice"].tracer.stats = lambda *a: 1 / 0
    status, body, _ = call(site.base, "/api/stats", cookie=a)
    assert status == 500 and b"ZeroDivision" not in body and b"log" in body


def test_logins_on_with_no_users_refuses_to_start(tmp_path):
    with pytest.raises(SystemExit, match="no users yet"):
        make_auth_server(Settings(home=tmp_path, auth=True), 0, require_users=True)


def test_users_get_no_mac_tools_no_own_mcp_and_no_docker(tmp_path):
    from harness.web import user_agents
    settings = Settings(home=tmp_path, auth=True, notes=True, calendar=True, browser=True, sandbox="on")
    agent = user_agents(settings, FakeClient([]))("carol", tmp_path / "users" / "carol")
    s = agent.settings
    assert (s.notes, s.calendar, s.browser, s.sandbox, s.user_mcp) == (False, False, False, "off", False)
    assert agent.registry.get("run_code") is None
    agent.close()


def test_listening_beyond_localhost_needs_logins(tmp_path):
    with pytest.raises(SystemExit, match="refusing to listen on 0.0.0.0"):
        check_bind(Settings(home=tmp_path, bind="0.0.0.0"))
    check_bind(Settings(home=tmp_path, bind="0.0.0.0", auth=True))
    check_bind(Settings(home=tmp_path, bind="0.0.0.0", allow_unauthenticated_bind=True))
    check_bind(Settings(home=tmp_path))


def test_users_command_adds_copies_and_never_takes_a_password_argument(tmp_path, monkeypatch):
    from harness.cli import users_command
    old = tmp_path / "old-home"
    old.mkdir()
    (old / "memory.db").write_text("x")
    (old / ".env").write_text("ANTHROPIC_API_KEY=secret")
    monkeypatch.setenv("SORA_HOME", str(tmp_path / "root"))
    args = lambda copy_from="", reuse=False: SimpleNamespace(home=None, action="add", name="aryan", admin=True,
                                                             copy_from=copy_from, reuse=reuse)
    out = StringIO()
    answers = iter([PW, PW])
    assert users_command(args(copy_from=str(old)), out, lambda _: next(answers)) == 0
    home = tmp_path / "root" / "users" / "aryan"
    assert (home / "memory.db").read_text() == "x" and not (home / ".env").exists()
    assert (old / "memory.db").read_text() == "x"                  # the original is untouched
    answers = iter([PW, "different password"])
    out = StringIO()
    assert users_command(SimpleNamespace(home=None, action="add", name="bea", admin=False, copy_from="", reuse=False),
                         out, lambda _: next(answers)) == 1
    assert "don't match" in out.getvalue()
    # a removed account's data isn't handed to a new person by accident
    from ops.auth import Auth as A
    A(tmp_path / "root").remove_user("aryan")
    answers = iter([PW, PW])
    out = StringIO()
    assert users_command(args(), out, lambda _: next(answers)) == 1 and "already has data" in out.getvalue()
    answers = iter([PW, PW])
    assert users_command(args(reuse=True), StringIO(), lambda _: next(answers)) == 0
