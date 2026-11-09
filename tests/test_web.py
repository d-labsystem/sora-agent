"""The web gateway over real HTTP: security checks, chat streaming, the API."""

import json
import threading
import time
import urllib.request
from urllib.error import HTTPError

import pytest

from fake import FakeClient, message, text, tool_use
from harness.app import Settings, Sora
from harness.web import make_server
from test_app import SmallModel


@pytest.fixture
def server(tmp_path):
    def script(request):
        last = request["messages"][-1]["content"]
        if isinstance(last, list):
            return message(text("Saved it."))
        if "remember" in last:
            return message(tool_use("remember", {"fact": "Aryan likes chai"}), stop_reason="tool_use")
        return message(text("Hello from sora"))

    agent = Sora(Settings(home=tmp_path), FakeClient(script, small=SmallModel(
        lessons=[{"situation": "s", "lesson": "Be concrete", "tool": ""}])), connect_mcp=False)
    srv = make_server(agent, 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", agent
    srv.shutdown()
    srv.server_close()


def call(base, path, body=None, method=None, headers=None):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(base + path, data=data, method=method, headers={
        **({"Content-Type": "application/json"} if data else {}), **(headers or {})})
    with urllib.request.urlopen(request, timeout=10) as res:
        raw = res.read().decode()
        return res.status, res.headers.get("Content-Type"), raw


def events(raw):
    return [json.loads(line[6:]) for line in raw.split("\n\n") if line.startswith("data: ")]


def test_page_and_static_files(server):
    base, _ = server
    status, kind, body = call(base, "/")
    assert status == 200 and "text/html" in kind and "/static/app.js" in body
    assert "javascript" in call(base, "/static/app.js")[1]
    with pytest.raises(HTTPError) as err:
        call(base, "/static/..%2F..%2Fweb.py")
    assert err.value.code == 404


@pytest.mark.parametrize("headers", [{"Host": "evil.com"}, {"Origin": "https://evil.com"},
                                     {"Host": "localhost.evil.com"}])
def test_other_sites_are_refused(server, headers):
    base, _ = server
    with pytest.raises(HTTPError) as err:
        call(base, "/api/status", headers=headers)
    assert err.value.code == 403


def test_post_must_be_json(server):
    base, _ = server
    request = urllib.request.Request(base + "/api/chat", data=b"message=hi", method="POST",
                                     headers={"Content-Type": "application/x-www-form-urlencoded"})
    with pytest.raises(HTTPError) as err:
        urllib.request.urlopen(request, timeout=10)
    assert err.value.code == 415


def test_chat_streams_events_then_api_reads(server):
    base, agent = server
    status, kind, raw = call(base, "/api/chat", {"message": "please remember I like chai"})
    assert status == 200 and kind == "text/event-stream"
    evs = events(raw)
    kinds = [e["kind"] for e in evs]
    assert kinds[0] == "turn" and "tool" in kinds and "text" in kinds and kinds[-1] == "done"
    done = evs[-1]
    assert done["status"] == "ok" and done["reply"] == "Saved it." and "spans" not in done
    session = done["session_id"]

    # the same conversation continues
    call(base, "/api/chat", {"message": "hi again", "session_id": session})
    msgs = json.loads(call(base, f"/api/sessions/{session}")[2])
    assert [m["content"] for m in msgs] == ["please remember I like chai", "Saved it.", "hi again", "Hello from sora"]
    assert json.loads(call(base, "/api/sessions")[2])[0]["id"] == session

    turns = json.loads(call(base, "/api/turns?source=chat")[2])
    assert len(turns) == 2
    detail = json.loads(call(base, f"/api/turns/{done['id']}")[2])
    assert any(s["kind"] == "tool" for s in detail["spans"])
    stats = json.loads(call(base, "/api/stats")[2])
    assert stats["turns"] == 2 and stats["tool_calls"] == 1
    memory = json.loads(call(base, "/api/memory")[2])
    assert memory["facts"][0]["content"] == "Aryan likes chai"
    export = call(base, "/api/export")[2].strip().splitlines()
    assert len(export) == 2


def test_feedback_judge_to_eval_and_deletes(server):
    base, agent = server
    done = events(call(base, "/api/chat", {"message": "hello"})[2])[-1]
    fb = json.loads(call(base, "/api/feedback", {"turn_id": done["id"], "value": -1, "note": "vague"})[2])
    assert fb["lessons"][0]["lesson"] == "Be concrete"
    assert json.loads(call(base, "/api/judge", {"turn_id": done["id"]})[2])["grade"] == 4
    assert json.loads(call(base, "/api/to-eval", {"turn_id": done["id"]})[2])["id"] == f"from-{done['id']}"
    with pytest.raises(HTTPError) as err:
        call(base, "/api/feedback", {"turn_id": "nope", "value": 1})
    assert err.value.code == 404

    lesson = json.loads(call(base, "/api/memory")[2])["lessons"][0]
    call(base, f"/api/memory/lesson/{lesson['id']}", method="DELETE")
    assert json.loads(call(base, "/api/memory")[2])["lessons"] == []
    with pytest.raises(HTTPError) as err:
        call(base, "/api/skills/research-brief", method="DELETE")
    assert err.value.code == 404                        # bundled skills stay


def test_evals_listing_and_background_run(server):
    base, agent = server
    data = json.loads(call(base, "/api/evals")[2])
    assert any(d["name"] == "smoke.jsonl" and d["cases"] >= 5 for d in data["datasets"])
    # the case saved by to-eval is a dataset of "yours"
    done = events(call(base, "/api/chat", {"message": "hello"})[2])[-1]
    call(base, "/api/to-eval", {"turn_id": done["id"]})
    mine = next(d for d in json.loads(call(base, "/api/evals")[2])["datasets"] if d["name"] == "mine.jsonl")
    assert mine["kind"] == "yours" and mine["cases"] == 1
    assert call(base, "/api/evals/run", {"dataset": "mine.jsonl"})[0] == 202
    for _ in range(100):
        data = json.loads(call(base, "/api/evals")[2])
        if not data["running"] and data["runs"]:
            break
        time.sleep(0.1)
    run = json.loads(call(base, f"/api/evals/{data['runs'][0]['id']}")[2])
    assert run["cases"] == 1 and run["results"][0]["turn_id"]
    with pytest.raises(HTTPError) as err:
        call(base, "/api/evals/run", {"dataset": "missing.jsonl"})
    assert err.value.code == 404


def test_status(server):
    base, _ = server
    status = json.loads(call(base, "/api/status")[2])
    assert status["model"] == "claude-opus-5-5" and status["skills"]
