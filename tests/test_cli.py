"""The terminal gateway, driven with scripted input and a scripted model."""

import io
import json

from fake import FakeClient, message, text, tool_use
from harness.app import Settings, Sora
from harness.cli import main
from test_app import SmallModel


def factory(tmp_path, script):
    def create(home=None):
        client = FakeClient(script, small=SmallModel(lessons=[
            {"situation": "x", "lesson": "Ask before guessing", "tool": ""}]))
        return Sora(Settings(home=tmp_path), client, connect_mcp=False)
    return create


def run(tmp_path, argv, script, inputs=()):
    out = io.StringIO()
    feed = iter(inputs)

    def read(prompt):
        out.write(prompt)
        try:
            return next(feed)
        except StopIteration:
            raise EOFError
    code = main(argv, create=factory(tmp_path, script), out=out, read=read)
    return code, out.getvalue()


def test_ask_streams_reply_and_shows_tools(tmp_path):
    code, out = run(tmp_path, ["ask", "remember", "chai"], [
        message(tool_use("remember", {"fact": "Aryan likes chai"}), stop_reason="tool_use"),
        message(text("Saved: you like chai.")),
    ])
    assert code == 0
    assert "↳ remember ✓" in out and "Saved: you like chai." in out


def test_chat_session_and_commands(tmp_path):
    script = [message(text("Hello!")), message(text("Still here.")), message(text("Fresh start."))]
    code, out = run(tmp_path, ["chat"], script, [
        "hi", "/trace", "/facts", "again", "/sessions", "/bad you guessed", "/lessons",
        "/new", "third", "/skills", "/nope", "/quit"])
    assert code == 0
    assert "sora › Hello!" in out and "Still here." in out
    assert "model call 1" in out and "turn " in out                       # /trace
    assert "(nothing yet)" in out                                         # /facts
    assert "learned: Ask before guessing" in out                          # /bad reflects
    assert "[general] Ask before guessing" in out                         # /lessons
    assert "Started a new conversation." in out
    assert "research-brief (bundled)" in out and "Unknown command /nope" in out
    agent = Sora(Settings(home=tmp_path), FakeClient(), connect_mcp=False)
    sessions = agent.store.list_sessions()
    assert len(sessions) == 2                                             # /new split them
    assert sum(len(agent.store.messages(s["id"])) for s in sessions) == 6


def test_errors_are_shown_and_exit_code(tmp_path):
    def broken(request):
        raise RuntimeError("bad api_key")
    code, out = run(tmp_path, ["ask", "hi"], broken)
    assert code == 1 and "No valid API key" in out


def test_status(tmp_path):
    code, out = run(tmp_path, ["status"], [])
    assert code == 0 and "remember" in out and "web_search" in out


def test_export(tmp_path, monkeypatch):
    run(tmp_path, ["ask", "hi"], [message(text("hello"))])
    monkeypatch.setenv("SORA_HOME", str(tmp_path))
    out = io.StringIO()
    assert main(["export", "--source", "cli"], out=out) == 0
    record = json.loads(out.getvalue().splitlines()[0])
    assert record["message"] == "hi" and record["source"] == "cli"
