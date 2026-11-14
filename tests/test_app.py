"""The whole turn, end to end, with a scripted model: every part wired together."""

import pytest

from fake import FakeClient, message, text, tool_use
from harness.app import NO_KEY, Settings, Sora, build_system, explain, system_text
from memory.gate import GateResult


class SmallModel:
    """Answers the small-model helpers by recognising their prompts."""

    def __init__(self, gate=None, lessons=None, consolidation=None, grade=4):
        self.gate = gate or {"retrieve": True, "query": "", "correction": False, "reason": "r"}
        self.lessons = lessons or []
        self.consolidation = consolidation or {"facts": [], "episode": ""}
        self.grade = grade
        self.seen = []

    def __call__(self, request):
        prompt = request["messages"][0]["content"]
        for key, answer in (("You decide whether", self.gate),
                            ("review a turn", {"lessons": self.lessons}),
                            ("long-term memory of a personal", self.consolidation),
                            ("Grade a personal", {"score": self.grade, "reason": "fine"})):
            if key in prompt:
                self.seen.append(key)
                return answer
        raise AssertionError(f"unexpected small-model prompt: {prompt[:80]}")


def make(tmp_path, script, small=None, **settings):
    small = small or SmallModel()
    client = FakeClient(script, small=small)
    agent = Sora(Settings(home=tmp_path, **settings), client, connect_mcp=False)
    return agent, client, small


def test_full_turn_with_memory_tools_and_trace(tmp_path):
    agent, client, small = make(tmp_path, [
        message(tool_use("remember", {"fact": "Aryan likes masala chai"}), stop_reason="tool_use"),
        message(text("Noted, masala chai it is.")),
    ])
    agent.store.add_fact("Aryan lives in Pune")
    live = []
    turn = agent.respond("I like masala chai. Where do I live?",
                         listener=lambda kind, data: live.append(kind))
    assert turn["status"] == "ok" and turn["reply"] == "Noted, masala chai it is."
    assert turn["checks"] == {"replied": True, "no_tool_errors": True, "within_step_limit": True}
    # memory recall reached the system prompt as labelled data
    system = system_text(client.requests[0]["system"])
    assert "## Memory" in system and "- Aryan lives in Pune" in system
    assert "## Skills" in system and "research-brief" in system
    # the tool ran and was traced with provenance
    assert any("masala chai" in f["content"] and f["source"] == f"remember:{turn['id']}"
               for f in agent.store.list_facts())
    kinds = [s["kind"] for s in turn["spans"]]
    assert kinds[:4] == ["turn", "llm", "gate", "context"] and "tool" in kinds and "reply" in kinds
    assert live[0] == "turn" and "text" in live
    # the exchange is saved and linked to the trace
    assert agent.store.messages(turn["session_id"])[-1]["turn_id"] == turn["id"]
    assert small.seen == ["You decide whether"]                # nothing went wrong: no reflection


def test_history_carries_across_turns(tmp_path):
    agent, client, _ = make(tmp_path, [message(text("Hi Meera's friend")), message(text("SQL"))])
    first = agent.respond("My colleague Meera is great at SQL.")
    agent.respond("What is she good at?", session_id=first["session_id"])
    sent = client.requests[1]["messages"]
    assert [m["role"] for m in sent] == ["user", "assistant", "user"]
    assert sent[0]["content"] == "My colleague Meera is great at SQL."


def test_tool_error_triggers_reflection_and_lesson_returns(tmp_path):
    lesson = {"situation": "loading skills", "lesson": "Check the skill catalog for exact names",
              "tool": "load_skill"}
    agent, client, small = make(tmp_path, [
        message(tool_use("load_skill", {"name": "nope"}), stop_reason="tool_use"),
        message(text("Sorry, no such skill.")),
        message(text("ok")),
    ], small=SmallModel(lessons=[lesson]))
    turn = agent.respond("use the nope skill")
    assert turn["checks"]["no_tool_errors"] is False and turn["tool_errors"] == 1
    assert "review a turn" in small.seen
    assert [s["kind"] for s in turn["spans"]].count("learning") == 1
    # next turn: the lesson is in load_skill's description
    agent.respond("hello")
    load_skill = next(t for t in client.requests[-1]["tools"] if t["name"] == "load_skill")
    assert "Check the skill catalog for exact names" in load_skill["description"]


def test_correction_reflects_with_previous_reply(tmp_path):
    small = SmallModel()
    agent, client, _ = make(tmp_path, [message(text("51")), message(text("Sorry, 68."))], small=small)
    first = agent.respond("17 times 4?")
    small.gate = {"retrieve": False, "query": "", "correction": True, "reason": "user says wrong"}
    agent.respond("No, that's wrong.", session_id=first["session_id"])
    reflection = next(r for r in client.small_requests if "review a turn" in r["messages"][0]["content"])
    assert "Assistant: 51" in reflection["messages"][0]["content"]


def test_consolidation_runs_on_schedule(tmp_path):
    small = SmallModel(consolidation={"facts": ["Aryan drinks chai daily"], "episode": "Chatted"})
    agent, _, _ = make(tmp_path, [message(text("ok"))] * 3, small=small, consolidate_every=2)
    session = None
    for word in ("a", "b"):
        session = agent.respond(word, session_id=session)["session_id"]
    assert "long-term memory of a personal" in small.seen
    assert agent.store.search_facts("chai") == ["Aryan drinks chai daily"]


def test_failure_is_traced_not_raised(tmp_path):
    def broken(request):
        raise RuntimeError("Could not resolve authentication method")
    agent, _, _ = make(tmp_path, broken)
    turn = agent.respond("hi")
    assert turn["status"] == "error" and turn["error"] == NO_KEY
    assert turn["spans"][-1]["kind"] == "error"
    assert agent.store.messages(turn["session_id"]) == []        # failed turns aren't history


def test_thumbs_down_reflects_and_case_export(tmp_path):
    lesson = {"situation": "the user asks for times", "lesson": "Call current_time first", "tool": ""}
    agent, _, small = make(tmp_path, [message(text("It's probably noon."))],
                           small=SmallModel(lessons=[lesson]))
    turn = agent.respond("what time is it?")
    result = agent.feedback(turn["id"], -1, "you guessed")
    assert result["lessons"] == [lesson]
    assert agent.tracer.turn(turn["id"])["feedback"] == -1
    assert agent.lessons.list()[0]["signal"] == "thumbs_down"
    case = agent.add_to_evals(turn["id"])
    assert "you guessed" in case["criteria"]
    assert (tmp_path / "evals" / "mine.jsonl").exists()
    assert agent.judge_turn(turn["id"])["score"] == 0.75
    with pytest.raises(KeyError):
        agent.feedback("missing", 1)


def test_refusal_and_limit_statuses(tmp_path):
    agent, _, _ = make(tmp_path, [message(text(""), stop_reason="refusal")])
    assert agent.respond("x")["status"] == "refused"
    loop_forever = lambda request: (message(text("done")) if request.get("tool_choice")
                                    else message(tool_use("current_time", {}), stop_reason="tool_use"))
    agent, _, _ = make(tmp_path / "b", loop_forever, max_iterations=2)
    turn = agent.respond("y")
    assert turn["status"] == "limit" and turn["checks"]["within_step_limit"] is False


def test_status_and_system_prompt_order(tmp_path):
    agent, _, _ = make(tmp_path, [])
    status = agent.status()
    assert {"remember", "web_search"} <= {t["name"] for t in status["tools"]}
    assert {"research-brief", "paper-research"} <= {k["name"] for k in status["skills"]}
    found = GateResult(True, "q", False, "r", facts=["f1"], episodes=["(2026-11-01) e1"])
    from datetime import datetime
    blocks = build_system("SOUL", datetime(2026, 11, 9, 10, 0), "- s: d", [], ["When x: y"], found)
    system = system_text(blocks)
    assert system.index("SOUL") < system.index("## Skills") < system.index("## Lessons") \
        < system.index("## Memory")
    # the cached block holds only what's stable; the clock and memory come after it
    assert blocks[0]["cache_control"] == {"type": "ephemeral"} and "cache_control" not in blocks[1]
    assert "Current local time" not in blocks[0]["text"] and "## Memory" not in blocks[0]["text"]
    later = build_system("SOUL", datetime(2026, 11, 9, 10, 1), "- s: d", [], [], None)
    assert later[0] == blocks[0]                       # a minute later: the cached prefix is identical
    assert "Past conversations:\n- (2026-11-01) e1" in system


def test_explain_messages():
    assert explain(RuntimeError("missing api_key")) == NO_KEY
    assert explain(ValueError("x")) == "ValueError: x"


def test_with_workspace_mcp(tmp_path):
    client = FakeClient([
        message(tool_use("workspace__write_file", {"path": "todo.md", "content": "- milk"}),
                stop_reason="tool_use"),
        message(text("Saved todo.md")),
    ], small=SmallModel())
    agent = Sora(Settings(home=tmp_path), client)
    try:
        turn = agent.respond("save a todo")
        assert turn["status"] == "ok" and turn["tool_errors"] == 0
        assert (tmp_path / "workspace" / "todo.md").read_text() == "- milk"
        assert agent.status()["mcp"][0]["status"] == "connected"
    finally:
        agent.close()


def test_web_limits_reach_the_request(tmp_path, monkeypatch):
    from harness.app import load_settings
    monkeypatch.setenv("SORA_WEB_MAX_SEARCHES", "2")
    monkeypatch.setenv("SORA_WEB_FETCH_MAX_TOKENS", "0")
    settings = load_settings(tmp_path)
    assert (settings.web_max_searches, settings.web_max_fetches, settings.web_fetch_max_tokens) == (2, 4, 0)
    client = FakeClient([message(text("ok"))], small=SmallModel())
    agent = Sora(settings, client, connect_mcp=False)
    agent.respond("hi")
    tools = {t["name"]: t for t in client.requests[0]["tools"]}
    assert tools["web_search"]["max_uses"] == 2
    assert tools["web_fetch"]["max_uses"] == 4 and "max_content_tokens" not in tools["web_fetch"]
    assert agent.status()["web_limits"] == {"searches": 2, "fetches": 4, "fetch_tokens": 0}


def test_action_note_and_history(tmp_path):
    from harness.app import action_note
    note = action_note([{"tool": "remember", "args": {"fact": "x" * 200}, "output": "Remembered", "is_error": False},
                        {"tool": "forget", "args": {"text": "y"}, "output": "nope", "is_error": True}],
                       [{"tool": "web_search", "args": {"query": "vllm"}}])
    assert note.startswith("[Recorded by the app, not written by you.")
    assert "remember(fact='" in note and "…" in note and "forget(text='y') -> failed: nope" in note
    assert "web_search('vllm')" in note
    assert action_note([], []) is None

    agent, client, _ = make(tmp_path, [
        message(tool_use("remember", {"fact": "Aryan likes chai"}), stop_reason="tool_use"),
        message(text("Saved!")), message(text("You're welcome")),
    ])
    first = agent.respond("remember I like chai")
    agent.respond("thanks", session_id=first["session_id"])
    previous = client.requests[-1]["messages"][1]["content"]
    assert previous.startswith("Saved!\n\n[Recorded by the app") and "remember(fact='Aryan likes chai')" in previous
