import re
"""The whole turn, end to end, with a scripted model: every part wired together."""

import pytest

from fake import FakeClient, message, text, tool_use
from harness.app import NO_KEY, Settings, Sora, build_system, explain, system_text
from memory.gate import GateResult


class SmallModel:
    """Answers the small-model helpers by recognising their prompts."""

    def __init__(self, gate=None, lessons=None, consolidation=None, grade=4, claims=None):
        self.gate = gate or {"retrieve": True, "query": "", "correction": False, "reason": "r"}
        self.lessons = lessons or []
        self.consolidation = consolidation or {"facts": [], "episode": ""}
        self.grade = grade
        self.claims = claims or []
        self.seen = []

    def __call__(self, request):
        prompt = request["messages"][0]["content"]
        for key, answer in (("You decide whether", self.gate),
                            ("review a turn", {"lessons": self.lessons}),
                            ("long-term memory of a personal", self.consolidation),
                            ("Grade a personal", {"score": self.grade, "reason": "fine"}),
                            ("You check whether a personal assistant did", {"claims": self.claims})):
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
    assert turn["checks"] == {"replied": True, "no_tool_errors": True, "within_step_limit": True,
                              "within_budget": True}
    # memory recall reached the system prompt as labelled data
    system = system_text(client.requests[0]["system"])
    assert "## Memory" in system and re.search(r"- \(\d{4}-\d\d-\d\d, today\) Aryan lives in Pune", system)   # dated
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
    # next turn: the lesson reaches the model in the per-turn system block, and
    # load_skill's description is untouched, so the cached tool prefix stays valid
    agent.respond("hello")
    request = client.requests[-1]
    load_skill = next(t for t in request["tools"] if t["name"] == "load_skill")
    assert "Check the skill catalog" not in load_skill["description"]
    assert "load_skill: Check the skill catalog for exact names" in request["system"][1]["text"]


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


def test_current_time_tool_sees_the_turns_date(tmp_path):
    """respond(now=...) sets the clock for the tool too, not only for the prompt."""
    from datetime import datetime, timezone
    seen = []

    def script(request):
        results = [b for m in request["messages"] if isinstance(m["content"], list)
                   for b in m["content"] if isinstance(b, dict) and b.get("type") == "tool_result"]
        if results:
            seen.append(str(results[-1]["content"]))
            return message(text("It's 2023."))
        return message(tool_use("current_time", {}), stop_reason="tool_use")
    agent, _, _ = make(tmp_path, script)
    agent.respond("what's the date?", now=datetime(2023, 5, 30, 9, 0, tzinfo=timezone.utc))
    assert seen and "2023" in seen[0]
    agent.respond("and now?")                         # a live turn: back to the real clock
    assert str(datetime.now().year) in seen[-1] and agent.context.now is None


def test_a_claimed_action_with_no_record_is_a_false_claim(tmp_path):
    """The reply says it saved a file; no tool ran. Before: every check passed."""
    claims = [{"claim": "saved notes.md", "evidence": 0, "why": "no tool was used"}]
    lesson = {"situation": "saving files", "lesson": "Call write_file before saying a file is saved", "tool": ""}
    agent, client, small = make(tmp_path, [message(text("I've saved your notes to notes.md."))],
                                small=SmallModel(claims=claims, lessons=[lesson]), claim_check=True)
    turn = agent.respond("save my notes")
    assert turn["checks"]["claims_supported"] is False
    span = next(s for s in turn["spans"] if s["kind"] == "claims")
    assert span["ok"] in (False, 0) and span["data"]["claims"][0]["claim"] == "saved notes.md"
    assert "review a turn" in small.seen
    review = next(r for r in client.small_requests if "review a turn" in r["messages"][0]["content"])
    text_ = review["messages"][0]["content"]
    assert "false_claim" in text_ and "not in the turn's record: saved notes.md" in text_
    assert agent.lessons.list()[0]["signal"] == "false_claim"


def test_no_claim_check_without_action_words_or_when_off(tmp_path):
    agent, client, small = make(tmp_path, [message(text("Paris is the capital of France."))], claim_check=True)
    turn = agent.respond("capital of France?")
    assert "claims_supported" not in turn["checks"]
    assert "You check whether a personal assistant did" not in small.seen
    agent, client, small = make(tmp_path / "off", [message(text("I've saved your notes to notes.md."))])
    turn = agent.respond("save my notes")                      # off by default: no call, no signal
    assert "claims_supported" not in turn["checks"] and "You check whether a personal assistant did" not in small.seen


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
    assert "Past conversations:\n- (2026-11-01, 8 days before today) e1" in system


def test_memory_is_shown_oldest_first_with_days_before_today():
    """Relevance order put "37 coins" above "added a quarter" and January's "5 women" above
    July's "6 women"; Qwen3.5-9B answered with the older one both times (2026-11-17)."""
    from datetime import datetime
    from harness.app import in_date_order
    now = datetime(2023, 6, 9, 11, 14)
    found = ["(2023-05-29) added a 1915-S quarter", "undated note", "(2023-05-27) 37 coins",
             "(2023-06-09) said today", "(2023-06-08) one day ago"]
    assert in_date_order(found, now) == [
        "(2023-05-27, 13 days before today) 37 coins", "(2023-05-29, 11 days before today) added a 1915-S quarter",
        "(2023-06-08, 1 day before today) one day ago", "(2023-06-09, today) said today", "undated note"]
    blocks = build_system("SOUL", now, "", [], [], GateResult(True, "q", False, "r",
                          facts=["(2023-07-20) 6 women", "(2023-01-18) 5 women"]))
    system = system_text(blocks)
    assert system.index("5 women") < system.index("6 women") and "oldest first" in system


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


def test_web_off_is_stated_and_replies_are_capped(tmp_path):
    from fake import FakeClient, message, text
    from harness.app import Settings, Sora
    gate = {"retrieve": False, "query": "", "correction": False, "complex": False, "reason": "q"}
    client = FakeClient([message(text("a")), message(text("b")), message(text("c"))], small=gate)
    agent = Sora(Settings(home=tmp_path, memory_mirror=False, web_tools=False), client, connect_mcp=False)
    agent.respond("who wrote it?")
    agent.respond("a long report please", long=True)
    first, second = client.requests[:2]
    assert "no web_search tool" in first["system"][0]["text"] and "Never write tool calls as text" in first["system"][0]["text"]
    assert (first["max_tokens"], second["max_tokens"]) == (16_000, 32_000)
    bare = Sora(Settings(home=tmp_path / "b", memory_mirror=False, web_tools=False, bare=True),
                FakeClient([message(text("x"))], small=gate), connect_mcp=False)
    bare.respond("hi")
    assert "no web_search tool" in bare.providers.anthropic_client.requests[0]["system"][0]["text"]
    with_web = Sora(Settings(home=tmp_path / "w", memory_mirror=False), FakeClient([]), connect_mcp=False)
    assert not any("web_search tool" in line for line in with_web.limits)
