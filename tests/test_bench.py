"""Benchmark runner and FRAMES, offline: no downloads, no real model."""

import pytest

from fake import FakeClient, message, text
from harness.app import Settings, Sora
from ops.bench import runner
from ops.bench.frames import Frames
from ops.bench.grading import grade_answer
from ops.tracing import Tracer

ROWS = ("\tPrompt\tAnswer\treasoning_types\n"
        "0\tWho was first?\tJane Ballou\tMultiple constraints\n"
        "1\tHow many?\t42\tNumerical reasoning | Tabular reasoning\n"
        "2\tWhen?\t1912\tTemporal reasoning\n")


def test_wilson_interval_is_honest_about_small_samples():
    low, high = runner.wilson(30, 50)
    assert 0.45 < low < 0.47 and 0.72 < high < 0.74          # 60% of 50 is really "46-73%"
    assert runner.wilson(0, 10)[0] == 0.0 and runner.wilson(10, 10)[1] == 1.0
    assert runner.wilson(0, 0) == (0.0, 0.0)


def test_sampling_is_seeded_and_order_independent():
    items = [runner.Item(f"q{i}", "", "") for i in range(100)]
    a = runner.sample(items, 10, seed=0)
    assert [i.id for i in a] == [i.id for i in runner.sample(list(reversed(items)), 10, seed=0)]
    assert [i.id for i in a] != [i.id for i in runner.sample(items, 10, seed=1)]
    assert len(runner.sample(items, 0)) == 100


def test_the_grader_reads_the_final_answer():
    grader = FakeClient(small=[{"extracted_final_answer": "Jane Ballou", "reasoning": "same", "correct": True},
                               {"extracted_final_answer": "1913", "reasoning": "off by one", "correct": False}])
    ok = grade_answer(grader, "haiku", "Who?", "…so it's Jane Ballou.\nExact answer: Jane Ballou", "Jane Ballou")
    assert ok["checks"] == {"correct": True} and ok["judge"]["extracted"] == "Jane Ballou"
    prompt = grader.small_requests[0]["messages"][0]["content"]
    assert "[correct_answer]: Jane Ballou" in prompt and "Exact answer: Jane Ballou" in prompt
    bad = grade_answer(grader, "haiku", "When?", "Exact answer: 1913", "1912")
    assert bad["checks"] == {"correct": False} and "gold '1912'" in bad["judge"]["reason"]
    assert grade_answer(grader, "haiku", "x", "  ", "y")["checks"] == {"correct": False}   # no call made
    assert len(grader.small_requests) == 2


def test_frames_loads_from_its_local_copy(tmp_path):
    (tmp_path / "test.tsv").write_text(ROWS, encoding="utf-8")         # already downloaded
    items = Frames(None, "").load(tmp_path)
    assert [(i.id, i.answer) for i in items] == [("frames-000", "Jane Ballou"), ("frames-001", "42"),
                                                 ("frames-002", "1912")]
    assert items[1].meta["reasoning"] == "Numerical reasoning | Tabular reasoning"
    assert Frames(None, "").case(items[0])["message"].startswith("Who was first?")
    assert "Exact answer:" in Frames(None, "").case(items[0])["message"]


def test_a_run_grades_traces_breaks_down_and_respects_the_budget(tmp_path):
    (tmp_path / "test.tsv").write_text(ROWS, encoding="utf-8")
    gate = {"retrieve": False, "query": "", "correction": False, "complex": False, "reason": "q"}
    import itertools
    grades = itertools.cycle([True, False, True])
    grader = FakeClient(small=lambda req: {"extracted_final_answer": "x", "reasoning": "r", "correct": next(grades)})
    bench = Frames(grader, "haiku")
    tracer = Tracer(tmp_path / "traces.db")
    factory = lambda home, tr: Sora(Settings(home=home, memory_mirror=False, workspace=False),
                                    FakeClient([message(text("Exact answer: x"))], small=gate),
                                    tracer=tr, connect_mcp=False)
    items = bench.load(tmp_path)
    summary = runner.run(bench, items, factory, tracer, budget_usd=10.0, label="test")
    assert (summary["passed"], summary["cases"]) == (2, 3) and not summary["stopped_by_budget"]
    assert summary["groups"]["Numerical reasoning"] == (0, 1) and summary["groups"]["Temporal reasoning"] == (1, 1)
    saved = tracer.eval_run(summary["id"])
    assert saved["dataset"] == "frames (test)" and saved["cases"] == 3
    first = tracer.turn(summary["results"][0]["turn_id"])
    assert first["judge_score"] == 1.0 and first["source"] == "eval"
    assert "2/3 correct = 67%" in runner.report(summary)

    tiny = runner.run(bench, items, factory, tracer, budget_usd=1e-9)       # stops after the first spend
    assert tiny["cases"] == 1 and tiny["stopped_by_budget"]
    assert "stopped at the budget after 1 of 3" in runner.report(tiny)



def test_a_failing_grader_still_counts_what_the_question_cost(tmp_path):
    (tmp_path / "test.tsv").write_text(ROWS, encoding="utf-8")
    gate = {"retrieve": False, "query": "", "correction": False, "complex": False, "reason": "q"}
    bench = Frames(FakeClient(small=RuntimeError("grader down")), "haiku")
    tracer = Tracer(tmp_path / "traces.db")
    factory = lambda home, tr: Sora(Settings(home=home, memory_mirror=False, workspace=False),
                                    FakeClient([message(text("x"))], small=gate), tracer=tr, connect_mcp=False)
    summary = runner.run(bench, bench.load(tmp_path), factory, tracer, budget_usd=1e-9)
    assert summary["results"][0]["error"] and summary["results"][0]["cost_usd"] > 0
    assert summary["stopped_by_budget"] and summary["cases"] == 1


# ---- BrowseComp

def test_browsecomp_decrypts_with_the_published_scheme():
    from ops.bench.browsecomp import decrypt, encrypt, derive_key
    import base64, hashlib
    assert derive_key("canary", 40)[:32] == hashlib.sha256(b"canary").digest()          # SHA-256, repeated
    secret = encrypt("Which match ended 2-1 after extra time?", "BROWSECOMP:abc")
    assert base64.b64decode(secret) != b"Which match"                                     # actually scrambled
    assert decrypt(secret, "BROWSECOMP:abc") == "Which match ended 2-1 after extra time?"
    try:                                                   # a wrong key gives gibberish or invalid text
        assert decrypt(secret, "wrong key") != "Which match ended 2-1 after extra time?"
    except UnicodeDecodeError:
        pass


def test_browsecomp_loads_cases_as_long_tasks_and_reads_confidence(tmp_path):
    from ops.bench.browsecomp import BrowseComp, confidence, encrypt
    (tmp_path / "browse_comp_test_set.csv").write_text(
        "problem,answer,problem_topic,canary\n"
        f"{encrypt('Q one?', 'k1')},{encrypt('Answer One', 'k1')},Sports,k1\n", encoding="utf-8")
    [item] = BrowseComp(None, "").load(tmp_path)
    assert (item.id, item.question, item.answer, item.meta) == ("bc-0000", "Q one?", "Answer One", {"topic": "Sports"})
    case = BrowseComp(None, "").case(item)
    assert case["long"] is True and "Exact Answer:" in case["message"] and "Confidence:" in case["message"]
    assert confidence("Explanation: ...\nExact Answer: X\nConfidence: 85%") == 85
    assert confidence("**Confidence:** 40 %") == 40 and confidence("no idea") is None
    assert BrowseComp.settings == {"long_budget_usd": 0.50}


def test_calibration_report():
    from ops.bench.runner import calibrate
    results = [{"passed": True, "judge": {"confidence": 90}}, {"passed": False, "judge": {"confidence": 90}},
               {"passed": False, "judge": {"confidence": 10}}, {"passed": True, "judge": {"confidence": None}}]
    text = calibrate(results)
    assert "3 answers with a stated confidence" in text and "accuracy 33%" in text
    assert "said 70-100%: 1/2 right (said 90% on average)" in text
    assert "said 0-30%: 0/1 right" in text and "calibration error 30%" in text    # 2/3 x 40% + 1/3 x 10%
    assert calibrate([{"passed": True, "judge": {}}]) == ""


# ---- LongMemEval and dated memory

def test_facts_and_episodes_can_be_dated(tmp_path):
    from memory.store import MemoryStore
    s = MemoryStore(tmp_path / "m.db")
    s.add_fact("Aryan adopted a cat named Miso", "consolidation", "2023-05-20 02:21:00")
    sid = s.ensure_session(None, "x")
    s.add_episode("Talked about the new cat", sid, "2023-05-20 02:21:00")
    assert s.list_facts()[0]["created_at"] == "2023-05-20 02:21:00"
    assert s.search_episodes("cat") == ["(2023-05-20) Talked about the new cat"]


def test_a_turn_can_run_as_of_another_date(tmp_path):
    from datetime import datetime
    gate = {"retrieve": False, "query": "", "correction": False, "complex": False, "reason": "q"}
    client = FakeClient([message(text("ok"))], small=gate)
    agent = Sora(Settings(home=tmp_path, memory_mirror=False), client, connect_mcp=False)
    agent.respond("what day is it?", now=datetime(2023, 5, 30, 23, 40))
    assert "Current local time: Tuesday 30 May 2023, 23:40" in client.requests[0]["system"][1]["text"]


def test_longmemeval_parsing_and_exchanges():
    from ops.bench.longmemeval import exchanges, kind_of, parse_date
    assert parse_date("2023/05/30 (Tue) 23:40").isoformat() == "2023-05-30T23:40:00"
    session = [{"role": "user", "content": "hi"}, {"role": "user", "content": "also this"},
               {"role": "assistant", "content": "hello"}, {"role": "user", "content": "bye"}]
    assert exchanges(session) == [("hi\n\nalso this", "hello"), ("bye", "")]
    assert kind_of({"question_id": "q1_abs", "question_type": "multi-session"}) == "abstention"
    assert kind_of({"question_id": "q2", "question_type": "temporal-reasoning"}) == "temporal-reasoning"


def lme_row():
    return {"question_id": "q1", "question_type": "temporal-reasoning", "question": "When did I get the cat?",
            "answer": "20 May 2023", "question_date": "2023/05/30 (Tue) 23:40",
            "haystack_dates": ["2023/05/25 (Thu) 10:00", "2023/05/20 (Sat) 02:21"],
            "haystack_sessions": [[{"role": "user", "content": "Miso loves boxes"}, {"role": "assistant", "content": "Cute!"}],
                                  [{"role": "user", "content": "I adopted a cat, Miso"}, {"role": "assistant", "content": "Congrats"}]]}


def test_history_goes_through_soras_memory_with_session_dates(tmp_path):
    from ops.bench.longmemeval import ingest
    small = iter([{"facts": ["The user adopted a cat named Miso"], "episode": "User adopted a cat"},
                  {"facts": ["Miso loves boxes"], "episode": "Chatted about Miso and boxes"}])
    client = FakeClient([], small=lambda req: next(small))
    agent = Sora(Settings(home=tmp_path, memory_mirror=False), client, connect_mcp=False)
    counts = ingest(agent, lme_row())
    assert counts == {"sessions": 2, "exchanges": 2, "consolidations": 2}
    facts = {f["content"]: f["created_at"] for f in agent.store.list_facts()}
    assert facts == {"The user adopted a cat named Miso": "2023-05-20 02:21:00",      # oldest session first
                     "Miso loves boxes": "2023-05-25 10:00:00"}
    assert len(agent.store.list_sessions()) == 2


def test_longmemeval_cases_grading_and_counted_ingestion(tmp_path):
    from ops.bench.longmemeval import LongMemEval, RULES
    from ops.evals import run_case
    grader = FakeClient(small=lambda req: {"correct": "off-by-one" in req["messages"][0]["content"], "reason": "ok"})
    bench = LongMemEval(grader, "haiku")
    bench.rows = {"q1": lme_row()}
    item = runner.Item("q1", "When did I get the cat?", "20 May 2023", {"type": "temporal-reasoning"})
    case = bench.case(item)
    assert case["now"].year == 2023 and callable(case["prepare"])
    bench.closed_book = True
    assert "prepare" not in bench.case(item)
    graded = bench.grade(item, {"reply": "On 20 May 2023."})
    assert graded["checks"] == {"correct": True}                     # the temporal rule was used
    assert "Rubric:" in GRADE_for(bench, "single-session-preference") and RULES["abstention"].startswith("I will give you an unanswerable")

    gate = {"retrieve": True, "query": "cat", "correction": False, "complex": False, "reason": "q"}
    def small(req):
        content = req["messages"][0]["content"]
        return gate if "search its long-term memory" in content else {"facts": ["x"], "episode": "y"}
    tracer = Tracer(tmp_path / "traces.db")
    factory = lambda home, tr: Sora(Settings(home=home, memory_mirror=False, workspace=False),
                                    FakeClient([message(text("20 May 2023"))], small=small), tracer=tr, connect_mcp=False)
    bench.closed_book = False
    result = run_case({"id": "q1", **bench.case(item)}, factory, tracer, "run-x",
                      grade=lambda c, turn: bench.grade(item, turn))
    turns = tracer.turns(limit=10)
    assert any(t["message"].startswith("(memory: ingest 2 past sessions)") for t in turns)
    assert result["cost_usd"] > sum(t["cost_usd"] for t in turns if not t["message"].startswith("(memory"))


def GRADE_for(bench, kind):
    item = runner.Item("q", "Q?", "A", {"type": kind})
    captured = {}
    bench.grader = FakeClient(small=lambda req: captured.setdefault("p", req["messages"][0]["content"]) and
                              {"correct": True, "reason": "r"})
    bench.grade(item, {"reply": "r"})
    return captured["p"]


def test_full_context_puts_every_dated_session_in_the_prompt_and_skips_memory():
    from ops.bench.longmemeval import LongMemEval
    bench = LongMemEval(None, "haiku")
    bench.rows = {"q1": lme_row()}
    bench.full_context = True
    case = bench.case(runner.Item("q1", "When did I get the cat?", "20 May 2023", {"type": "temporal-reasoning"}))
    msg = case["message"]
    assert "prepare" not in case and case["now"].year == 2023          # no ingestion, same "as of" date
    assert msg.index("### Session 1 (2023/05/20") < msg.index("### Session 2 (2023/05/25")   # oldest first
    assert "user: I adopted a cat, Miso" in msg and "assistant: Cute!" in msg
    assert msg.rstrip().endswith("Question: When did I get the cat?")
    assert "Current date: 2023/05/30 (Tue) 23:40" in msg


def test_an_account_error_stops_the_run_and_is_not_scored(tmp_path):
    (tmp_path / "test.tsv").write_text(ROWS, encoding="utf-8")
    gate = {"retrieve": False, "query": "", "correction": False, "complex": False, "reason": "q"}
    bench = Frames(FakeClient(small={"extracted_final_answer": "x", "reasoning": "r", "correct": True}), "haiku")
    tracer = Tracer(tmp_path / "traces.db")

    def broke(request):
        raise RuntimeError("Error code: 400 - Your credit balance is too low to access the Anthropic API.")
    factory = lambda home, tr: Sora(Settings(home=home, memory_mirror=False, workspace=False),
                                    FakeClient(broke, small=gate), tracer=tr, connect_mcp=False)
    summary = runner.run(bench, bench.load(tmp_path), factory, tracer, budget_usd=10.0)
    assert summary["cases"] == 0 and "credit balance" in summary["stopped_by_account"]
    assert "STOPPED: the API refused the account after 0 of 3" in runner.report(summary)


def test_conversation_caching_can_be_turned_off_for_one_off_prompts():
    from loop.providers import AnthropicProvider
    client = FakeClient([message(text("a")), message(text("b"))])
    provider = AnthropicProvider(client)
    provider.step("m", "sys", [{"role": "user", "content": "hi"}], [], "low", lambda d: None)
    provider.cache_conversation = False
    provider.step("m", "sys", [{"role": "user", "content": "hi"}], [], "low", lambda d: None)
    assert "cache_control" in client.requests[0] and "cache_control" not in client.requests[1]
