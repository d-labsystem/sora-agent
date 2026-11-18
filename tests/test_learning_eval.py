"""The learning-curve experiment: quirky tools, learning on vs off."""

from dataclasses import replace

import pytest

from fake import FakeClient, message, text, tool_use
from harness.app import Settings
from ops.learning_eval import QUIRKS, RunResult, format_summary, run_one, summarize
from tools.registry import ToolError


def tools(name, log):
    q = next(q for q in QUIRKS if q.name == name)
    return {t.name: t.function for t in q.make(log)}


def test_each_quirk_has_one_learnable_rule():
    log = []
    t = tools("book_room", log)
    with pytest.raises(ToolError, match="DD/MM/YYYY"):
        t["book_room"]("Orion", "2026-12-03")
    assert t["book_room"]("Orion", "03/12/2026").startswith("Booked Orion")
    with pytest.raises(ToolError):
        t["book_room"]("Orion", "31/02/2027")                 # the right shape, not a real day
    t = tools("send_invoice", log)
    with pytest.raises(ToolError, match="USD 45.50"):
        t["send_invoice"]("Priya", "$45.50")
    assert "USD 45.50" in t["send_invoice"]("Priya", "USD 45.50")
    t = tools("find_customer", log)
    with pytest.raises(ToolError, match="lowercase"):
        t["find_customer"]("Priya Shah")
    assert "priya@shahdesign.in" in t["find_customer"]("priya shah")
    t = tools("export_report", log)
    with pytest.raises(ToolError, match="csv, tsv"):
        t["export_report"]("March 2026", "xlsx")
    assert t["export_report"]("March 2026", "CSV").endswith(".csv.")
    t = tools("delete_record", log)
    with pytest.raises(ToolError, match="confirm_token"):
        t["delete_record"]("1042")
    token = t["confirm_token"]("1042").rsplit(" ", 1)[1]
    assert t["delete_record"]("1042", token) == "Deleted record 1042."
    assert [c["broke"] for c in log] == [True, False, True, True, False, True, False, True, False, True, False]
    for q in QUIRKS:                                          # the rule isn't given away up front
        for tool in q.make([]):
            assert not any(w in tool.description for w in ("DD/MM", "USD", "lowercase", "csv", "token first"))


def test_summary_counts_repeat_errors_and_the_curve():
    def task(q, k, broke, done=True):
        return {"quirk": q, "k": k, "called": True, "first_broke": broke > 0, "broke": broke, "done": done}
    on = RunResult("on", 0, [task("a", 1, 1), task("a", 2, 0), task("a", 3, 0), task("b", 1, 2), task("b", 2, 1)])
    off = RunResult("off", 0, [task("a", 1, 1), task("a", 2, 1), task("a", 3, 1), task("b", 1, 1), task("b", 2, 1)])
    s = summarize([on, off])
    assert s["on"]["repeat_errors"] == 1 and s["off"]["repeat_errors"] == 3
    assert s["on"]["curve"] == {1: (2, 2), 2: (1, 2), 3: (0, 1)}
    assert s["reduction"] == pytest.approx(2 / 3)
    assert s["quirk_reduction"] == {"a": 1.0, "b": 0.0}
    assert "67% fewer" in format_summary(s)


def test_a_run_with_learning_on_saves_lessons_and_off_does_not(tmp_path):
    """A scripted model: first try the date as ISO (rule broken), then DD/MM/YYYY."""
    def main(request):
        last = request["messages"][-1]["content"]
        results = [b for b in last if isinstance(b, dict) and b.get("type") == "tool_result"] if isinstance(last, list) else []
        if not results:
            return message(tool_use("book_room", {"room": "Orion", "date": "2026-12-03"}), stop_reason="tool_use")
        if results[-1].get("is_error"):
            return message(tool_use("book_room", {"room": "Orion", "date": "03/12/2026"}, "call_2"), stop_reason="tool_use")
        return message(text("Booked."))

    def small(request):
        prompt = request["messages"][0]["content"]
        if "You decide whether" in prompt:
            return {"retrieve": False, "query": "", "correction": False, "reason": "r"}
        if "review a turn" in prompt:
            return {"lessons": [{"situation": "booking rooms", "lesson": "Pass dates to book_room as DD/MM/YYYY",
                                 "tool": "book_room"}]}
        return {"facts": [], "episode": ""}

    quirk = [q for q in QUIRKS if q.name == "book_room"]
    base = Settings(home=tmp_path)
    on = run_one(base, quirk, "on", 0, 2, client=FakeClient(main, small=small))
    off = run_one(base, quirk, "off", 0, 2, client=FakeClient(main, small=small))
    assert not on.error and not off.error
    assert [t["k"] for t in on.tasks] == [1, 2] and all(t["done"] and t["first_broke"] for t in on.tasks)
    assert [l["lesson"] for l in on.lessons] == ["Pass dates to book_room as DD/MM/YYYY"]
    assert off.lessons == []
