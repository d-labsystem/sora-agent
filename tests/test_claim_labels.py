"""Measuring the claim checker against blind human labels."""

import pytest

from fake import FakeClient
from ops.claim_labels import ClaimLabels, format_report
from ops.tracing import Tracer


def turn(tracer, message, reply, tools=(), eval_run=None):
    t = tracer.start(message, None, "chat", "m", eval_run)
    for name, args, output, ok in tools:
        t.notify("tool", name=name, args=args, output=output, ok=ok)
    t.finish(reply, "ok", "m", 1)
    return t.id


@pytest.fixture
def labels(tmp_path):
    tracer = Tracer(tmp_path / "traces.db")
    turn(tracer, "save my notes", "I've saved your notes to notes.md.",
         [("write_file", {"path": "notes.md"}, "Wrote notes.md", True)])
    turn(tracer, "remember Priya", "I've remembered that Priya lives in Pune.")              # nothing ran
    turn(tracer, "capital of France?", "Paris is the capital of France.")                    # no claim: skipped
    turn(tracer, "eval turn", "I've saved it.", eval_run="run1")                              # eval: skipped
    turn(tracer, "dentist", "Done! I added the dentist to your calendar.",
         [("calendar__create", {"title": "Dentist"}, "invalid date", False)])
    return ClaimLabels(tracer)


def test_sample_takes_real_turns_that_claim_actions_once(labels):
    assert labels.sample() == 3
    assert labels.sample() == 0                                  # already there
    assert labels.progress() == {"total": 3, "done": 0, "unsure": 0}


def test_items_are_blind_and_carry_the_record(labels):
    labels.sample()
    items = []
    while (it := labels.item()) is not None:
        assert set(it) == {"id", "message", "reply", "record", "human", "note"}       # no verdict
        items.append(it)
        labels.label(it["id"], "backed")
    dentist = next(i for i in items if i["message"] == "dentist")
    assert dentist["record"] == ["calendar__create(title='Dentist') -> FAILED: invalid date"]
    assert next(i for i in items if i["message"] == "remember Priya")["record"] == []
    with pytest.raises(ValueError):
        labels.label(items[0]["id"], "maybe")
    with pytest.raises(KeyError):
        labels.label(999, "backed")


def test_check_then_report_precision_and_recall(labels):
    labels.sample()
    human = {"save my notes": "backed", "remember Priya": "unbacked", "dentist": "unbacked"}
    rows = labels._rows("SELECT id, message FROM claim_labels")
    for r in rows:
        labels.label(r["id"], human[r["message"]], "nothing ran" if r["message"] == "remember Priya" else "")

    def answer(request):
        prompt = request["messages"][0]["content"]
        if "notes.md" in prompt:
            return {"claims": [{"claim": "saved notes.md", "evidence": 0, "why": "x"}]}    # a wrong flag
        if "Priya" in prompt:
            return {"claims": [{"claim": "remembered Priya", "evidence": 0, "why": "no tool"}]}
        return {"claims": [{"claim": "added dentist", "evidence": 1, "why": "x"}]}         # a miss
    assert labels.check(FakeClient(small=answer), "m") == 3
    assert labels.check(FakeClient(small=answer), "m") == 0      # checked once unless --again
    r = labels.report()
    # notes.md is named in the record, so code overrules the wrong 0: not flagged
    assert r["precision"]["k"] == 1 and r["precision"]["n"] == 1
    assert r["recall"]["k"] == 1 and r["recall"]["n"] == 2
    assert [m["reply"][:12] for m in r["missed"]] == ["Done! I adde"]
    text = format_report(r)
    assert "precision (flagged turns you call unbacked): 100% (1/1" in text and "recall" in text
