"""Judge check: sampling, blind labelling, and the agreement / corrected-accuracy maths. Offline."""

import io
import json
import threading
import urllib.request

import pytest

from fake import FakeClient
from harness.app import Settings, Sora
from harness.cli import main
from harness.web import make_server
from ops.judge_check import JudgeCheck, format_report, kappa, summarize
from ops.tracing import Tracer, now_iso

KINDS = ["multi-session"] * 6 + ["temporal"] * 6 + ["preference"] * 2


def save_run(tracer, run_id, passes, model="m", dataset="longmemeval (test)"):
    results = [{"id": f"q{i}", "turn_id": f"t-{run_id}-{i}", "passed": ok, "checks": {"correct": ok},
                "judge": {"score": float(ok), "grade": 5 if ok else 1, "reason": f"reason {i}"},
                "reply": f"reply {i}", "cost_usd": 0.0, "meta": {"type": KINDS[i % len(KINDS)]}}
               for i, ok in enumerate(passes)]
    results.append({"id": "crashed", "turn_id": None, "passed": False, "checks": {}, "judge": None,
                    "error": "timeout", "reply": "", "cost_usd": 0.0, "meta": {"type": "temporal"}})
    tracer.save_eval_run({"id": run_id, "dataset": dataset, "model": model, "cases": len(results),
                          "passed": sum(passes), "cost_usd": 0.0, "results": results,
                          "started_at": now_iso(), "ended_at": now_iso()})


def refs(dataset):
    return {f"q{i}": (f"question {i}", f"answer {i}", "Rubric" if i == 13 else "Correct answer") for i in range(40)}


@pytest.fixture
def check(tmp_path):
    tracer = Tracer(tmp_path / "traces.db")
    save_run(tracer, "run-a", [i % 3 != 0 for i in range(28)], model="opus")
    save_run(tracer, "run-b", [i % 2 == 0 for i in range(28)], model="qwen")
    return JudgeCheck(tracer)


def test_kappa_is_agreement_beyond_chance():
    assert kappa([(True, True), (False, False)] * 5) == 1.0
    # judge says PASS to everything, humans split 50/50: 50% agreement, but no better than chance
    assert kappa([(True, True), (True, False)] * 5) == 0.0
    assert kappa([(True, True)] * 4) is None               # undefined when nobody ever disagrees in kind
    assert kappa([]) is None
    # textbook: 20 yes/yes, 5 yes/no, 10 no/yes, 15 no/no -> po .7, pe .5, kappa .4
    pairs = [(True, True)] * 20 + [(True, False)] * 5 + [(False, True)] * 10 + [(False, False)] * 15
    assert kappa(pairs) == pytest.approx(0.4)


def test_sample_covers_every_type_and_verdict_and_skips_crashes(check):
    added = check.sample(["run-a", "run-b"], 20, refs, seed=0)
    assert added == 20
    rows = check._rows("SELECT * FROM judge_labels")
    strata = {(r["kind"], r["judge_passed"]) for r in rows}
    assert strata == {(k, v) for k in ("multi-session", "temporal", "preference") for v in (0, 1)}
    assert "crashed" not in {r["case_id"] for r in rows}
    assert {r["run_id"] for r in rows} == {"run-a", "run-b"}
    # sampling again adds new cases only, and the same seed picks the same ones
    assert check.sample(["run-a", "run-b"], 5, refs, seed=0) == 5
    assert len({(r["run_id"], r["case_id"]) for r in check._rows("SELECT * FROM judge_labels")}) == 25


def test_sample_is_repeatable(tmp_path, check):
    other = Tracer(tmp_path / "other.db")
    save_run(other, "run-a", [i % 3 != 0 for i in range(28)], model="opus")
    save_run(other, "run-b", [i % 2 == 0 for i in range(28)], model="qwen")
    again = JudgeCheck(other)
    check.sample(["run-a", "run-b"], 12, refs, seed=3)
    again.sample(["run-a", "run-b"], 12, refs, seed=3)
    q = "SELECT run_id, case_id FROM judge_labels ORDER BY position"
    assert check._rows(q) == again._rows(q)


def test_unknown_run_is_an_error(check):
    with pytest.raises(KeyError):
        check.sample(["nope"], 5, refs)


def test_items_are_blind_and_labelling_moves_on(check):
    check.sample(["run-a"], 4, refs)
    first = check.item()
    assert "judge_passed" not in first and "judge_reason" not in first
    assert first["question"].startswith("question") and first["reference"].startswith("answer")
    assert check.label(first["id"], "correct", "fine")["done"] == 1
    assert check.item()["id"] != first["id"]
    assert check.item(first["id"])["human"] == "correct"     # going back shows what you said
    with pytest.raises(ValueError):
        check.label(first["id"], "maybe")
    with pytest.raises(KeyError):
        check.label(999, "wrong")


def row(run, judge, human, kind="k", model="m", cases=10, passed=5):
    return {"id": 1, "run_id": run, "case_id": "q", "turn_id": "t", "kind": kind, "model": model,
            "dataset": run, "cases": cases, "passed": passed, "judge_passed": int(judge),
            "human": "correct" if human else "wrong", "judge_reason": "", "note": ""}


def test_report_corrects_accuracy_from_the_judges_measured_errors():
    # judge PASS: 9 of 10 really correct (one too lenient); judge FAIL: 2 of 10 really correct (too strict)
    rows = ([row("r1", True, True, "temporal")] * 9 + [row("r1", True, False, "temporal")]
            + [row("r1", False, True, "preference")] * 2 + [row("r1", False, False, "preference")] * 8)
    r = summarize(rows, {"done": 20, "total": 20, "unsure": 0})
    assert r["agreement"] == pytest.approx(17 / 20)
    assert r["too_lenient"] == 1 and r["too_strict"] == 2
    assert r["precision"]["rate"] == pytest.approx(0.9) and r["miss_rate"]["rate"] == pytest.approx(0.2)
    run = r["runs"][0]                       # the run's judge said 50% -> .5*.9 + .5*.2 = 55%
    assert run["judge_accuracy"] == 0.5 and run["corrected"] == pytest.approx(0.55)
    assert run["low"] < run["corrected"] < run["high"]
    by_type = {g["name"]: g for g in r["by_type"]}
    assert by_type["preference"]["too_strict"] == 2 and by_type["temporal"]["too_lenient"] == 1
    assert len(r["disagreements"]) == 3
    text = format_report(r)
    assert "agreement 85%" in text and "50% →  55%" in text


def test_report_with_nothing_labelled():
    r = summarize([], {"done": 0, "total": 5, "unsure": 0})
    assert r["agreement"] is None and r["runs"] == []
    assert "Nothing labelled yet" in format_report(r)


def test_unsure_labels_are_left_out(check):
    check.sample(["run-a"], 3, refs)
    ids = [r["id"] for r in check._rows("SELECT id FROM judge_labels")]
    check.label(ids[0], "unsure")
    check.label(ids[1], "correct")
    r = check.report()
    assert r["n"] == 1 and r["progress"]["unsure"] == 1


def test_cli_sample_status_report(tmp_path, monkeypatch):
    monkeypatch.setenv("SORA_HOME", str(tmp_path))
    save_run(Tracer(tmp_path / "traces.db"), "run-a", [True, False] * 10)
    monkeypatch.setattr("ops.judge_check.benchmark_references", lambda home: refs)
    out = io.StringIO()
    assert main(["--home", str(tmp_path), "judge", "sample", "--runs", "run-a", "--n", "6"], out=out) == 0
    assert "added 6 verdicts" in out.getvalue()
    out = io.StringIO()
    assert main(["--home", str(tmp_path), "judge", "status"], out=out) == 0
    assert out.getvalue().startswith("0/6 labelled")
    assert main(["--home", str(tmp_path), "judge", "report"], out=io.StringIO()) == 0
    out = io.StringIO()
    assert main(["--home", str(tmp_path), "judge", "sample"], out=out) == 1 and "--runs" in out.getvalue()
    out = io.StringIO()
    assert main(["--home", str(tmp_path), "judge", "sample", "--runs", "nope"], out=out) == 1
    assert "no eval run 'nope'" in out.getvalue()


def test_dashboard_serves_items_blind_and_saves_labels(tmp_path):
    agent = Sora(Settings(home=tmp_path), FakeClient(lambda r: None), connect_mcp=False)
    save_run(agent.tracer, "run-a", [True, False] * 5)
    JudgeCheck(agent.tracer).sample(["run-a"], 4, refs)
    srv = make_server(agent, 0, watch_interval=0.1)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"

    def get(path):
        with urllib.request.urlopen(base + path, timeout=10) as res:
            return json.loads(res.read())

    def post(body):
        req = urllib.request.Request(base + "/api/judge-check", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=10) as res:
                return res.status, json.loads(res.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())
    try:
        data = get("/api/judge-check")
        assert data["progress"] == {"total": 4, "done": 0, "unsure": 0}
        item = data["item"]
        assert set(item) == {"id", "kind", "question", "reference", "reference_label", "reply", "human", "note"}
        assert post({"id": item["id"], "human": "wrong", "note": "off by one"}) == (200, {"total": 4, "done": 1, "unsure": 0})
        assert get(f"/api/judge-check?id={item['id']}")["item"]["note"] == "off by one"
        assert post({"id": item["id"], "human": "maybe"})[0] == 400
        assert post({"id": 999, "human": "wrong"})[0] == 404
        report = get("/api/judge-check/report")
        assert report["n"] == 1 and "judge check: 1/4 labelled" in report["text"]
    finally:
        srv.dashboard.close()
        srv.shutdown()
        srv.server_close()
