"""Is the judge right? You label a sample of its verdicts blind; this measures how often it agrees.

Every benchmark accuracy here is "what Haiku said was correct". If the judge is
too strict on one question type, or too lenient on one model's style, every
comparison built on it inherits the error. So:

1. `sample`: draw verdicts from finished runs, stratified by question type and
   verdict (PASS/FAIL), so rare types and both kinds of verdict are covered.
2. You label each one in the dashboard (Evals → Judge check) without seeing the
   judge's verdict: correct, wrong, or unsure.
3. `report`: agreement, Cohen's kappa, where the judge is too strict or too
   lenient, and each run's accuracy corrected for the judge's measured errors.

The correction uses the two numbers a verdict-stratified sample estimates well:
    precision  P(really correct | judge said PASS)
    miss rate  P(really correct | judge said FAIL)
    corrected  = pass_rate * precision + (1 - pass_rate) * miss rate
Kappa and agreement describe the sample. Because rare strata are oversampled,
they're not the population figures; the corrected accuracies are.

Labels live in traces.db (table judge_labels), next to the runs they describe.
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Callable

from ops.bench.runner import wilson
from ops.tracing import Tracer, now_iso

TABLE = """
CREATE TABLE IF NOT EXISTS judge_labels (
    id INTEGER PRIMARY KEY,
    run_id TEXT NOT NULL,
    case_id TEXT NOT NULL,
    turn_id TEXT,
    kind TEXT,                                    -- question type, e.g. multi-session
    question TEXT,
    reference TEXT,                               -- the dataset's answer, rubric or explanation
    reference_label TEXT,
    reply TEXT,
    judge_passed INTEGER NOT NULL,
    judge_reason TEXT,
    human TEXT,                                   -- correct | wrong | unsure; NULL until labelled
    note TEXT,
    position INTEGER NOT NULL,                    -- labelling order (shuffled)
    labeled_at TEXT,
    UNIQUE (run_id, case_id));
"""
HUMAN = ("correct", "wrong", "unsure")
BLIND = ("id", "kind", "question", "reference", "reference_label", "reply", "human", "note")

# Given a run's dataset name, return {case_id: (question, reference, reference_label)}.
References = Callable[[str], dict[str, tuple[str, str, str]]]


def benchmark_references(home: Path) -> References:
    """References from the downloaded benchmark files, loaded once per benchmark."""
    from ops.bench import BENCHMARKS, benchmark_class
    cache: dict[str, dict] = {}

    def lookup(dataset: str) -> dict:
        name = dataset.split(" ", 1)[0]
        if name not in BENCHMARKS:
            raise ValueError(f"{dataset!r} isn't a benchmark run; only benchmark runs carry references")
        if name not in cache:
            bench = benchmark_class(name)(None, "")
            label = getattr(bench, "reference_label", lambda item: "Correct answer")
            cache[name] = {i.id: (i.question, i.answer, label(i)) for i in bench.load(home / "benchmarks" / name)}
        return cache[name]
    return lookup


class JudgeCheck:
    def __init__(self, tracer: Tracer):
        self.tracer = tracer
        with tracer.lock:
            tracer.conn.executescript(TABLE)

    def _rows(self, sql: str, params: tuple = ()) -> list[dict]:
        with self.tracer.lock:
            return [dict(r) for r in self.tracer.conn.execute(sql, params)]

    # ---- building the sample

    def candidates(self, run_ids: list[str]) -> list[dict]:
        """Every graded case in these runs. Cases that crashed have no verdict to check."""
        found = []
        for run_id in run_ids:
            run = self.tracer.eval_run(run_id)
            if run is None:
                raise KeyError(f"no eval run {run_id!r}")
            for r in run["results"]:
                if r.get("error") or not r.get("judge"):
                    continue
                found.append({"run_id": run_id, "dataset": run["dataset"], "case_id": r["id"],
                              "turn_id": r.get("turn_id"), "kind": (r.get("meta") or {}).get("type", ""),
                              "judge_passed": bool(r["passed"]), "judge_reason": r["judge"].get("reason", ""),
                              "reply": r.get("reply") or ""})
        return found

    @staticmethod
    def stratified(cands: list[dict], n: int, seed: int = 0) -> list[dict]:
        """Round-robin over (type, verdict) strata, so a type with 6 questions gets as
        many picks as one with 200 until it runs out. Seeded, so it's repeatable."""
        rng = random.Random(seed)
        strata = defaultdict(list)
        for c in sorted(cands, key=lambda c: (c["run_id"], c["case_id"])):
            strata[(c["kind"], c["judge_passed"])].append(c)
        queues = [strata[k] for k in sorted(strata, key=str)]
        for q in queues:
            rng.shuffle(q)
        picked = []
        while len(picked) < n and any(queues):
            for q in queues:
                if q and len(picked) < n:
                    picked.append(q.pop())
        return picked

    def sample(self, run_ids: list[str], n: int, references: References, seed: int = 0) -> int:
        """Add n verdicts to the label set (cases already in it are skipped). Returns how many were added."""
        have = {(r["run_id"], r["case_id"]) for r in self._rows("SELECT run_id, case_id FROM judge_labels")}
        pool = [c for c in self.candidates(run_ids) if (c["run_id"], c["case_id"]) not in have]
        picked = self.stratified(pool, n, seed)
        random.Random(seed + 1).shuffle(picked)        # labelling order mixes runs and types
        start = (self._rows("SELECT COALESCE(MAX(position), 0) AS p FROM judge_labels")[0]["p"])
        rows = []
        for i, c in enumerate(picked, start + 1):
            question, reference, label = references(c["dataset"]).get(c["case_id"], ("", "", "Correct answer"))
            rows.append((c["run_id"], c["case_id"], c["turn_id"], c["kind"], question, reference, label,
                         c["reply"], int(c["judge_passed"]), c["judge_reason"], i))
        with self.tracer.lock:
            self.tracer.conn.executemany(
                "INSERT INTO judge_labels (run_id, case_id, turn_id, kind, question, reference, reference_label, "
                "reply, judge_passed, judge_reason, position) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
            self.tracer.conn.commit()
        return len(rows)

    # ---- labelling (what the dashboard calls; never shows the judge's verdict)

    def progress(self) -> dict:
        row = self._rows("SELECT COUNT(*) AS total, COUNT(human) AS done, "
                         "SUM(human = 'unsure') AS unsure FROM judge_labels")[0]
        return {"total": row["total"], "done": row["done"], "unsure": row["unsure"] or 0}

    def item(self, label_id: int | None = None) -> dict | None:
        """One item, blind: the given one, or else the next unlabelled one."""
        cols = ", ".join(BLIND)
        rows = (self._rows(f"SELECT {cols} FROM judge_labels WHERE id = ?", (label_id,)) if label_id is not None
                else self._rows(f"SELECT {cols} FROM judge_labels WHERE human IS NULL ORDER BY position LIMIT 1"))
        return rows[0] if rows else None

    def label(self, label_id: int, human: str, note: str = "") -> dict:
        if human not in HUMAN:
            raise ValueError(f"label must be one of {', '.join(HUMAN)}")
        if not self._rows("SELECT id FROM judge_labels WHERE id = ?", (label_id,)):
            raise KeyError(f"no label item {label_id}")
        self.tracer._write("UPDATE judge_labels SET human = ?, note = ?, labeled_at = ? WHERE id = ?",
                           (human, note[:2000], now_iso(), label_id))
        return self.progress()

    # ---- the report

    def report(self) -> dict:
        rows = self._rows("SELECT l.*, r.model, r.dataset, r.cases, r.passed FROM judge_labels l "
                          "LEFT JOIN eval_runs r ON r.id = l.run_id WHERE l.human IN ('correct', 'wrong')")
        return summarize(rows, self.progress())


def kappa(pairs: list[tuple[bool, bool]]) -> float | None:
    """Cohen's kappa for (judge, human) yes/no pairs: agreement beyond what chance gives.
    1 = perfect, 0 = no better than chance. None when it's undefined (everyone says the same)."""
    n = len(pairs)
    if not n:
        return None
    observed = sum(j == h for j, h in pairs) / n
    pj = sum(j for j, _ in pairs) / n
    ph = sum(h for _, h in pairs) / n
    chance = pj * ph + (1 - pj) * (1 - ph)
    return None if chance == 1 else (observed - chance) / (1 - chance)


def _rate(hits: int, total: int) -> dict:
    low, high = wilson(hits, total)
    return {"hits": hits, "n": total, "rate": hits / total if total else None, "low": low, "high": high}


def summarize(rows: list[dict], progress: dict) -> dict:
    pairs = [(bool(r["judge_passed"]), r["human"] == "correct") for r in rows]
    passed = [h for j, h in pairs if j]
    failed = [h for j, h in pairs if not j]
    precision = _rate(sum(passed), len(passed))             # P(correct | judge PASS)
    miss = _rate(sum(failed), len(failed))                  # P(correct | judge FAIL)

    def group(key: str) -> list[dict]:
        out = defaultdict(list)
        for r, (j, h) in zip(rows, pairs):
            out[r[key] or "?"].append((j, h))
        return [{"name": k, "n": len(v), "agree": sum(j == h for j, h in v),
                 "too_strict": sum(h and not j for j, h in v), "too_lenient": sum(j and not h for j, h in v)}
                for k, v in sorted(out.items())]

    runs = {}
    if precision["n"] and miss["n"]:
        for r in rows:
            if r["run_id"] in runs or not r["cases"]:
                continue
            a = r["passed"] / r["cases"]
            runs[r["run_id"]] = {
                "run_id": r["run_id"], "dataset": r["dataset"], "model": r["model"], "judge_accuracy": a,
                "corrected": a * precision["rate"] + (1 - a) * miss["rate"],
                "low": a * precision["low"] + (1 - a) * miss["low"],
                "high": a * precision["high"] + (1 - a) * miss["high"],
                "labelled": sum(x["run_id"] == r["run_id"] for x in rows)}
    disagreements = [{"id": r["id"], "run_id": r["run_id"], "case_id": r["case_id"], "turn_id": r["turn_id"],
                      "kind": r["kind"], "judge": "PASS" if j else "FAIL", "human": r["human"],
                      "judge_reason": r["judge_reason"], "note": r["note"] or ""}
                     for r, (j, h) in zip(rows, pairs) if j != h]
    return {"progress": progress, "n": len(pairs),
            "agreement": sum(j == h for j, h in pairs) / len(pairs) if pairs else None,
            "kappa": kappa(pairs), "precision": precision, "miss_rate": miss,
            "too_strict": sum(h and not j for j, h in pairs), "too_lenient": sum(j and not h for j, h in pairs),
            "by_type": group("kind"), "by_model": group("model"),
            "runs": sorted(runs.values(), key=lambda x: x["run_id"]), "disagreements": disagreements}


def pct(v: float | None) -> str:
    return "–" if v is None else f"{v * 100:.0f}%"


def _rate_line(what: str, x: dict, verdict: str) -> str:
    if not x["n"]:
        return f"{what}: no {verdict} verdicts labelled yet"
    return f"{what} {pct(x['rate'])} of the time ({x['hits']}/{x['n']}, 95% {pct(x['low'])}–{pct(x['high'])})"


def format_report(r: dict) -> str:
    p = r["progress"]
    lines = [f"judge check: {p['done']}/{p['total']} labelled ({p['unsure']} unsure, left out)"]
    if not r["n"]:
        return lines[0] + "\nNothing labelled yet: open the dashboard, Evals → Judge check."
    k = r["kappa"]
    lines += [f"agreement {pct(r['agreement'])} on {r['n']} verdicts, kappa "
              + ("–" if k is None else f"{k:.2f}"),
              f"judge too strict (FAIL, you said correct): {r['too_strict']}   "
              f"too lenient (PASS, you said wrong): {r['too_lenient']}",
              _rate_line("when the judge says PASS it's right", r["precision"], "PASS"),
              _rate_line("when the judge says FAIL the reply was still correct", r["miss_rate"], "FAIL"),
              "", "by question type:                 n  agree  strict  lenient"]
    for g in r["by_type"]:
        lines.append(f"  {g['name'][:30]:30} {g['n']:3}  {g['agree']:5}  {g['too_strict']:6}  {g['too_lenient']:7}")
    lines += ["", "by model:"]
    for g in r["by_model"]:
        lines.append(f"  {g['name'][:30]:30} {g['n']:3}  {g['agree']:5}  {g['too_strict']:6}  {g['too_lenient']:7}")
    if r["runs"]:
        lines += ["", "accuracy, judge's vs corrected (rough 95% range):"]
        for x in r["runs"]:
            lines.append(f"  {pct(x['judge_accuracy']):>4} → {pct(x['corrected']):>4} "
                         f"({pct(x['low'])}–{pct(x['high'])})  {x['dataset']}")
    lines += ["", "agreement and kappa describe the sample (rare types oversampled); corrected accuracies "
              "weight by each run's own pass rate."]
    return "\n".join(lines)
