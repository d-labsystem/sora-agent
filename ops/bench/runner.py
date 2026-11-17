"""Benchmarks: public test sets, run through sora exactly as a user would.

Each question becomes an ordinary eval case (ops/evals.run_case): a fresh
home, the real respond() path, a full trace. So a benchmark run shows up in
the dashboard's Evals tab, in Phoenix and in Langfuse like any other run.
A benchmark supplies three things:

    load(folder)        its questions, downloaded once into ~/.sora/benchmarks/<name>/
    case(item)          the message (and any setup) sora receives
    grade(item, turn)   correct or not, by the benchmark's own rules

The runner adds what makes the numbers trustworthy and affordable:

    sampling    a seeded random subset: the same questions every run, so a
                before/after comparison measures the change, not the luck
    a budget    a hard dollar cap; the run stops cleanly when it's reached and
                reports how many questions it finished
    intervals   accuracy with a 95% Wilson interval. With 50 questions, 60%
                means "somewhere between 46% and 72%": differences smaller
                than that between two runs are noise

These are not leaderboard numbers: published scores use other agents, all
questions, sometimes several attempts. They measure sora as built, and above
all whether a change made it better or worse.
"""

from __future__ import annotations

import math
import random
import urllib.request
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from ops.evals import run_case
from ops.tracing import now_iso


@dataclass
class Item:
    id: str
    question: str
    answer: str
    meta: dict = field(default_factory=dict)   # e.g. reasoning type, level: used for breakdowns


class Benchmark(Protocol):
    name: str
    group_by: str | tuple                      # meta key(s) to break results down by ("" for none)

    def load(self, folder: Path) -> list[Item]: ...

    def case(self, item: Item) -> dict: ...

    def grade(self, item: Item, turn: dict) -> dict: ...   # {"checks": {...}, "judge": {...}}


def download(url: str, path: Path) -> Path:
    """Fetch once; later runs read the local copy."""
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as f:
            while chunk := response.read(1 << 20):
                f.write(chunk)
        tmp.replace(path)
    return path


def sample(items: list[Item], n: int, seed: int = 0, strata: str = "") -> list[Item]:
    """The same n items for the same seed, whatever order they were loaded in.
    With `strata` (a meta key, e.g. GAIA's level), each stratum gets its share
    of n in proportion to its size (largest remainder), so a small, hard level
    isn't left with two questions by chance."""
    ordered = sorted(items, key=lambda i: i.id)
    if n <= 0 or n >= len(ordered):
        return ordered
    rng = random.Random(seed)
    if not strata:
        return rng.sample(ordered, n)
    groups: dict[str, list[Item]] = defaultdict(list)
    for item in ordered:
        groups[str(item.meta.get(strata))].append(item)
    exact = {k: n * len(v) / len(ordered) for k, v in groups.items()}
    quota = {k: int(x) for k, x in exact.items()}
    for k in sorted(exact, key=lambda k: exact[k] - quota[k], reverse=True)[:n - sum(quota.values())]:
        quota[k] += 1
    return [item for k in sorted(groups) for item in rng.sample(groups[k], quota[k])]


def wilson(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval for a proportion; sensible even at 0 or 100%."""
    if total == 0:
        return 0.0, 0.0
    p = correct / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return max(0.0, centre - half), min(1.0, centre + half)


# Errors that no other question will get past: the account, not the question. Seen
# 2026-11-17: the credit ran out mid-run and 155 questions were logged as "wrong".
ACCOUNT_ERRORS = ("credit balance is too low", "authentication_error", "invalid x-api-key",
                  "permission_error", "billing", "no valid api key")


def account_error(result: dict) -> str:
    error = str(result.get("error") or "")
    return error if any(e in error.lower() for e in ACCOUNT_ERRORS) else ""


def run(bench: Benchmark, items: list[Item], factory, tracer, budget_usd: float, label: str = "",
        model: str = "", progress: Callable[[dict], None] | None = None, parallel: int = 1) -> dict:
    """Run `items` until done or the budget is spent. Saved as an eval run.
    `parallel` runs that many questions at once (each in its own home); the
    budget is then checked between batches, so it can be passed by up to one batch."""
    run_id = "bench-" + uuid.uuid4().hex[:8]
    started, results, spent, stopped, blocked = now_iso(), [], 0.0, False, ""
    by_id = {i.id: i for i in items}

    def one(item: Item) -> dict:
        case = {"id": item.id, **bench.case(item)}
        result = run_case(case, factory, tracer, run_id,
                          grade=lambda c, turn: bench.grade(by_id[c["id"]], turn))
        result["meta"] = item.meta
        return result

    batch = max(1, parallel)
    with ThreadPoolExecutor(max_workers=batch) as pool:
        for start in range(0, len(items), batch):
            if budget_usd and spent >= budget_usd:
                stopped = True
                break
            if blocked:
                break
            for result in pool.map(one, items[start:start + batch]):
                spent += result["cost_usd"]
                if account_error(result):        # not an answer: leave it out of the score
                    blocked = blocked or account_error(result)
                    continue
                results.append(result)
                if progress:
                    progress(result)
    correct = sum(r["passed"] for r in results)
    low, high = wilson(correct, len(results))
    groups: dict[str, list[bool]] = defaultdict(list)
    keys = (bench.group_by,) if isinstance(bench.group_by, str) else bench.group_by
    for key in filter(None, keys):
        for r in results:
            for value in str(r["meta"].get(key) or "other").split(" | "):
                groups[f"{key}: {value.strip()}" if len(keys) > 1 else value.strip()].append(r["passed"])
    run = {"id": run_id, "dataset": f"{bench.name}{f' ({label})' if label else ''}", "model": model,
           "cases": len(results), "passed": correct, "cost_usd": round(spent, 6), "results": results,
           "started_at": started, "ended_at": now_iso()}
    tracer.save_eval_run(run)
    return {**run, "accuracy": correct / len(results) if results else 0.0, "interval": (low, high),
            "planned": len(items), "stopped_by_budget": stopped, "stopped_by_account": blocked,
            "flagged": [r["id"] for r in results if (r.get("judge") or {}).get("contaminated")],
            "groups": {k: (sum(v), len(v)) for k, v in sorted(groups.items())}}


def report(summary: dict) -> str:
    low, high = summary["interval"]
    n = summary["cases"]
    lines = [f"{summary['dataset']}: {summary['passed']}/{n} correct = {summary['accuracy']:.0%} "
             f"(95% interval {low:.0%}–{high:.0%}), cost ${summary['cost_usd']:.2f} "
             f"(${summary['cost_usd'] / max(n, 1):.3f}/question), run {summary['id']}"]
    if summary.get("flagged"):
        lines.append(f"searched for the benchmark itself (possible contamination): {', '.join(summary['flagged'])}")
    if summary["stopped_by_budget"]:
        lines.append(f"stopped at the budget after {n} of {summary['planned']} questions")
    if summary.get("stopped_by_account"):
        lines.append(f"STOPPED: the API refused the account after {n} of {summary['planned']} questions "
                     f"(not scored): {summary['stopped_by_account'][:200]}")
    for group, (ok, total) in summary["groups"].items():
        lines.append(f"  {group:32} {ok}/{total}")
    calibration = calibrate(summary["results"])
    if calibration:
        lines.append(calibration)
    return "\n".join(lines)


def calibrate(results: list[dict]) -> str:
    """Does stated confidence match reality? Mean confidence vs accuracy, and
    accuracy within confidence bands. Calibration error is the gap between
    confidence and accuracy, averaged over answers (0 = perfectly calibrated)."""
    pairs = [((r.get("judge") or {}).get("confidence"), r["passed"]) for r in results]
    pairs = [(c, ok) for c, ok in pairs if c is not None]
    if not pairs:
        return ""
    bands = [(0, 30), (30, 70), (70, 101)]
    lines = [f"calibration ({len(pairs)} answers with a stated confidence): mean confidence "
             f"{sum(c for c, _ in pairs) / len(pairs):.0f}%, accuracy {sum(ok for _, ok in pairs) / len(pairs):.0%}"]
    error = 0.0
    for lo, hi in bands:
        group = [(c, ok) for c, ok in pairs if lo <= c < hi]
        if group:
            conf, acc = sum(c for c, _ in group) / len(group) / 100, sum(ok for _, ok in group) / len(group)
            error += len(group) / len(pairs) * abs(conf - acc)
            lines.append(f"  said {lo}-{min(hi, 100)}%: {sum(ok for _, ok in group)}/{len(group)} right "
                         f"(said {conf:.0%} on average)")
    lines.append(f"  calibration error {error:.0%}")
    return "\n".join(lines)
