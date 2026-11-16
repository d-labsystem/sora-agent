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
    group_by: str                              # meta key to break results down by ("" for none)

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


def sample(items: list[Item], n: int, seed: int = 0) -> list[Item]:
    """The same n items for the same seed, whatever order they were loaded in."""
    ordered = sorted(items, key=lambda i: i.id)
    if n <= 0 or n >= len(ordered):
        return ordered
    return random.Random(seed).sample(ordered, n)


def wilson(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval for a proportion; sensible even at 0 or 100%."""
    if total == 0:
        return 0.0, 0.0
    p = correct / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return max(0.0, centre - half), min(1.0, centre + half)


def run(bench: Benchmark, items: list[Item], factory, tracer, budget_usd: float, label: str = "",
        model: str = "", progress: Callable[[dict], None] | None = None) -> dict:
    """Run `items` until done or the budget is spent. Saved as an eval run."""
    run_id = "bench-" + uuid.uuid4().hex[:8]
    started, results, spent, stopped = now_iso(), [], 0.0, False
    by_id = {i.id: i for i in items}
    for item in items:
        if budget_usd and spent >= budget_usd:
            stopped = True
            break
        case = {"id": item.id, **bench.case(item)}
        result = run_case(case, factory, tracer, run_id,
                          grade=lambda c, turn: bench.grade(by_id[c["id"]], turn))
        result["meta"] = item.meta
        results.append(result)
        spent += result["cost_usd"]
        if progress:
            progress(result)
    correct = sum(r["passed"] for r in results)
    low, high = wilson(correct, len(results))
    groups: dict[str, list[bool]] = defaultdict(list)
    if bench.group_by:
        for r in results:
            for key in str(r["meta"].get(bench.group_by) or "other").split(" | "):
                groups[key.strip()].append(r["passed"])
    run = {"id": run_id, "dataset": f"{bench.name}{f' ({label})' if label else ''}", "model": model,
           "cases": len(results), "passed": correct, "cost_usd": round(spent, 6), "results": results,
           "started_at": started, "ended_at": now_iso()}
    tracer.save_eval_run(run)
    return {**run, "accuracy": correct / len(results) if results else 0.0, "interval": (low, high),
            "planned": len(items), "stopped_by_budget": stopped,
            "groups": {k: (sum(v), len(v)) for k, v in sorted(groups.items())}}


def report(summary: dict) -> str:
    low, high = summary["interval"]
    n = summary["cases"]
    lines = [f"{summary['dataset']}: {summary['passed']}/{n} correct = {summary['accuracy']:.0%} "
             f"(95% interval {low:.0%}–{high:.0%}), cost ${summary['cost_usd']:.2f} "
             f"(${summary['cost_usd'] / max(n, 1):.3f}/question), run {summary['id']}"]
    if summary["stopped_by_budget"]:
        lines.append(f"stopped at the budget after {n} of {summary['planned']} questions")
    for group, (ok, total) in summary["groups"].items():
        lines.append(f"  {group:32} {ok}/{total}")
    return "\n".join(lines)
