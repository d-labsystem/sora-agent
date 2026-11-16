"""`uv run main.py bench <name> --n 50 --budget 12`: run a public benchmark through sora.

See ops/bench/runner.py for how runs work, and each module for its benchmark.
"""

from __future__ import annotations

import sys

BENCHMARKS = {"frames": "ops.bench.frames:Frames", "gaia": "ops.bench.gaia:Gaia"}


def build(name: str, grader, grader_model: str):
    module, cls = BENCHMARKS[name].split(":")
    return getattr(__import__(module, fromlist=[cls]), cls)(grader, grader_model)


# The contamination check: the model alone, no tools at all. If it scores about as
# well as with tools, the benchmark's answers are in its training data and the
# benchmark is measuring memory (GAIA validation, 2026-11-16: 35 of 37 tool-free
# answers were right, and the model said "from memory" on audio it couldn't hear).
CLOSED_BOOK = {"bare": True, "web_tools": False, "workspace": False, "code_execution": False}


def main(name: str, n: int, budget: float, seed: int = 0, label: str = "", out=sys.stdout,
         bare: bool = False, closed_book: bool = False) -> int:
    from harness.app import eval_setup, load_settings
    from ops.bench import runner
    overrides = CLOSED_BOOK if closed_book else {"bare": True} if bare else {}
    tracer, grader, model, grader_model, factory = eval_setup(**overrides)
    bench = build(name, grader, grader_model)
    folder = load_settings().home / "benchmarks" / name
    items = runner.sample(bench.load(folder), n, seed, getattr(bench, "strata_key", ""))
    mode = "closed book" if closed_book else "bare model" if bare else ""
    label = ", ".join(x for x in (label, mode) if x)
    out.write(f"{name}: {len(items)} questions (seed {seed}), budget ${budget:.2f}, model {model}"
              f"{f' ({mode})' if mode else ''}\n")
    out.flush()

    def progress(r: dict) -> None:
        mark = "PASS" if r["passed"] else ("ERR " if r.get("error") else "FAIL")
        said = (r.get("judge") or {}).get("extracted") or r.get("error") or ""
        out.write(f"{mark}  {r['id']}  ${r['cost_usd']:.3f}  {str(said)[:70]}\n")
        out.flush()

    summary = runner.run(bench, items, factory, tracer, budget, label, model, progress)
    out.write("\n" + runner.report(summary) + "\n")
    return 0
