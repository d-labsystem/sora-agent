"""`uv run main.py bench <name> --n 50 --budget 12`: run a public benchmark through sora.

See ops/bench/runner.py for how runs work, and each module for its benchmark.
"""

from __future__ import annotations

import sys

BENCHMARKS = {"frames": "ops.bench.frames:Frames"}


def build(name: str, grader, grader_model: str):
    module, cls = BENCHMARKS[name].split(":")
    return getattr(__import__(module, fromlist=[cls]), cls)(grader, grader_model)


def main(name: str, n: int, budget: float, seed: int = 0, label: str = "", out=sys.stdout) -> int:
    from harness.app import eval_setup, load_settings
    from ops.bench import runner
    tracer, grader, model, grader_model, factory = eval_setup()
    bench = build(name, grader, grader_model)
    folder = load_settings().home / "benchmarks" / name
    items = runner.sample(bench.load(folder), n, seed)
    out.write(f"{name}: {len(items)} questions (seed {seed}), budget ${budget:.2f}, model {model}\n")
    out.flush()

    def progress(r: dict) -> None:
        mark = "PASS" if r["passed"] else ("ERR " if r.get("error") else "FAIL")
        said = (r.get("judge") or {}).get("extracted") or r.get("error") or ""
        out.write(f"{mark}  {r['id']}  ${r['cost_usd']:.3f}  {str(said)[:70]}\n")
        out.flush()

    summary = runner.run(bench, items, factory, tracer, budget, label, model, progress)
    out.write("\n" + runner.report(summary) + "\n")
    return 0
