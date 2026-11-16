"""FRAMES (Google, 2024): multi-hop questions answered from Wikipedia.

824 questions, each needing facts from 2 to 15 Wikipedia articles combined
with arithmetic, dates or constraints, e.g. "If my future wife has the same
first name as the 15th first lady's mother and the surname of the second
assassinated president's mother's maiden name, what is her name?"
(Jane Ballou). Apache-2.0: huggingface.co/datasets/google/frames-benchmark

sora gets the question as a user would ask it, with web search on, and is
asked to end with an exact answer. A model grader compares that answer with
the gold one (ops/bench/grading.py). Each question carries reasoning types
(numerical, tabular, multiple constraints, temporal, post-processing), which
the report breaks the score down by.
"""

from __future__ import annotations

import csv
from pathlib import Path

from ops.bench.grading import grade_answer
from ops.bench.runner import Item, download

URL = "https://huggingface.co/datasets/google/frames-benchmark/resolve/main/test.tsv"

PROMPT = """{question}

Look things up on the web as needed; don't rely on memory for facts that could have changed.
Finish your reply with one line: Exact answer: <your answer>"""


class Frames:
    name = "frames"
    group_by = "reasoning"

    def __init__(self, grader, grader_model: str):
        self.grader, self.grader_model = grader, grader_model

    def load(self, folder: Path) -> list[Item]:
        path = download(URL, folder / "test.tsv")
        csv.field_size_limit(10 ** 8)
        with path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        return [Item(f"frames-{int(r[''] or n):03d}", r["Prompt"].strip(), r["Answer"].strip(),
                     {"reasoning": r.get("reasoning_types", "").strip()})
                for n, r in enumerate(rows)]

    def case(self, item: Item) -> dict:
        return {"message": PROMPT.format(question=item.question)}

    def grade(self, item: Item, turn: dict) -> dict:
        return grade_answer(self.grader, self.grader_model, item.question, turn.get("reply") or "", item.answer)
