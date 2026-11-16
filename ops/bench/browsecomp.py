"""BrowseComp (OpenAI, 2025): facts that take persistent browsing to find.

1,266 questions whose answers are short and easy to check but hard to find,
e.g. a match, a paper or a person pinned down by several obscure constraints.
Built to measure deep, persistent web research.

Encrypted against training data. OpenAI publishes the questions and answers
XOR-encrypted, so they don't end up in what models learn: each row's key is
SHA-256 of its "canary" string, repeated to the text's length. We decrypt in
memory (decrypt()); the plaintext is only ever cached in ~/.sora/benchmarks.

Each question runs as a long task (more steps, the planner, helpers) with a
per-question dollar cap; at the cap sora answers with what it has. OpenAI's
answer format asks for an explanation, an exact answer and a confidence;
the grader is our adaptation of OpenAI's (ops/bench/grading.py). The stated
confidence is kept, so the report can say whether it matches reality.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import re
from pathlib import Path

from ops.bench.grading import grade_answer
from ops.bench.runner import Item, download

URL = "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"

# OpenAI's query template for BrowseComp.
PROMPT = """{question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}"""


def derive_key(password: str, length: int) -> bytes:
    digest = hashlib.sha256(password.encode()).digest()
    return (digest * (length // len(digest) + 1))[:length]


def decrypt(ciphertext_b64: str, password: str) -> str:
    data = base64.b64decode(ciphertext_b64)
    return bytes(a ^ b for a, b in zip(data, derive_key(password, len(data)))).decode()


def encrypt(plaintext: str, password: str) -> str:
    """The inverse, for tests."""
    data = plaintext.encode()
    return base64.b64encode(bytes(a ^ b for a, b in zip(data, derive_key(password, len(data))))).decode()


def confidence(reply: str) -> int | None:
    found = re.findall(r"confidence\W{0,5}\s*(\d{1,3})\s*%?", reply or "", re.I)
    return min(100, int(found[-1])) if found else None


class BrowseComp:
    name = "browsecomp"
    group_by = "topic"
    # Per question, through the long-task cap. The cap is checked between model calls, and one
    # web-heavy call (several searches and page reads on the provider's side) can cost $0.30-0.90
    # by itself, so a question can end somewhat above it.
    settings = {"long_budget_usd": 0.50}

    def __init__(self, grader, grader_model: str):
        self.grader, self.grader_model = grader, grader_model

    def load(self, folder: Path) -> list[Item]:
        path = download(URL, folder / "browse_comp_test_set.csv")
        with path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        return [Item(f"bc-{n:04d}", decrypt(r["problem"], r["canary"]), decrypt(r["answer"], r["canary"]),
                     {"topic": r.get("problem_topic", "") or "other"})
                for n, r in enumerate(rows)]

    def case(self, item: Item) -> dict:
        return {"message": PROMPT.format(question=item.question), "long": True}

    def grade(self, item: Item, turn: dict) -> dict:
        reply = turn.get("reply") or ""
        graded = grade_answer(self.grader, self.grader_model, item.question, reply, item.answer)
        graded["judge"]["confidence"] = confidence(reply)
        return graded
