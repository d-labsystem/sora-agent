"""GAIA (Meta and Hugging Face, 2023): tasks a real assistant should handle.

Questions that need the web, files (spreadsheets, images, PDFs, documents),
arithmetic and several steps, each with one short unambiguous answer.
Validation set: 165 questions with public answers, in three levels
(53 / 86 / 26). The test set's answers are private, so we use validation.

Gated. You accept the terms on Hugging Face once, and HF_TOKEN downloads.
The terms ask that the data isn't reshared in a crawlable form, so it lives
only in ~/.sora/benchmarks/gaia/, never in the repo, and Langfuse receives
no text from these runs (SORA_OTEL_CONTENT keeps text local by default).

Scoring is GAIA's own, reimplemented from the scorer published with the
benchmark: no model judge, so it's free and the same every time.
  numbers   "$1,234" and "1234" match: $ % and commas are removed first
  lists     split on , or ;, compared element by element, same length
  strings   compared lowercase without spaces or punctuation
The answer is whatever follows the last "FINAL ANSWER:" in the reply.

Contamination: some GAIA answers have been posted online since 2023. A
question where a search or fetch touched GAIA itself is flagged.
"""

from __future__ import annotations

import os
import re
import string
import urllib.error
import urllib.request
from pathlib import Path

from ops.bench.runner import Item

BASE = "https://huggingface.co/datasets/gaia-benchmark/GAIA/resolve/main/2023/validation/"

# GAIA's answer-format instruction, as given to models in the paper's setup.
PROMPT = """You are answering a question from a test of general AI assistants. Work it out, using your
tools as needed, then finish your answer with this template:
FINAL ANSWER: [YOUR FINAL ANSWER]
YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers
and/or strings. For a number, don't use commas, and don't add units such as $ or percent signs unless
asked. For a string, don't use articles or abbreviations (e.g. for cities), and write digits in plain
text unless asked otherwise. For a comma separated list, apply these rules to each element.

Question: {question}"""

INPUT_KIND = {"xlsx": "spreadsheet", "csv": "spreadsheet", "png": "image", "jpg": "image", "pdf": "pdf",
              "docx": "document", "pptx": "document", "txt": "text/code", "py": "text/code", "jsonld": "text/code",
              "pdb": "text/code", "zip": "archive", "mp3": "audio (unsupported)"}


# ---- GAIA's scorer (from the benchmark's published scorer.py)

def normalize_number(text: str) -> float:
    for ch in ("$", "%", ","):
        text = text.replace(ch, "")
    try:
        return float(text)
    except ValueError:
        return float("inf")


def split_list(text: str) -> list[str]:
    return re.split(r"[,;]", text)


def normalize_str(text: str, remove_punct: bool = True) -> str:
    text = re.sub(r"\s", "", text).lower()
    return text.translate(str.maketrans("", "", string.punctuation)) if remove_punct else text


def is_float(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def score(answer: str | None, gold: str) -> bool:
    answer = "None" if answer is None else answer
    if is_float(gold):
        return normalize_number(answer) == float(gold)
    if any(ch in gold for ch in ",;"):
        golds, answers = split_list(gold), split_list(answer)
        if len(golds) != len(answers):
            return False
        return all(normalize_number(a) == float(g) if is_float(g)
                   else normalize_str(a, remove_punct=False) == normalize_str(g, remove_punct=False)
                   for a, g in zip(answers, golds))
    return normalize_str(answer) == normalize_str(gold)


def final_answer(reply: str) -> str | None:
    """Text after the last FINAL ANSWER:, without markdown emphasis or a trailing period."""
    found = list(re.finditer(r"final answer\s*:\s*(.+)", reply or "", re.I))
    if not found:
        return None
    text = found[-1].group(1).strip().strip("*_`").strip()
    return text[:-1] if text.endswith(".") and not is_float(text) else text


def contaminated(turn: dict) -> bool:
    """Did a search or fetch touch GAIA itself (where answers may be posted)?"""
    for span in turn.get("spans", []):
        data = span.get("data") or {}
        if span["kind"] == "server_tool":
            what = str((data.get("input") or {}).get("query", "")) + str((data.get("input") or {}).get("url", ""))
            if re.search(r"\bgaia\b|gaia-benchmark", what, re.I):
                return True
    return False


class Gaia:
    name = "gaia"
    group_by = ("level", "input")
    strata_key = "level"

    def __init__(self, grader=None, grader_model: str = ""):
        self.folder: Path | None = None          # grader unused: GAIA scores without a model

    def _get(self, name: str, path: Path) -> Path:
        if not path.is_file():
            token = os.getenv("HF_TOKEN", "")
            request = urllib.request.Request(BASE + name, headers={"Authorization": f"Bearer {token}"} if token else {})
            try:
                data = urllib.request.urlopen(request, timeout=120).read()
            except urllib.error.HTTPError as exc:
                if exc.code in (401, 403):
                    raise RuntimeError("GAIA is gated: accept its terms on huggingface.co/datasets/gaia-benchmark/GAIA "
                                       "and put a token that can read gated repos in HF_TOKEN") from exc
                raise
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        return path

    def load(self, folder: Path) -> list[Item]:
        import pyarrow.parquet as pq                 # the `bench` extra
        self.folder = folder
        rows = pq.read_table(self._get("metadata.parquet", folder / "metadata.parquet")).to_pylist()
        items = []
        for r in rows:
            ext = r["file_name"].rsplit(".", 1)[-1].lower() if r.get("file_name") else ""
            items.append(Item(r["task_id"], r["Question"], str(r["Final answer"]),
                              {"level": f"level {r['Level']}", "file": r.get("file_name") or "",
                               "input": INPUT_KIND.get(ext, f"other ({ext})") if ext else "none"}))
        return items

    def case(self, item: Item) -> dict:
        case = {"message": PROMPT.format(question=item.question)}
        if item.meta["file"]:
            case["attachments"] = [str(self._get(item.meta["file"], self.folder / "files" / item.meta["file"]))]
        return case

    def grade(self, item: Item, turn: dict) -> dict:
        answer = final_answer(turn.get("reply") or "")
        ok = score(answer, item.answer)
        flagged = contaminated(turn)
        reason = f"answered {answer!r}; gold {item.answer!r}" + (" [searched for GAIA itself]" if flagged else "")
        return {"checks": {"correct": ok},
                "judge": {"score": 1.0 if ok else 0.0, "grade": 5 if ok else 1, "reason": reason,
                          "extracted": answer, "contaminated": flagged}}
