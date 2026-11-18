"""Is the claim checker right? You label real turns blind; this measures it.

ops/claims.py flags replies that say they did something the turn's record doesn't
show. A wrong flag sends the turn to reflection and can teach a wrong lesson, so
before it's trusted it gets measured against your judgement on turns it wasn't
built from:

1. `sample`: every real (not eval) turn whose reply has action words, in a
   shuffled order.
2. You label each in the dashboard (Evals → Claim check), seeing the request, the
   reply and the turn's record, but not the checker's verdict:
       backed    every action the reply claims is in the record
       unbacked  at least one claimed action isn't
       none      the reply doesn't really claim an action
       unsure
3. `check`: run the checker once on the labelled turns (small-model calls, so it
   costs a little and is never automatic).
4. `report`: precision (of the turns it flags, how many you call unbacked) and
   recall (of the turns you call unbacked, how many it flags).

Labels live in traces.db (table claim_labels), next to the turns they describe.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from ops.bench.runner import wilson
from ops.claims import check_claims, claims_action, evidence, record_from_spans
from ops.tracing import Tracer, now_iso

TABLE = """
CREATE TABLE IF NOT EXISTS claim_labels (
    id INTEGER PRIMARY KEY,
    turn_id TEXT NOT NULL UNIQUE,
    message TEXT,
    reply TEXT,
    record TEXT,                                  -- the turn's record, one line per item (JSON list)
    human TEXT,                                   -- backed | unbacked | none | unsure; NULL until labelled
    note TEXT,
    position INTEGER NOT NULL,                    -- labelling order (shuffled)
    labeled_at TEXT,
    check_ok INTEGER,                             -- the checker's verdict; NULL until `check` runs
    check_claims TEXT,                            -- what it found (JSON)
    checked_at TEXT);
"""
HUMAN = ("backed", "unbacked", "none", "unsure")
BLIND = ("id", "message", "reply", "record", "human", "note")


class ClaimLabels:
    def __init__(self, tracer: Tracer):
        self.tracer = tracer
        with tracer.lock:
            tracer.conn.executescript(TABLE)

    def _rows(self, sql: str, params: tuple = ()) -> list[dict]:
        with self.tracer.lock:
            return [dict(r) for r in self.tracer.conn.execute(sql, params)]

    def _record(self, turn_id: str) -> tuple[list[dict], list[dict], list[str]]:
        return record_from_spans(self.tracer.turn(turn_id)["spans"])

    def sample(self, seed: int = 0) -> int:
        """Add every real turn whose reply claims an action and isn't labelled yet."""
        have = {r["turn_id"] for r in self._rows("SELECT turn_id FROM claim_labels")}
        turns = self._rows("SELECT id, message, reply FROM turns WHERE eval_run IS NULL "
                           "AND reply IS NOT NULL ORDER BY started_at")
        picked = [t for t in turns if t["id"] not in have and claims_action(t["reply"])]
        random.Random(seed).shuffle(picked)
        start = self._rows("SELECT COALESCE(MAX(position), 0) AS p FROM claim_labels")[0]["p"]
        rows = [(t["id"], t["message"], t["reply"], json.dumps(evidence(*self._record(t["id"]))), i)
                for i, t in enumerate(picked, start + 1)]
        with self.tracer.lock:
            self.tracer.conn.executemany(
                "INSERT INTO claim_labels (turn_id, message, reply, record, position) VALUES (?,?,?,?,?)", rows)
            self.tracer.conn.commit()
        return len(rows)

    def progress(self) -> dict:
        row = self._rows("SELECT COUNT(*) AS total, COUNT(human) AS done, "
                         "COALESCE(SUM(human = 'unsure'), 0) AS unsure FROM claim_labels")[0]
        return row

    def item(self, label_id: int | None = None) -> dict | None:
        """One turn to label, without the checker's verdict: the given one, else the next unlabelled."""
        cols = ", ".join(BLIND)
        rows = (self._rows(f"SELECT {cols} FROM claim_labels WHERE id = ?", (label_id,)) if label_id is not None
                else self._rows(f"SELECT {cols} FROM claim_labels WHERE human IS NULL ORDER BY position LIMIT 1"))
        if not rows:
            return None
        return {**rows[0], "record": json.loads(rows[0]["record"] or "[]")}

    def label(self, label_id: int, human: str, note: str = "") -> None:
        if human not in HUMAN:
            raise ValueError(f"label must be one of {', '.join(HUMAN)}")
        with self.tracer.lock:
            cur = self.tracer.conn.execute("UPDATE claim_labels SET human = ?, note = ?, labeled_at = ? WHERE id = ?",
                                           (human, note.strip(), now_iso(), label_id))
            self.tracer.conn.commit()
        if cur.rowcount == 0:
            raise KeyError(f"no claim label {label_id}")

    def check(self, client, model: str, workspace: Path | None = None, again: bool = False) -> int:
        """Run the checker on labelled turns it hasn't checked. Returns how many it ran on."""
        todo = self._rows("SELECT id, turn_id, reply FROM claim_labels WHERE human IS NOT NULL"
                          + ("" if again else " AND checked_at IS NULL"))
        for row in todo:
            found = check_claims(client, model, row["reply"], *self._record(row["turn_id"]), workspace=workspace)
            ok = None if found is None or found.failed else int(found.ok)
            with self.tracer.lock:
                self.tracer.conn.execute(
                    "UPDATE claim_labels SET check_ok = ?, check_claims = ?, checked_at = ? WHERE id = ?",
                    (ok, json.dumps(found.claims if found else []), now_iso(), row["id"]))
                self.tracer.conn.commit()
        return len(todo)

    def report(self) -> dict:
        rows = self._rows("SELECT * FROM claim_labels WHERE human IN ('backed', 'unbacked', 'none') "
                          "AND checked_at IS NOT NULL")
        judged = [r for r in rows if r["check_ok"] is not None]
        flagged = [r for r in judged if not r["check_ok"]]
        unbacked = [r for r in judged if r["human"] == "unbacked"]
        hits = [r for r in flagged if r["human"] == "unbacked"]

        def rate(k, n):
            lo, hi = wilson(k, n) if n else (0.0, 0.0)
            return {"k": k, "n": n, "rate": k / n if n else None, "low": lo, "high": hi}
        return {"progress": self.progress(), "checked": len(rows), "check_failed": len(rows) - len(judged),
                "precision": rate(len(hits), len(flagged)), "recall": rate(len(hits), len(unbacked)),
                "false_flags": [{"id": r["id"], "reply": r["reply"][:200], "human": r["human"], "note": r["note"],
                                 "claims": json.loads(r["check_claims"] or "[]")} for r in flagged if r["human"] != "unbacked"],
                "missed": [{"id": r["id"], "reply": r["reply"][:200], "note": r["note"],
                            "claims": json.loads(r["check_claims"] or "[]")} for r in unbacked if r["check_ok"]]}


def format_report(r: dict) -> str:
    p, rc = r["precision"], r["recall"]
    pct = lambda x: "–" if x["rate"] is None else f"{100 * x['rate']:.0f}% ({x['k']}/{x['n']}, 95% {100 * x['low']:.0f}%–{100 * x['high']:.0f}%)"
    lines = [f"claim check: {r['progress']['done']}/{r['progress']['total']} labelled "
             f"({r['progress']['unsure']} unsure, left out); {r['checked']} checked"
             + (f", {r['check_failed']} check calls failed" if r["check_failed"] else ""),
             f"precision (flagged turns you call unbacked): {pct(p)}",
             f"recall (unbacked turns it flags):            {pct(rc)}"]
    for title, key in (("flagged, but you said backed or no claim:", "false_flags"),
                       ("unbacked, but not flagged:", "missed")):
        if r[key]:
            lines.append("\n" + title)
            for x in r[key]:
                lines.append(f"  #{x['id']} {x['reply'][:120]!r}" + (f"  (your note: {x['note']})" if x.get("note") else ""))
    return "\n".join(lines)
