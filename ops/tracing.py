"""Tracing: a complete, queryable record of every turn.

A turn is one user message and everything the agent did to answer it. A span
is one step inside a turn, the same vocabulary OpenTelemetry, LangSmith and
Langfuse use:

    turn  "plan my week"                              4.2s   $0.031  ok
    ├── llm     retrieval gate        haiku          0.4s   $0.0001
    ├── gate    memory recall         3 facts found
    ├── llm     model call 1          opus           1.9s   $0.024
    ├── tool    workspace__read_file  ok             0.1s
    ├── llm     model call 2          opus           1.6s   $0.007
    └── memory  consolidation         2 facts saved

Every module in sora reports through one callback, notify(kind, **data).
A Turn turns those calls into spans, and also forwards them to a live
listener, so the chat UI can show tool calls while they happen.

Storage: SQLite (traces.db, separate from memory.db so wiping traces never
touches memory). Fixed fields are columns, so the dashboard can aggregate
with SQL. Each span's details are a JSON column, because every kind of step
records different things; SQLite's json_extract can still query inside it.
For learning and analysis, export_jsonl writes one self-contained JSON line
per turn: message, spans, reply, feedback and scores.

Totals (tokens, searches, cost) are summed from the llm spans when the turn
finishes, so there's one source of truth: if a step spent money, it has a
span, and the turn's cost is the sum of its spans.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    source TEXT NOT NULL DEFAULT 'chat',          -- chat | cli | eval
    eval_run TEXT,
    message TEXT NOT NULL,
    reply TEXT,
    model TEXT,
    status TEXT NOT NULL DEFAULT 'running',       -- running | ok | error | refused | limit
    error TEXT,
    iterations INTEGER DEFAULT 0,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    web_searches INTEGER DEFAULT 0,
    tool_calls INTEGER DEFAULT 0,
    tool_errors INTEGER DEFAULT 0,
    cost_usd REAL DEFAULT 0,
    latency_ms INTEGER,
    checks TEXT,                                  -- JSON: {"name": passed}
    feedback INTEGER,                             -- 1 thumbs up, -1 thumbs down
    feedback_note TEXT,
    judge_score REAL,
    judge_reason TEXT,
    started_at TEXT NOT NULL,
    ended_at TEXT);
CREATE INDEX IF NOT EXISTS turns_by_time ON turns(started_at);
CREATE INDEX IF NOT EXISTS turns_by_session ON turns(session_id, started_at);
CREATE TABLE IF NOT EXISTS spans (
    id INTEGER PRIMARY KEY,
    turn_id TEXT NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    name TEXT,
    start_ms INTEGER NOT NULL,                    -- offset from the turn's start
    duration_ms INTEGER NOT NULL DEFAULT 0,
    ok INTEGER NOT NULL DEFAULT 1,
    data TEXT);                                   -- JSON details
CREATE INDEX IF NOT EXISTS spans_by_turn ON spans(turn_id, id);
CREATE TABLE IF NOT EXISTS eval_runs (
    id TEXT PRIMARY KEY,
    dataset TEXT,
    model TEXT,
    cases INTEGER DEFAULT 0,
    passed INTEGER DEFAULT 0,
    cost_usd REAL DEFAULT 0,
    results TEXT,                                 -- JSON per case
    started_at TEXT NOT NULL,
    ended_at TEXT);
"""

LIVE_ONLY = {"text"}        # streamed text deltas go to the UI, not the database
MAX_VALUE = 4000            # characters per string stored in a span


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _clip(value: Any) -> Any:
    """Keep span data small: long strings are cut, nested values clipped too."""
    if isinstance(value, str):
        return value if len(value) <= MAX_VALUE else value[:MAX_VALUE] + f"... [{len(value)} chars]"
    if isinstance(value, dict):
        return {k: _clip(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clip(v) for v in value[:50]]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _clip(str(value))


class Turn:
    """One traced turn. `notify` is the callback every module calls."""

    def __init__(self, tracer: Tracer, turn_id: str, listener: Callable | None):
        self.tracer = tracer
        self.id = turn_id
        self.listener = listener
        self.started = time.perf_counter()
        self.finished = False

    def elapsed_ms(self) -> int:
        return int((time.perf_counter() - self.started) * 1000)

    def notify(self, kind: str, **data: Any) -> None:
        if self.listener:
            try:
                self.listener(kind, data)
            except Exception:   # a broken UI connection must not break the turn
                pass
        if kind in LIVE_ONLY:
            return
        duration = int(data.pop("ms", 0) or 0)
        ok = data.pop("ok", None)
        ok = (not data.get("is_error", False)) if ok is None else bool(ok)
        name = str(data.pop("name", kind))
        start = max(0, self.elapsed_ms() - duration)   # events are reported when they end
        self.tracer._write(
            "INSERT INTO spans (turn_id, kind, name, start_ms, duration_ms, ok, data) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (self.id, kind, name, start, duration, int(ok),
             json.dumps(_clip(data), ensure_ascii=False, default=str)))

    def finish(self, reply: str | None, status: str = "ok", model: str | None = None,
               iterations: int = 0, error: str | None = None,
               checks: dict[str, bool] | None = None) -> dict:
        """Close the turn and roll span totals up into it. Safe to call twice."""
        if self.finished:
            return self.tracer.turn(self.id)
        self.finished = True
        t = self.tracer
        with t.lock:
            llm = t.conn.execute(
                "SELECT COALESCE(SUM(json_extract(data,'$.input_tokens')),0) AS i, "
                "COALESCE(SUM(json_extract(data,'$.output_tokens')),0) AS o, "
                "COALESCE(SUM(json_extract(data,'$.web_searches')),0) AS w, "
                "COALESCE(SUM(json_extract(data,'$.cost_usd')),0) AS c "
                "FROM spans WHERE turn_id = ? AND kind = 'llm'", (self.id,)).fetchone()
            tools = t.conn.execute(
                "SELECT COUNT(*) AS n, COALESCE(SUM(1 - ok),0) AS bad "
                "FROM spans WHERE turn_id = ? AND kind = 'tool'", (self.id,)).fetchone()
            t.conn.execute(
                "UPDATE turns SET reply=?, status=?, model=COALESCE(?, model), iterations=?, "
                "error=?, checks=?, input_tokens=?, output_tokens=?, web_searches=?, cost_usd=?, "
                "tool_calls=?, tool_errors=?, latency_ms=?, ended_at=? WHERE id=?",
                (reply, status, model, iterations, error,
                 json.dumps(checks) if checks is not None else None,
                 llm["i"], llm["o"], llm["w"], round(llm["c"], 6), tools["n"], tools["bad"],
                 self.elapsed_ms(), now_iso(), self.id))
            t.conn.commit()
        return t.turn(self.id)


class Tracer:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(path, check_same_thread=False, timeout=10)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)
        self._recover()

    def _write(self, sql: str, params: tuple = ()) -> None:
        with self.lock:
            self.conn.execute(sql, params)
            self.conn.commit()

    def _recover(self) -> None:
        """Turns still 'running' at startup were cut off by a crash or restart."""
        self._write("UPDATE turns SET status='error', error='interrupted (process stopped)' "
                    "WHERE status='running'")

    # ---- writing

    def start(self, message: str, session_id: str | None = None, source: str = "chat",
              model: str | None = None, eval_run: str | None = None,
              listener: Callable | None = None) -> Turn:
        turn_id = uuid.uuid4().hex[:12]
        self._write("INSERT INTO turns (id, session_id, source, eval_run, message, model, started_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (turn_id, session_id, source, eval_run, message, model, now_iso()))
        return Turn(self, turn_id, listener)

    def set_feedback(self, turn_id: str, value: int, note: str = "") -> bool:
        if value not in (-1, 0, 1):
            raise ValueError("feedback is 1, -1, or 0 to clear")
        with self.lock:
            cur = self.conn.execute("UPDATE turns SET feedback=?, feedback_note=? WHERE id=?",
                                    (value or None, note or None, turn_id))
            self.conn.commit()
        return cur.rowcount == 1

    def set_judge(self, turn_id: str, score: float, reason: str) -> None:
        self._write("UPDATE turns SET judge_score=?, judge_reason=? WHERE id=?",
                    (score, reason, turn_id))

    def delete(self, turn_id: str) -> None:
        self._write("DELETE FROM turns WHERE id=?", (turn_id,))

    # ---- reading

    @staticmethod
    def _turn_row(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["checks"] = json.loads(d["checks"]) if d.get("checks") else None
        return d

    def turns(self, limit: int = 50, offset: int = 0, source: str | None = None,
              session_id: str | None = None, status: str | None = None,
              eval_run: str | None = None) -> list[dict]:
        where, params = [], []
        for column, value in (("source", source), ("session_id", session_id),
                              ("status", status), ("eval_run", eval_run)):
            if value:
                where.append(f"{column} = ?")
                params.append(value)
        sql = "SELECT * FROM turns" + (" WHERE " + " AND ".join(where) if where else "")
        sql += " ORDER BY started_at DESC LIMIT ? OFFSET ?"
        with self.lock:
            rows = self.conn.execute(sql, (*params, limit, offset)).fetchall()
        return [self._turn_row(r) for r in rows]

    def spans(self, turn_id: str) -> list[dict]:
        with self.lock:
            rows = self.conn.execute("SELECT kind, name, start_ms, duration_ms, ok, data "
                                     "FROM spans WHERE turn_id=? ORDER BY id", (turn_id,)).fetchall()
        return [{**dict(r), "ok": bool(r["ok"]), "data": json.loads(r["data"] or "{}")} for r in rows]

    def turn(self, turn_id: str) -> dict | None:
        with self.lock:
            row = self.conn.execute("SELECT * FROM turns WHERE id=?", (turn_id,)).fetchone()
        return {**self._turn_row(row), "spans": self.spans(turn_id)} if row else None

    def stats(self, days: int = 7, source: str = "chat") -> dict:
        """Headline numbers for the dashboard over the last `days`."""
        since = datetime.fromtimestamp(time.time() - days * 86400, timezone.utc).isoformat()
        with self.lock:
            c = self.conn
            head = dict(c.execute(
                "SELECT COUNT(*) AS turns, "
                "COALESCE(SUM(status != 'ok' AND status != 'running'),0) AS failed, "
                "COALESCE(SUM(cost_usd),0) AS cost_usd, AVG(latency_ms) AS avg_latency_ms, "
                "COALESCE(SUM(web_searches),0) AS web_searches, "
                "COALESCE(SUM(tool_calls),0) AS tool_calls, COALESCE(SUM(tool_errors),0) AS tool_errors, "
                "COALESCE(SUM(feedback = 1),0) AS thumbs_up, COALESCE(SUM(feedback = -1),0) AS thumbs_down, "
                "AVG(judge_score) AS avg_judge "
                "FROM turns WHERE started_at >= ? AND source = ?", (since, source)).fetchone())
            latencies = [r[0] for r in c.execute(
                "SELECT latency_ms FROM turns WHERE started_at >= ? AND source = ? "
                "AND latency_ms IS NOT NULL ORDER BY latency_ms", (since, source))]
            per_day = [dict(r) for r in c.execute(
                "SELECT substr(started_at,1,10) AS day, COUNT(*) AS turns, "
                "ROUND(SUM(cost_usd),6) AS cost_usd FROM turns WHERE started_at >= ? AND source = ? "
                "GROUP BY day ORDER BY day", (since, source))]
            per_tool = [dict(r) for r in c.execute(
                "SELECT s.name AS tool, COUNT(*) AS calls, SUM(1 - s.ok) AS errors, "
                "ROUND(AVG(s.duration_ms)) AS avg_ms FROM spans s JOIN turns t ON t.id = s.turn_id "
                "WHERE s.kind = 'tool' AND t.started_at >= ? AND t.source = ? "
                "GROUP BY s.name ORDER BY calls DESC", (since, source))]
        pct = lambda p: latencies[min(len(latencies) - 1, int(p * len(latencies)))] if latencies else None
        head.update(p50_latency_ms=pct(0.5), p95_latency_ms=pct(0.95),
                    error_rate=head["failed"] / head["turns"] if head["turns"] else 0.0,
                    per_day=per_day, per_tool=per_tool, days=days)
        head["cost_usd"] = round(head["cost_usd"], 6)
        return head

    def export_jsonl(self, out: IO[str], source: str | None = None, since: str | None = None) -> int:
        """Write one JSON object per finished turn, with its spans. Returns the count.
        This is the format to load into pandas or turn into eval cases."""
        where, params = ["status != 'running'"], []
        if source:
            where.append("source = ?")
            params.append(source)
        if since:
            where.append("started_at >= ?")
            params.append(since)
        with self.lock:
            ids = [r[0] for r in self.conn.execute(
                f"SELECT id FROM turns WHERE {' AND '.join(where)} ORDER BY started_at", params)]
        for turn_id in ids:
            out.write(json.dumps(self.turn(turn_id), ensure_ascii=False) + "\n")
        return len(ids)

    # ---- eval runs (written by ops/evals.py)

    def save_eval_run(self, run: dict) -> None:
        self._write(
            "INSERT OR REPLACE INTO eval_runs (id, dataset, model, cases, passed, cost_usd, "
            "results, started_at, ended_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (run["id"], run["dataset"], run["model"], run["cases"], run["passed"],
             run["cost_usd"], json.dumps(run["results"], ensure_ascii=False),
             run["started_at"], run["ended_at"]))

    def eval_runs(self, limit: int = 50) -> list[dict]:
        with self.lock:
            rows = self.conn.execute("SELECT id, dataset, model, cases, passed, cost_usd, started_at, "
                                     "ended_at FROM eval_runs ORDER BY started_at DESC LIMIT ?",
                                     (limit,)).fetchall()
        return [{**dict(r), "pass_rate": r["passed"] / r["cases"] if r["cases"] else 0.0} for r in rows]

    def eval_run(self, run_id: str) -> dict | None:
        with self.lock:
            row = self.conn.execute("SELECT * FROM eval_runs WHERE id=?", (run_id,)).fetchone()
        if not row:
            return None
        return {**dict(row), "results": json.loads(row["results"] or "[]"),
                "pass_rate": row["passed"] / row["cases"] if row["cases"] else 0.0}
