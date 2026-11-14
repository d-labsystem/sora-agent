"""Server push for the dashboard: a hub of open browser tabs, and a watcher.

The browser keeps one connection open, GET /api/events, and the server writes
a line to it whenever something it shows has changed. This is Server-Sent
Events (SSE): plain HTTP, one direction (server to browser), and browsers
reconnect by themselves if the server restarts. WebSockets would allow both
directions, which the dashboard doesn't need.

    Hub       one queue per open tab. publish() drops a notice into every
              queue; each tab's request thread waits on its own queue and
              writes what arrives to its connection.

    Watcher   a thread that, once a second, takes a cheap fingerprint of each
              table the dashboard shows (row counts, newest timestamps) and
              publishes a topic when a fingerprint changes:
                  sessions  a chat was created or got a new message
                  turns     a turn started, finished, or was rated or judged
                  memory    facts, episodes or lessons changed
                  evals     an eval run started or finished

Why watch the databases instead of calling publish() from the code that
writes? Because not every writer is in this process: `uv run main.py` in a
terminal is a separate process writing to the same files. Comparing
fingerprints catches changes from any process, for a few tiny queries a
second.

Notices carry only the topic. The page then fetches what it needs through the
normal API, so there is one way to read data, and a missed notice only means
the next one brings the page up to date.
"""

from __future__ import annotations

import json
import logging
import queue
import sqlite3
import threading
from collections.abc import Callable
from pathlib import Path

log = logging.getLogger(__name__)

KEEPALIVE_SECONDS = 15     # a comment line, so proxies and browsers don't drop an idle connection


class Hub:
    def __init__(self):
        self._queues: set[queue.Queue] = set()
        self._lock = threading.Lock()

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=100)
        with self._lock:
            self._queues.add(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            self._queues.discard(q)

    @property
    def listeners(self) -> int:
        with self._lock:
            return len(self._queues)

    def publish(self, topic: str, **data) -> None:
        message = {"topic": topic, **data}
        with self._lock:
            for q in self._queues:
                try:
                    q.put_nowait(message)
                except queue.Full:      # a stuck tab must not block the others
                    pass

    def close(self) -> None:
        """Tell every open stream to end (None is the stop signal)."""
        with self._lock:
            for q in self._queues:
                try:
                    q.put_nowait(None)
                except queue.Full:
                    pass


def _readonly(path: Path) -> sqlite3.Connection:
    # mode=ro: the watcher can never write. Each SELECT outside a transaction
    # sees the latest committed data, so one long-lived connection is enough.
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False, timeout=5)


# topic -> (database, one query per table). Queried separately so a table that
# doesn't exist yet (lessons appears on first use) can't hide changes in the others.
FINGERPRINTS = {
    "sessions": ("memory", ["SELECT COUNT(*), MAX(updated_at) FROM sessions"]),
    "memory": ("memory", ["SELECT COUNT(*), MAX(rowid) FROM facts",
                          "SELECT COUNT(*), MAX(rowid) FROM episodes",
                          "SELECT COUNT(*), MAX(rowid), TOTAL(uses) FROM lessons"]),
    "turns": ("traces", ["SELECT COUNT(*), MAX(started_at), MAX(ended_at), "
                         "TOTAL(IFNULL(feedback, 0) * 1000 + LENGTH(IFNULL(feedback_note, ''))), "
                         "COUNT(judge_score) FROM turns"]),
    "evals": ("traces", ["SELECT COUNT(*), MAX(ended_at) FROM eval_runs"]),
}


class Watcher:
    def __init__(self, hub: Hub, memory_db: Path, traces_db: Path, interval: float = 1.0,
                 extra: Callable[[], dict] | None = None):
        self.hub, self.interval, self.extra = hub, interval, extra
        self.paths = {"memory": memory_db, "traces": traces_db}
        self.conns: dict[str, sqlite3.Connection] = {}
        self.last: dict[str, object] = {}
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="dashboard-watcher", daemon=True)

    def start(self) -> "Watcher":
        self.last = self.snapshot()          # changes before start are not news
        self.thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self.thread.is_alive():          # stop() is safe even if start() never ran
            self.thread.join(self.interval * 3)
        for conn in self.conns.values():
            conn.close()

    def snapshot(self) -> dict[str, object]:
        out: dict[str, object] = {}
        for topic, (db, queries) in FINGERPRINTS.items():
            parts = []
            for sql in queries:
                try:
                    conn = self.conns.get(db) or self.conns.setdefault(db, _readonly(self.paths[db]))
                    parts.append(conn.execute(sql).fetchone())
                except sqlite3.Error as exc:    # table not created yet, or a migration running
                    log.debug("watcher couldn't read %s: %s", topic, exc)
                    parts.append(None)
            out[topic] = tuple(parts)
        for topic, value in (self.extra() if self.extra else {}).items():
            out[topic] = (out.get(topic), value)
        return out

    def check(self) -> list[str]:
        """Compare with the last snapshot and publish what changed. Returns the topics."""
        now = self.snapshot()
        changed = [t for t in now if now[t] != self.last.get(t)]
        self.last = now
        for topic in changed:
            self.hub.publish(topic)
        return changed

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            if self.hub.listeners:          # nobody watching: skip the queries
                self.check()


def sse(data: dict | None = None, comment: str | None = None) -> bytes:
    """One Server-Sent Events frame: 'data: <json>' or a ': comment', then a blank line."""
    if comment is not None:
        return f": {comment}\n\n".encode()
    return f"data: {json.dumps(data)}\n\n".encode()
