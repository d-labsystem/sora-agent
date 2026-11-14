"""The event hub and the database watcher, without a server."""

import sqlite3
import time

from harness.events import Hub, Watcher, sse
from memory.store import MemoryStore
from ops.tracing import Tracer


def test_hub_fans_out_and_never_blocks():
    hub = Hub()
    a, b = hub.subscribe(), hub.subscribe()
    hub.publish("sessions")
    assert a.get_nowait() == {"topic": "sessions"} and b.get_nowait() == {"topic": "sessions"}
    for _ in range(150):                      # b never reads: its queue fills, publish still returns
        hub.publish("turns")
    assert a.qsize() == 100 and hub.listeners == 2
    hub.unsubscribe(b)
    hub.close()
    assert hub.listeners == 1


def test_watcher_sees_changes_from_another_process(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    Tracer(tmp_path / "traces.db")
    running = {"name": None}
    hub = Hub()
    q = hub.subscribe()
    watcher = Watcher(hub, tmp_path / "memory.db", tmp_path / "traces.db", interval=60,
                      extra=lambda: {"evals": running["name"]})
    watcher.last = watcher.snapshot()
    assert watcher.check() == []                                  # nothing changed

    store.ensure_session(None, "hello")
    assert watcher.check() == ["sessions"]

    # a different connection, as the terminal (another process) would use
    other = sqlite3.connect(tmp_path / "memory.db")
    other.execute("INSERT INTO facts VALUES ('Aryan likes chai', 'cli', datetime('now'))")
    other.commit()
    assert watcher.check() == ["memory"]

    tracer = Tracer(tmp_path / "traces.db")
    turn = tracer.start("hi")
    assert watcher.check() == ["turns"]
    turn.finish("hello")
    tracer.set_feedback(turn.id, -1, "too long")
    assert watcher.check() == ["turns"]                           # finish + rating: one notice per tick

    running["name"] = "smoke.jsonl"
    assert watcher.check() == ["evals"]
    assert [q.get_nowait()["topic"] for _ in range(q.qsize())] == ["sessions", "memory", "turns", "turns", "evals"]
    watcher.stop()


def test_watcher_thread_publishes_only_with_listeners(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    Tracer(tmp_path / "traces.db")
    hub = Hub()
    watcher = Watcher(hub, tmp_path / "memory.db", tmp_path / "traces.db", interval=0.05).start()
    store.ensure_session(None, "nobody is watching")
    time.sleep(0.2)
    assert watcher.last["sessions"][0][0] == 0                   # skipped: no tabs open
    q = hub.subscribe()
    time.sleep(0.2)
    assert q.get(timeout=1) == {"topic": "sessions"}
    watcher.stop()


def test_sse_frames():
    assert sse({"topic": "x"}) == b'data: {"topic": "x"}\n\n'
    assert sse(comment="keepalive") == b": keepalive\n\n"
