"""A call on a dead connection gives up and retries in seconds, not 10 minutes.
(LongMemEval, 2026-11-16: the Mac slept mid-call, and five calls on dead
connections stalled the run overnight under the SDK's 600 s default.)"""

import socket
import threading
import time

import anthropic
import pytest

from loop import providers


@pytest.fixture
def silent_server():
    """Accepts connections and never answers, like a peer that vanished."""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(8)
    accepted = []

    def serve():
        while True:
            try:
                conn, _ = server.accept()
            except OSError:
                return
            accepted.append(conn)                     # keep it open, say nothing
    threading.Thread(target=serve, daemon=True).start()
    yield f"http://127.0.0.1:{server.getsockname()[1]}", accepted
    server.close()
    for c in accepted:
        c.close()


def test_a_silent_connection_times_out_and_is_retried(silent_server):
    url, accepted = silent_server
    client = providers.new_anthropic_client(timeout=0.5, max_retries=1, api_key="test", base_url=url)
    started = time.monotonic()
    with pytest.raises(anthropic.APITimeoutError):
        client.messages.create(model="claude-haiku-5-5", max_tokens=10,
                               messages=[{"role": "user", "content": "hi"}])
    assert time.monotonic() - started < 10          # not 600 s per try
    assert len(accepted) == 2                         # the first try, then one retry


def test_defaults_are_short_and_retry_more_than_the_sdk():
    client = providers.new_anthropic_client(api_key="test")
    assert client.timeout.read == providers.API_TIMEOUT == 120
    assert client.timeout.connect == 10
    assert client.max_retries == providers.API_RETRIES == 4
    assert providers.build("anthropic").client.timeout.read == 120   # what the app actually uses
