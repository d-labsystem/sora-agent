"""WhatsApp gateway: signed requests in, replies out, without Twilio or Meta."""

import json
import threading
import urllib.error
import urllib.parse
import urllib.request

import pytest

from harness import whatsapp as wa

ME = "919800000000"
URL = "https://abc.trycloudflare.com/whatsapp"


class FakeAgent:
    def __init__(self, reply="Hi from sora", spans=(), fail=False):
        self.calls, self.reply, self.spans, self.fail = [], reply, list(spans), fail

    def respond(self, message, session_id=None, source="chat"):
        self.calls.append((message, session_id, source))
        if self.fail:
            return {"reply": None, "status": "error", "error": "No valid API key", "session_id": session_id or "s-new"}
        return {"reply": self.reply, "status": "ok", "session_id": session_id or f"s{len(self.calls)}",
                "spans": self.spans}


class Sent:
    """Stands in for http_post: records what would go to the provider."""

    def __init__(self, fail=None):
        self.requests, self.fail = [], fail

    def __call__(self, url, data, headers, timeout=20):
        if self.fail:
            raise wa.SendError(self.fail)
        self.requests.append((url, data, headers))
        return {"sid": "SM1", "messages": [{"id": "wamid.1"}]}

    def bodies(self):
        out = []
        for url, data, _ in self.requests:
            if "twilio" in url:
                out.append(urllib.parse.parse_qs(data.decode())["Body"][0])
            else:
                payload = json.loads(data)
                if payload.get("type") == "text":
                    out.append(payload["text"]["body"])
        return out


def twilio(sent=None):
    return wa.Twilio("AC123", "secret-token", post=sent or Sent())


def twilio_form(provider, url=URL, **fields):
    params = {"From": f"whatsapp:+{ME}", "To": "whatsapp:+14155238886", "Body": "hello",
              "MessageSid": "SM-in-1", "NumMedia": "0", "ProfileName": "Aryan", **fields}
    return urllib.parse.urlencode(params).encode(), {"X-Twilio-Signature": provider.signature(url, params)}


def gateway(tmp_path, agent=None, provider=None):
    return wa.Gateway(agent or FakeAgent(), provider or twilio(), [f"+{ME}"], tmp_path / "sessions.json")


# ---- Twilio: signatures and parsing

def test_twilio_signature_matches_twilios_published_example():
    """The vector from twilio-python's own tests (tests/unit/test_request_validator.py)."""
    t = wa.Twilio("AC", "12345")
    params = {"CallSid": "CA1234567890ABCDE", "Caller": "+14158675309", "Digits": "1234",
              "From": "+14158675309", "To": "+18005551212"}
    assert t.signature("https://mycompany.com/myapp.php?foo=1&bar=2", params) == "RSOYDt4T1cUTdK1PDd93/VVr8B8="


def test_twilio_accepts_signed_and_rejects_forged_requests():
    t = twilio()
    body, headers = twilio_form(t)
    parsed = t.parse(URL, headers, body)
    [m] = parsed.messages
    assert (m.sender, m.text, m.message_id, m.name, m.media) == (ME, "hello", "SM-in-1", "Aryan", False)
    assert parsed.reply[1] == "text/xml"
    assert t.parse(URL, {"X-Twilio-Signature": "forged"}, body) is None
    assert t.parse("https://other.host/whatsapp", headers, body) is None      # signed for another URL
    tampered = body.replace(b"hello", b"rm+-rf")
    assert t.parse(URL, headers, tampered) is None


def test_twilio_sends_through_the_messages_api():
    sent = Sent()
    wa.Twilio("AC123", "tok", post=sent).send(ME, "hi")
    url, data, headers = sent.requests[0]
    assert url == "https://api.twilio.com/2010-04-01/Accounts/AC123/Messages.json"
    assert urllib.parse.parse_qs(data.decode()) == {"From": ["whatsapp:+14155238886"], "To": [f"whatsapp:+{ME}"],
                                                    "Body": ["hi"]}
    assert headers["Authorization"].startswith("Basic ")


# ---- Meta: verification, signatures, parsing

def meta(sent=None):
    return wa.Meta("EAAtoken", "1428176873714282", "app-secret", "my-verify", post=sent or Sent())


def meta_payload(messages=(), statuses=()):
    return json.dumps({"object": "whatsapp_business_account", "entry": [{"id": "1", "changes": [{"field": "messages", "value": {
        "messaging_product": "whatsapp", "metadata": {"phone_number_id": "1428176873714282"},
        "contacts": [{"wa_id": ME, "profile": {"name": "Aryan"}}],
        "messages": list(messages), "statuses": list(statuses)}}]}]}).encode()


def test_meta_webhook_check_echoes_the_challenge_only_with_the_right_token():
    m = meta()
    assert m.verify_get({"hub.mode": "subscribe", "hub.verify_token": "my-verify", "hub.challenge": "42"}) == \
        (200, "text/plain", b"42")
    assert m.verify_get({"hub.mode": "subscribe", "hub.verify_token": "wrong", "hub.challenge": "42"})[0] == 403


def test_meta_parses_text_media_and_the_locked_account_status():
    m = meta()
    body = meta_payload(
        messages=[{"from": ME, "id": "wamid.A", "type": "text", "text": {"body": "hey"}},
                  {"from": ME, "id": "wamid.B", "type": "audio", "audio": {"id": "x"}}],
        statuses=[{"status": "failed", "errors": [{"code": 131031, "title": "Business Account locked"}]}])
    parsed = m.parse(URL, {"X-Hub-Signature-256": m.signature(body)}, body)
    assert [(x.text, x.media, x.name) for x in parsed.messages] == [("hey", False, "Aryan"), ("", True, "Aryan")]
    assert "131031" in parsed.problems[0] and "locked" in parsed.problems[0]
    assert m.parse(URL, {"X-Hub-Signature-256": "sha256=00"}, body) is None


def test_meta_sends_text_and_read_receipts():
    sent = Sent()
    m = meta(sent)
    m.send(ME, "hi")
    m.mark_read("wamid.A")
    (url, data, headers), (_, receipt, _) = sent.requests
    assert url == "https://graph.facebook.com/v23.0/1428176873714282/messages"
    assert json.loads(data)["text"]["body"] == "hi" and headers["Authorization"] == "Bearer EAAtoken"
    assert json.loads(receipt) == {"messaging_product": "whatsapp", "status": "read", "message_id": "wamid.A"}


# ---- the gateway

def test_only_allowed_numbers_reach_sora_and_retries_are_dropped(tmp_path):
    agent = FakeAgent()
    g = gateway(tmp_path, agent)
    assert g.accept(wa.Incoming("15550001111", "hi", "m1")) == "not allowed"
    assert g.accept(wa.Incoming(ME, "hi", "m2")) == "queued"
    assert g.accept(wa.Incoming(ME, "hi", "m2")) == "duplicate"           # the provider retried
    g.drain()
    assert agent.calls == [("hi", None, "whatsapp")]


def test_one_conversation_per_number_and_new_starts_over(tmp_path):
    agent, sent = FakeAgent(), Sent()
    g = gateway(tmp_path, agent, twilio(sent))
    for i, text in enumerate(["one", "two", "/new", "three"]):
        g.accept(wa.Incoming(ME, text, f"m{i}"))
    g.drain()
    assert [c[1] for c in agent.calls] == [None, "s1", None]               # 'three' starts fresh
    assert sent.bodies() == ["Hi from sora", "Hi from sora", wa.NEW_REPLY, "Hi from sora"]
    again = gateway(tmp_path, agent, twilio(sent))                         # survives a restart
    assert again.sessions == {ME: "s3"}


def test_media_errors_and_charts_get_sensible_replies(tmp_path):
    sent = Sent()
    g = gateway(tmp_path, FakeAgent(fail=True), twilio(sent))
    g.accept(wa.Incoming(ME, "", "m1", media=True))
    g.accept(wa.Incoming(ME, "hello", "m2"))
    g.drain()
    assert sent.bodies()[0] == wa.MEDIA_REPLY
    assert "No valid API key" in sent.bodies()[1]
    sent2 = Sent()
    g2 = gateway(tmp_path, FakeAgent(spans=[{"kind": "tool", "name": "show_chart"}]), twilio(sent2))
    g2.accept(wa.Incoming(ME, "chart it", "m3"))
    g2.drain()
    assert sent2.bodies()[0].endswith(wa.CHART_NOTE)


def test_a_failed_send_is_logged_not_raised(tmp_path, caplog):
    g = gateway(tmp_path, FakeAgent(), twilio(Sent(fail="HTTP 400 code 63015: not in sandbox")))
    g.accept(wa.Incoming(ME, "hi", "m1"))
    g.drain()
    assert "63015" in caplog.text


# ---- formatting

def test_markdown_becomes_whatsapp_formatting():
    md = ("## Plan\n**Bold** and *italic* and ~~gone~~ and [docs](https://x.dev/a).\n- one\n- two\n\n"
          "| a | b |\n|---|---|\n| 1 | 2 |\n```\n**not bold in code**\n```")
    assert wa.to_whatsapp(md) == ("*Plan*\n*Bold* and _italic_ and ~gone~ and docs (https://x.dev/a).\n• one\n• two\n\n"
                                  "a  b\n1  2\n```\n**not bold in code**\n```")


def test_long_replies_split_on_boundaries_and_are_numbered():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 60 for i in range(12))
    pieces = wa.split(text, 1600)
    assert len(pieces) > 1 and all(len(p) <= 1600 for p in pieces)
    assert pieces[0].endswith(f"(1/{len(pieces)})") and pieces[1].startswith("Paragraph")
    assert wa.split("short", 1600) == ["short"]
    assert all(len(p) <= 100 for p in wa.split("x" * 500, 100))          # one giant word still fits
    long_paragraph = "\n".join(f"line {i} " + "y" * 30 for i in range(80))   # one paragraph, many lines
    parts = wa.split(long_paragraph, 400)
    assert all("\n\n" not in p for p in parts)                           # no blank lines invented
    assert "".join(p.rsplit(" (", 1)[0] + "\n" for p in parts).split() == long_paragraph.split()


# ---- the server, end to end over HTTP

def test_the_webhook_server_end_to_end(tmp_path):
    from http.server import ThreadingHTTPServer
    sent, agent = Sent(), FakeAgent()
    provider = twilio(sent)
    g = gateway(tmp_path, agent, provider)
    server = ThreadingHTTPServer(("127.0.0.1", 0), wa.make_handler(g, public_url="https://abc.trycloudflare.com"))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        body, headers = twilio_form(provider)
        req = urllib.request.Request(base + "/whatsapp", data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req) as r:
            assert r.status == 200 and b"<Response>" in r.read()
        forged = urllib.request.Request(base + "/whatsapp", data=body, headers={"X-Twilio-Signature": "x"}, method="POST")
        with pytest.raises(urllib.error.HTTPError) as no:
            urllib.request.urlopen(forged)
        assert no.value.code == 403
        with pytest.raises(urllib.error.HTTPError) as nf:
            urllib.request.urlopen(base + "/api/memory")                  # nothing else is served
        assert nf.value.code == 404
        g.drain()
        assert agent.calls == [("hello", None, "whatsapp")] and sent.bodies() == ["Hi from sora"]
    finally:
        server.shutdown()


# ---- settings

def test_provider_is_picked_from_settings(tmp_path):
    from harness.app import Settings
    assert wa.provider_from(Settings(home=tmp_path))[0] is None
    p, _ = wa.provider_from(Settings(home=tmp_path, twilio_account_sid="AC", twilio_auth_token="t"))
    assert p.name == "twilio"
    p, why = wa.provider_from(Settings(home=tmp_path, whatsapp_provider="meta", whatsapp_access_token="x"))
    assert p is None and "WHATSAPP_APP_SECRET" in why


def test_whatsapp_turns_tell_the_model_where_the_user_is():
    from datetime import datetime
    from harness.app import CHANNEL_NOTES, build_system
    on_phone = build_system("soul", datetime.now(), "", [], [], None, channel="whatsapp")
    in_chat = build_system("soul", datetime.now(), "", [], [], None, channel="chat")
    assert CHANNEL_NOTES["whatsapp"] in on_phone[1]["text"] and on_phone[0] == in_chat[0]   # cache prefix unchanged
    assert "WhatsApp" not in in_chat[1]["text"]
