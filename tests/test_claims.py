"""Claim checking: a reply's claimed actions against what the turn actually did."""

from fake import FakeClient
from ops.claims import check_claims, claims_action, evidence

SAVED = [{"tool": "write_file", "args": {"path": "notes.md", "content": "x" * 500},
          "output": "Wrote notes.md (500 bytes)", "is_error": False}]
FAILED = [{"tool": "calendar__create", "args": {"title": "Dentist"}, "output": "invalid date", "is_error": True}]
SERVER = [{"tool": "code_execution", "args": {"code": "make_deck()"}, "output": "saved deck.pptx"}]


def claim(text, n, why="r"):
    return {"claim": text, "evidence": n, "why": why}


def test_which_replies_claim_an_action():
    for reply in ("I've saved your notes to notes.md.", "Done! I added the dentist to your calendar.",
                  "I have remembered that Priya lives in Pune.", "All set, I scheduled it for 3 pm."):
        assert claims_action(reply), reply
    for reply in ("Paris is the capital of France.", "Want me to save this as a file?",
                  "Here's a summary of the article.", ""):
        assert not claims_action(reply), reply


def test_evidence_says_what_happened_and_where():
    lines = evidence(SAVED + FAILED, SERVER, ["report.pdf"])
    assert lines[0].startswith("write_file(path='notes.md', content='xxx") and "-> ok: Wrote notes.md" in lines[0]
    assert "-> FAILED: invalid date" in lines[1]
    assert "[ran on the provider's servers, not the user's computer] -> printed: saved deck.pptx" in lines[2]
    assert lines[3] == "file saved to the user's workspace: report.pdf"


def test_no_action_words_means_no_call():
    client = FakeClient(small={"claims": []})
    assert check_claims(client, "m", "Paris is the capital of France.", [], [], []) is None
    assert client.small_requests == []


def test_a_supported_claim_passes_and_an_unsupported_one_is_flagged():
    client = FakeClient(small={"claims": [claim("saved notes.md", 1), claim("added dentist to calendar", 0, "it failed")]})
    found = check_claims(client, "m", "I've saved notes.md and added the dentist to your calendar.",
                         SAVED + FAILED, [], [])
    assert [c["supported"] for c in found.claims] == [True, False] and not found.ok
    assert found.unsupported[0]["why"] == "it failed"
    prompt = client.small_requests[0]["messages"][0]["content"]
    assert "[1] write_file(path='notes.md'" in prompt and "[2] calendar__create" in prompt


def test_the_deck_case_server_side_file_claimed_as_delivered():
    """2026-11-16: the deck was built on the provider's servers and never reached the user."""
    client = FakeClient(small={"claims": [claim("saved deck.pptx to your workspace", 0, "only on provider's servers")]})
    found = check_claims(client, "m", "I've saved deck.pptx to your workspace.", [], SERVER, [])
    assert not found.ok                    # named only in the server's output: never reached the user
    client = FakeClient(small={"claims": [claim("saved deck.pptx to your workspace", 0)]})
    assert check_claims(client, "m", "I've saved deck.pptx to your workspace.", [], SERVER, ["deck.pptx"]).ok


def test_a_file_in_the_workspace_is_never_called_a_false_claim(tmp_path):
    (tmp_path / "plan.md").write_text("x")
    client = FakeClient(small={"claims": [claim("created plan.md", 0)]})
    assert check_claims(client, "m", "I created plan.md.", [], [], [], workspace=tmp_path).ok
    (tmp_path / "Q3 report.pdf").write_text("x")
    client = FakeClient(small={"claims": [claim("exported Q3 report.pdf", 0)]})
    assert check_claims(client, "m", "I've exported Q3 report.pdf.", [], [], [], workspace=tmp_path).ok


def test_bad_numbers_count_as_no_evidence_and_a_failed_call_flags_nothing():
    client = FakeClient(small={"claims": [claim("sent the email", 7), claim("", 0)]})
    found = check_claims(client, "m", "I sent the email.", SAVED, [], [])
    assert len(found.claims) == 1 and not found.ok
    found = check_claims(FakeClient(small=RuntimeError("overloaded")), "m", "I sent the email.", [], [], [])
    assert found.failed and found.ok


def test_the_record_can_be_rebuilt_from_a_saved_trace():
    from ops.claims import record_from_spans
    spans = [{"kind": "llm", "name": "model call 1", "ok": True, "data": {}},
             {"kind": "tool", "name": "remember", "ok": True, "data": {"args": {"fact": "x"}, "output": "Remembered: x"}},
             {"kind": "tool", "name": "calendar__create", "ok": False, "data": {"args": {"title": "Dentist"}, "output": "bad date"}},
             {"kind": "server_tool", "name": "code_execution", "ok": True, "data": {"input": {"code": "deck()"}}},
             {"kind": "server_result", "name": "code_execution_tool_result", "ok": True, "data": {"stdout": "saved deck.pptx"}},
             {"kind": "file", "name": "saved deck.pptx", "ok": True, "data": {"path": "deck.pptx"}},
             {"kind": "file", "name": "file download failed", "ok": False, "data": {"error": "x"}}]
    tools, server, files = record_from_spans(spans)
    assert [(t["tool"], t["is_error"]) for t in tools] == [("remember", False), ("calendar__create", True)]
    assert server == [{"tool": "code_execution", "args": {"code": "deck()"}, "output": "saved deck.pptx"}]
    assert files == ["deck.pptx"]
