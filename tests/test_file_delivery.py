"""Files the provider's code makes reach the user's workspace, and the record
says what ran where. (2026-11-16: a deck was built four times on Anthropic's
servers, never reached the Mac, and each next turn read a history without
those runs and "confessed" it had never made the file.)"""

from types import SimpleNamespace as NS

from fake import FakeClient, message, text
from harness.app import action_note, save_to_workspace
from loop.agent import run_loop
from loop.providers import _output_files
from tools.registry import WEB_TOOLS, ToolRegistry


def quiet(*args, **kwargs):
    pass


def code_run(command: str, stdout: str, file_ids=()):
    """A server-side bash run and its result, as the API returns them."""
    use = NS(type="server_tool_use", id="srv_1", name="bash_code_execution", input={"command": command})
    result = NS(type="bash_code_execution_tool_result", tool_use_id="srv_1",
                content=NS(type="bash_code_execution_result", stdout=stdout, stderr="", return_code=0,
                           content=[NS(type="bash_code_execution_output", file_id=f) for f in file_ids]))
    return use, result


def with_files(client, files: dict):
    """Give the fake client a Files API: id -> (name, bytes)."""
    client.files = NS(retrieve_metadata=lambda fid: NS(filename=files[fid][0]),
                      download=lambda fid: NS(read=lambda: files[fid][1]))
    return client


def test_output_file_ids_are_read_from_a_code_result():
    _, result = code_run("cp deck.pptx $OUTPUT_DIR/", "", ["file_a", "file_b"])
    assert _output_files(result) == ["file_a", "file_b"]
    assert _output_files(NS(type="web_search_tool_result", content=[NS(url="u", title="t")])) == []


def test_files_made_on_the_providers_servers_are_saved_to_the_workspace(tmp_path):
    use, result = code_run("python3 make.py && cp deck.pptx $OUTPUT_DIR/", "slides: 15\n", ["file_1"])
    client = with_files(FakeClient([message(use, result, text("Your deck is ready."))]),
                        {"file_1": ("deck.pptx", b"PK-deck")})
    events = []
    loop = run_loop(client, "claude-opus-5-5", "sys", [{"role": "user", "content": "make a deck"}],
                    ToolRegistry(server_tools=WEB_TOOLS), lambda kind, **d: events.append((kind, d)),
                    save_file=lambda name, data: save_to_workspace(tmp_path, name, data))
    assert (tmp_path / "deck.pptx").read_bytes() == b"PK-deck"
    assert loop.files == ["deck.pptx"]
    assert any(k == "file" and d["ok"] and d["path"] == "deck.pptx" for k, d in events)
    # the code run is recorded with what it printed, not only web searches
    assert loop.server_calls == [{"tool": "bash_code_execution",
                                  "args": {"command": "python3 make.py && cp deck.pptx $OUTPUT_DIR/"},
                                  "output": "slides: 15\n"}]


def test_a_failed_download_is_traced_and_the_turn_goes_on(tmp_path):
    use, result = code_run("cp x $OUTPUT_DIR/", "", ["file_gone"])
    client = FakeClient([message(use, result, text("done"))])
    client.files = NS(retrieve_metadata=lambda fid: (_ for _ in ()).throw(RuntimeError("404")))
    events = []
    loop = run_loop(client, "claude-opus-5-5", "sys", [{"role": "user", "content": "x"}],
                    ToolRegistry(), lambda kind, **d: events.append((kind, d)),
                    save_file=lambda name, data: save_to_workspace(tmp_path, name, data))
    assert loop.reply == "done" and loop.files == []
    assert any(k == "file" and not d["ok"] and "404" in d["error"] for k, d in events)


def test_saving_never_overwrites_and_never_leaves_the_workspace(tmp_path):
    (tmp_path / "deck.pptx").write_bytes(b"old")
    assert save_to_workspace(tmp_path, "deck.pptx", b"new") == "deck (2).pptx"
    assert save_to_workspace(tmp_path, "../../etc/evil.pptx", b"x") == "evil.pptx"
    assert (tmp_path / "deck.pptx").read_bytes() == b"old"
    assert not (tmp_path.parent / "evil.pptx").exists()


def test_history_note_says_what_ran_on_the_providers_servers_and_what_was_saved():
    note = action_note(
        [{"tool": "run_code", "args": {"code": "import pptx"}, "output": "stdout: False", "is_error": False}],
        [{"tool": "web_search", "args": {"query": "IMG interview"}},
         {"tool": "bash_code_execution", "args": {"command": "python3 make.py"}, "output": "slides: 15"}],
        files=["deck.pptx"])
    assert "web_search('IMG interview')" in note
    assert "bash_code_execution('python3 make.py') [on the provider's servers] -> slides: 15" in note
    assert "files saved to the user's workspace: deck.pptx" in note


def test_eval_checks_read_office_files_and_count_slides(tmp_path):
    import zipfile
    from ops.evals import case_checks
    with zipfile.ZipFile(tmp_path / "deck.pptx", "w") as z:
        for n, word in enumerate(["Title", "Monday", "Friday"], 1):
            z.writestr(f"ppt/slides/slide{n}.xml", f"<p:sld><a:t>{word}</a:t></p:sld>")
    (tmp_path / "fake.pptx").write_text("not a zip")
    checks = case_checks({"files": {"deck.pptx": ["monday"]}, "slides": {"deck.pptx": 3, "fake.pptx": 1}},
                         {"reply": "", "tool_calls": []}, [], workspace=tmp_path)
    assert checks["deck.pptx has 'monday'"] and checks["deck.pptx has 3 slides"]
    assert checks["fake.pptx has 1 slides"] is False


def test_a_file_check_can_accept_either_of_two_texts(tmp_path):
    from ops.evals import case_checks
    (tmp_path / "t.txt").write_text("Total =SUM(B2:B4)")
    checks = case_checks({"files": {"t.txt": ["2850||sum(", "2850||9999"]}}, {"reply": "", "tool_calls": []}, [],
                         workspace=tmp_path)
    assert checks["t.txt has '2850||sum('"] and not checks["t.txt has '2850||9999'"]
