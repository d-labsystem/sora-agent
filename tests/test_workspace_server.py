"""The workspace sandbox, directly and through the MCP client."""

import os

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from tools.mcp_client import MCPManager
from tools.registry import ToolRegistry
from tools.workspace_server import Workspace, config


@pytest.fixture
def ws(tmp_path):
    return Workspace(tmp_path / "workspace")


@pytest.mark.parametrize("path", ["../secret", "notes/../../secret", "/etc/passwd/../..", "a\x00b"])
def test_paths_cannot_escape(ws, path):
    with pytest.raises(ToolError):
        ws.write_file(path, "x")


def test_absolute_paths_are_treated_as_relative(ws):
    ws.write_file("/notes/a.md", "hi")
    assert (ws.root / "notes" / "a.md").read_text() == "hi"


def test_symlink_out_is_blocked(ws, tmp_path):
    (tmp_path / "outside.txt").write_text("secret")
    os.symlink(tmp_path / "outside.txt", ws.root / "link.txt")
    with pytest.raises(ToolError, match="outside"):
        ws.read_file("link.txt")
    assert "link.txt" not in ws.list_files()
    assert ws.grep("secret") == "No line matches 'secret'."


def test_write_append_read_list_search(ws):
    assert ws.write_file("todo/list.md", "- buy milk\n").startswith("Wrote todo/list.md")
    ws.write_file("todo/list.md", "- call Priya\n", append=True)
    assert ws.read_file("todo/list.md") == "1\t- buy milk\n2\t- call Priya"
    assert "todo/list.md  (" in ws.list_files() and "todo/" in ws.list_files()
    assert ws.grep("PRIYA", ignore_case=True) == "todo/list.md:2: - call Priya"


def test_long_files_are_read_by_line_range(ws):
    ws.write_file("big.txt", "\n".join(f"line {i}" for i in range(1, 31)))
    assert ws.read_file("big.txt", 5, 2) == "5\tline 5\n6\tline 6\n[lines 7-30 not shown: read again with start_line=7]"
    assert ws.read_file("big.txt", 29).endswith("30\tline 30")
    with pytest.raises(ToolError, match="only 30 lines"):
        ws.read_file("big.txt", 31)


def test_helpful_errors(ws):
    with pytest.raises(ToolError, match="list_files"):
        ws.read_file("missing.md")
    with pytest.raises(ToolError, match="folder"):
        ws.write_file(".", "x")
    (ws.root / "img.bin").write_bytes(b"\xff\xfe\x00")
    with pytest.raises(ToolError, match="not a text file"):
        ws.read_file("img.bin")


def test_through_mcp(tmp_path):
    manager = MCPManager({"workspace": config(tmp_path)}, tmp_path / "logs")
    manager.start(timeout=20)
    try:
        registry = ToolRegistry()
        assert sorted(manager.register(registry)) == [
            "workspace__edit_file", "workspace__find_files", "workspace__grep", "workspace__list_files",
            "workspace__read_file", "workspace__undo_change", "workspace__write_file"]
        assert not registry.execute("workspace__write_file", {"path": "a.md", "content": "hello"})[1]
        assert registry.execute("workspace__read_file", {"path": "a.md"}) == ("1\thello", False)
        assert not registry.execute("workspace__edit_file", {"path": "a.md", "old": "hello", "new": "hi"})[1]
        output, error = registry.execute("workspace__read_file", {"path": "../../x"})
        assert error and "outside the workspace" in output
        assert (tmp_path / "workspace" / "a.md").read_text() == "hi"
    finally:
        manager.close()



# ---- precise editing

NOTE = "# Trip\nLeave on 20 Dec.\nTrain at 08:15.\nReturn on 23 Dec.\nTrain at 08:15.\n"


def seeded(ws, text=NOTE, name="trip.md"):
    ws.write_file(name, text)
    return name


def test_edit_replaces_exactly_one_occurrence_and_nothing_else(ws):
    seeded(ws)
    ws.read_file("trip.md")
    out = ws.edit_file("trip.md", "Leave on 20 Dec.", "Leave on 21 Dec.")
    assert (ws.root / "trip.md").read_text() == NOTE.replace("20 Dec", "21 Dec")
    assert out.startswith("Edited trip.md: 1 replacement(s).") and "2\tLeave on 21 Dec." in out


def test_ambiguous_or_missing_text_is_refused_with_a_hint(ws):
    seeded(ws)
    ws.read_file("trip.md")
    with pytest.raises(ToolError, match="appears 2 times \\(lines 3, 5\\)"):
        ws.edit_file("trip.md", "Train at 08:15.", "Train at 09:00.")
    assert (ws.root / "trip.md").read_text() == NOTE                       # untouched
    ws.edit_file("trip.md", "Return on 23 Dec.\nTrain at 08:15.", "Return on 23 Dec.\nTrain at 18:40.")
    assert "Train at 18:40." in (ws.root / "trip.md").read_text()
    with pytest.raises(ToolError, match="line\\(s\\) 2"):
        ws.edit_file("trip.md", "Leave on 20 Dec .", "x")                   # near miss: points at line 2
    ws.edit_file("trip.md", "Train at", "Bus at", replace_all=True)
    assert (ws.root / "trip.md").read_text().count("Bus at") == 2


def test_changes_need_a_fresh_read(ws):
    seeded(ws, name="list.md")
    fresh = Workspace(ws.root)                                              # a new session: hasn't read it
    with pytest.raises(ToolError, match="Read 'list.md' first"):
        fresh.edit_file("list.md", "Trip", "Holiday")
    with pytest.raises(ToolError, match="Read 'list.md' first"):
        fresh.write_file("list.md", "replaced")
    fresh.write_file("list.md", "- more\n", append=True)                   # appending is allowed
    fresh.read_file("list.md")
    import os, time
    time.sleep(0.01)
    (ws.root / "list.md").write_text("someone else changed this\n")        # changed behind its back
    os.utime(ws.root / "list.md")
    with pytest.raises(ToolError, match="changed since you read it"):
        fresh.edit_file("list.md", "someone", "no one")
    assert fresh.write_file("new.md", "brand new").startswith("Wrote new.md")  # new files need no read


def test_every_change_can_be_undone(ws):
    seeded(ws)
    ws.read_file("trip.md")
    ws.edit_file("trip.md", "20 Dec", "21 Dec")
    ws.edit_file("trip.md", "21 Dec", "22 Dec")
    ws.write_file("trip.md", "gone")
    assert ws.undo_change("trip.md").endswith("(2 older version(s) left).")
    assert "22 Dec" in (ws.root / "trip.md").read_text()
    ws.undo_change("trip.md")
    ws.undo_change("trip.md")
    assert (ws.root / "trip.md").read_text() == NOTE
    with pytest.raises(ToolError, match="No earlier version"):
        ws.undo_change("trip.md")
    assert ".history" not in ws.list_files() and ws.grep("Dec").count("trip.md") == 2
    with pytest.raises(ToolError, match="managed by undo_change"):
        ws.read_file(".history/trip.md")


def test_history_keeps_the_last_ten(ws):
    seeded(ws, "v0", "f.txt")
    ws.read_file("f.txt")
    for i in range(1, 15):
        ws.write_file("f.txt", f"v{i}")
    assert len(list((ws.root / ".history" / "f.txt").iterdir())) == 10


def test_grep_regex_globs_context_and_bad_patterns(ws):
    ws.write_file("a.md", "intro\nTODO: call Priya\noutro\n")
    ws.write_file("b.txt", "todo: pay rent\n")
    ws.write_file("c.md", "nothing\n")
    (ws.root / "pic.png").write_bytes(b"\x89PNG\xff TODO")
    assert ws.grep(r"todo:", ignore_case=True) == "a.md:2: TODO: call Priya\nb.txt:1: todo: pay rent"
    assert ws.grep("TODO", glob="*.md", context=1) == "a.md-1- intro\na.md:2: TODO: call Priya\na.md-3- outro"
    with pytest.raises(ToolError, match="not a valid regular expression"):
        ws.grep("call (Priya")
    assert "No line matches" in ws.grep("zebra", glob="*.md")


def test_find_files_newest_first(ws):
    import os
    for name, age in (("old.csv", 300), ("notes/new.csv", 10), ("x.md", 5)):
        ws.write_file(name, "1")
        t = __import__("time").time() - age
        os.utime(ws.root / name, (t, t))
    assert ws.find_files("*.csv").splitlines()[0].startswith("notes/new.csv")
    assert ws.find_files("notes/**").startswith("notes/new.csv") and "No file matches" in ws.find_files("*.pdf")


@pytest.mark.parametrize("call", [
    lambda ws: ws.edit_file("../outside.md", "a", "b"),
    lambda ws: ws.read_file("../../etc/passwd"),
    lambda ws: ws.undo_change("../x"),
    lambda ws: ws.write_file(".history/trip.md/1", "forged version"),
])
def test_new_tools_cannot_escape_or_forge_history(ws, call):
    with pytest.raises(ToolError):
        call(ws)
