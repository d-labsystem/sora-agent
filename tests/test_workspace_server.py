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
    assert ws.search_files("secret") == "No file contains 'secret'."


def test_write_append_read_list_search(ws):
    assert ws.write_file("todo/list.md", "- buy milk\n").startswith("Wrote todo/list.md")
    ws.write_file("todo/list.md", "- call Priya\n", append=True)
    assert ws.read_file("todo/list.md") == "- buy milk\n- call Priya\n"
    assert "todo/list.md  (" in ws.list_files() and "todo/" in ws.list_files()
    assert ws.search_files("PRIYA") == "todo/list.md:2: - call Priya"


def test_long_files_are_paged(ws, monkeypatch):
    monkeypatch.setattr("tools.workspace_server.MAX_READ", 10)
    ws.write_file("big.txt", "a" * 25)
    first = ws.read_file("big.txt")
    assert first.startswith("a" * 10) and "offset=10" in first
    assert ws.read_file("big.txt", 20) == "a" * 5


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
            "workspace__list_files", "workspace__read_file",
            "workspace__search_files", "workspace__write_file"]
        assert not registry.execute("workspace__write_file", {"path": "a.md", "content": "hello"})[1]
        assert registry.execute("workspace__read_file", {"path": "a.md"}) == ("hello", False)
        output, error = registry.execute("workspace__read_file", {"path": "../../x"})
        assert error and "outside the workspace" in output
        assert (tmp_path / "workspace" / "a.md").read_text() == "hello"
    finally:
        manager.close()
