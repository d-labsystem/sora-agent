"""The workspace: an MCP server that gives the agent one folder to work in.

The agent can list, read, write and search files, but only inside the
workspace folder (~/.sora/workspace by default). It's how sora drafts notes,
keeps a to-do list, or saves a research brief for you to open later.

Why an MCP server, when these could be built-in tools? Three reasons:

  isolation   it runs as a separate process, so a bug here can't corrupt
              the agent's memory, and it could later run in a container
  dogfooding  it exercises our MCP client on every turn, so MCP breakage
              shows up immediately, not when you add your first real server
  reuse       Claude Desktop, or any MCP client, can use the same server

The sandbox. Every path goes through `resolve()`, which turns it into an
absolute path with `..` and symlinks resolved, and must still be inside the
root. This blocks "../../.ssh/id_rsa", absolute paths, and symlinks that
point outside. The file is self-contained (no project imports), so it runs
as a plain script from any working directory.

Run: python tools/workspace_server.py /path/to/workspace
"""

from __future__ import annotations

import sys
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

MAX_READ = 100_000       # characters returned by one read
MAX_WRITE = 1_000_000    # characters written by one call
MAX_LIST = 500           # entries listed
MAX_HITS = 50            # search results


class Workspace:
    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, path: str) -> Path:
        if "\x00" in path:
            raise ToolError("path contains a null byte")
        target = (self.root / path.strip().lstrip("/")).resolve()
        if target != self.root and self.root not in target.parents:
            raise ToolError(f"{path!r} is outside the workspace; use a path relative to it")
        return target

    def show(self, path: Path) -> str:
        return str(path.relative_to(self.root)) or "."

    def list_files(self, path: str = ".") -> str:
        folder = self.resolve(path)
        if not folder.is_dir():
            raise ToolError(f"{path!r} is not a folder")
        entries = sorted(folder.rglob("*"))
        lines = [f"{self.show(p)}/" if p.is_dir() else f"{self.show(p)}  ({p.stat().st_size} bytes)"
                 for p in entries[:MAX_LIST] if self.root in p.resolve().parents]
        if len(entries) > MAX_LIST:
            lines.append(f"... and {len(entries) - MAX_LIST} more")
        return "\n".join(lines) or "(empty folder)"

    def read_file(self, path: str, offset: int = 0) -> str:
        file = self.resolve(path)
        if not file.is_file():
            raise ToolError(f"{path!r} does not exist. Use list_files to see what's there.")
        try:
            text = file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise ToolError(f"{path!r} is not a text file") from None
        chunk = text[offset:offset + MAX_READ]
        rest = len(text) - offset - len(chunk)
        return chunk + (f"\n[{rest} more characters: read again with offset={offset + len(chunk)}]"
                        if rest > 0 else "")

    def write_file(self, path: str, content: str, append: bool = False) -> str:
        if len(content) > MAX_WRITE:
            raise ToolError(f"content is over {MAX_WRITE} characters; split it into parts")
        file = self.resolve(path)
        if file == self.root or file.is_dir():
            raise ToolError(f"{path!r} is a folder")
        file.parent.mkdir(parents=True, exist_ok=True)
        with file.open("a" if append else "w", encoding="utf-8") as f:
            f.write(content)
        return f"{'Appended to' if append else 'Wrote'} {self.show(file)} ({file.stat().st_size} bytes)"

    def search_files(self, text: str) -> str:
        needle = text.lower()
        if not needle.strip():
            raise ToolError("search text is empty")
        hits = []
        for file in sorted(self.root.rglob("*")):
            if not file.is_file() or self.root not in file.resolve().parents:
                continue
            try:
                lines = file.read_text(encoding="utf-8").splitlines()
            except (UnicodeDecodeError, OSError):
                continue
            hits += [f"{self.show(file)}:{n}: {line.strip()[:200]}"
                     for n, line in enumerate(lines, 1) if needle in line.lower()]
            if len(hits) >= MAX_HITS:
                break
        return "\n".join(hits[:MAX_HITS]) or f"No file contains {text!r}."


def build(root: Path) -> MCPServer:
    ws = Workspace(root)
    server = MCPServer("workspace", instructions=f"Files in the user's workspace folder {ws.root}.")

    @server.tool()
    def list_files(path: str = ".") -> str:
        """List files and folders in the workspace (recursively), with sizes."""
        return ws.list_files(path)

    @server.tool()
    def read_file(path: str, offset: int = 0) -> str:
        """Read a text file from the workspace. Long files come in parts: pass the offset given."""
        return ws.read_file(path, offset)

    @server.tool()
    def write_file(path: str, content: str, append: bool = False) -> str:
        """Create or overwrite a text file in the workspace (folders are created).
        Set append=true to add to the end instead, e.g. for a running log or to-do list."""
        return ws.write_file(path, content, append)

    @server.tool()
    def search_files(text: str) -> str:
        """Find lines containing some text (case-insensitive) across workspace files."""
        return ws.search_files(text)

    return server


def config(home: Path) -> dict:
    """The mcp.json entry for this server, used when the user hasn't configured one."""
    return {"command": sys.executable, "args": [str(Path(__file__).resolve()), str(home / "workspace")]}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: workspace_server.py WORKSPACE_FOLDER")
    build(Path(sys.argv[1])).run()
