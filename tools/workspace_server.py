"""The workspace: an MCP server that gives the agent one folder to work in.

The agent can list, find, read, search, write and precisely edit files, but
only inside the workspace folder (~/.sora/workspace by default). It's how
sora drafts notes, keeps a to-do list, or saves a research brief for you.

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

Changing files safely. Rewriting a whole file to change one line costs
tokens and risks silently altering lines the model never meant to touch.
So:
  edit_file     replaces an exact piece of text, and refuses when it's not
                found or found more than once: no guessing which one
  read first    editing or overwriting an existing file is refused unless
                it was read in this session and hasn't changed since (its
                modification time and size are remembered at each read)
  history       before every change the previous version goes to
                .history/ (the last 10 per file); undo_change restores it

Run: python tools/workspace_server.py /path/to/workspace
"""

from __future__ import annotations

import fnmatch
import re
import shutil
import sys
import time
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

# What each tool does, for permission rules (tools/permissions.py): reading only,
# or changing things here that can be undone and notify no one.
READ = ToolAnnotations(read_only_hint=True)
LOCAL = ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=False)

MAX_READ = 100_000       # characters returned by one read
MAX_LINES = 2_000        # lines returned by one read
MAX_WRITE = 1_000_000    # characters written by one call
MAX_LIST = 500           # entries listed or found
MAX_HITS = 100           # search results
HISTORY = ".history"     # previous versions; hidden from listing and search
KEEP = 10                # versions kept per file
CONTEXT = 3              # lines shown around an edit


class Workspace:
    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.seen: dict[Path, tuple[int, int]] = {}      # file -> (mtime_ns, size) when last read or written

    # ---- paths

    def resolve(self, path: str) -> Path:
        if "\x00" in path:
            raise ToolError("path contains a null byte")
        target = (self.root / path.strip().lstrip("/")).resolve()
        if target != self.root and self.root not in target.parents:
            raise ToolError(f"{path!r} is outside the workspace; use a path relative to it")
        if (self.root / HISTORY) in target.parents or target == self.root / HISTORY:
            raise ToolError("the .history folder is managed by undo_change")
        return target

    def show(self, path: Path) -> str:
        return str(path.relative_to(self.root)) or "."

    def _files(self) -> list[Path]:
        """Every regular file inside the workspace, excluding history and
        anything a symlink would take outside."""
        history = self.root / HISTORY
        return [p for p in sorted(self.root.rglob("*"))
                if p.is_file() and self.root in p.resolve().parents and history not in p.parents]

    def _text(self, file: Path, path: str) -> str:
        if not file.is_file():
            raise ToolError(f"{path!r} does not exist. Use list_files or find_files to see what's there.")
        try:
            return file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise ToolError(f"{path!r} is not a text file") from None

    # ---- read-before-change and history

    @staticmethod
    def _stamp(file: Path) -> tuple[int, int]:
        st = file.stat()
        return st.st_mtime_ns, st.st_size

    def _check_fresh(self, file: Path) -> None:
        if file not in self.seen:
            raise ToolError(f"Read {self.show(file)!r} first (read_file), then change it: editing a file you "
                            "haven't read risks overwriting what's in it.")
        if self.seen[file] != self._stamp(file):
            raise ToolError(f"{self.show(file)!r} changed since you read it. Read it again, then change it.")

    def _save_version(self, file: Path) -> None:
        folder = self.root / HISTORY / self.show(file)
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, folder / f"{time.time_ns()}")
        for old in sorted(folder.iterdir())[:-KEEP]:
            old.unlink()

    def _written(self, file: Path) -> None:
        self.seen[file] = self._stamp(file)       # our own write is a fresh view of the file

    # ---- tools

    def list_files(self, path: str = ".") -> str:
        folder = self.resolve(path)
        if not folder.is_dir():
            raise ToolError(f"{path!r} is not a folder")
        history = self.root / HISTORY
        entries = [p for p in sorted(folder.rglob("*"))
                   if self.root in p.resolve().parents and p != history and history not in p.parents]
        lines = [f"{self.show(p)}/" if p.is_dir() else f"{self.show(p)}  ({p.stat().st_size} bytes)"
                 for p in entries[:MAX_LIST]]
        if len(entries) > MAX_LIST:
            lines.append(f"... and {len(entries) - MAX_LIST} more")
        return "\n".join(lines) or "(empty folder)"

    def find_files(self, pattern: str = "**/*") -> str:
        """Files whose workspace path matches a glob, newest first."""
        pattern = pattern.strip().lstrip("/") or "**/*"
        hits = [p for p in self._files()
                if fnmatch.fnmatch(self.show(p), pattern) or fnmatch.fnmatch(p.name, pattern)]
        hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        lines = [f"{self.show(p)}  ({p.stat().st_size} bytes)" for p in hits[:MAX_LIST]]
        return "\n".join(lines) + (f"\n... and {len(hits) - MAX_LIST} more" if len(hits) > MAX_LIST else "") \
            if hits else f"No file matches {pattern!r}."

    def read_file(self, path: str, start_line: int = 1, lines: int = 0) -> str:
        """Lines with numbers ("  12\\ttext"), from start_line, `lines` of them (0 = to the end)."""
        file = self.resolve(path)
        text = self._text(file, path)
        self.seen[file] = self._stamp(file)
        all_lines = text.splitlines()
        if not all_lines:
            return f"({path} is empty)"
        start = max(1, start_line)
        if start > len(all_lines):
            raise ToolError(f"{path!r} has only {len(all_lines)} lines")
        stop = len(all_lines) if lines <= 0 else min(len(all_lines), start - 1 + lines)
        stop = min(stop, start - 1 + MAX_LINES)
        width = len(str(stop))
        out, size = [], 0
        for n in range(start, stop + 1):
            line = f"{n:>{width}}\t{all_lines[n - 1]}"
            size += len(line) + 1
            if size > MAX_READ:
                stop = n - 1
                break
            out.append(line)
        if stop < len(all_lines):
            out.append(f"[lines {stop + 1}-{len(all_lines)} not shown: read again with start_line={stop + 1}]")
        return "\n".join(out)

    def write_file(self, path: str, content: str, append: bool = False) -> str:
        if len(content) > MAX_WRITE:
            raise ToolError(f"content is over {MAX_WRITE} characters; split it into parts")
        file = self.resolve(path)
        if file == self.root or file.is_dir():
            raise ToolError(f"{path!r} is a folder")
        existed = file.is_file()
        if existed and not append:
            self._check_fresh(file)                # overwriting: must have seen what's being replaced
            self._save_version(file)
        file.parent.mkdir(parents=True, exist_ok=True)
        with file.open("a" if append else "w", encoding="utf-8") as f:
            f.write(content)
        self._written(file)
        verb = "Appended to" if append else "Overwrote" if existed else "Wrote"
        return f"{verb} {self.show(file)} ({file.stat().st_size} bytes)"

    def edit_file(self, path: str, old: str, new: str, replace_all: bool = False) -> str:
        """Replace exact text. Unique unless replace_all; shows the result around the change."""
        if old == "":
            raise ToolError("old is empty: give the exact text to replace (to add text, include a nearby line)")
        if old == new:
            raise ToolError("old and new are the same")
        file = self.resolve(path)
        text = self._text(file, path)
        self._check_fresh(file)
        count = text.count(old)
        if count == 0:
            # The usual cause is whitespace (a stray space, tabs vs spaces, a different line
            # break), so look for the first line again with all whitespace removed.
            squash = lambda t: re.sub(r"\s+", "", t).lower()
            key = squash(old.strip().splitlines()[0]) if old.strip() else ""
            near = [n for n, line in enumerate(text.splitlines(), 1) if key and key in squash(line)]
            hint = f" Similar text is on line(s) {', '.join(map(str, near[:5]))}: check spaces and line breaks." \
                if near else " Read the file again and copy the text exactly (without the line numbers)."
            raise ToolError(f"The text to replace isn't in {path!r}.{hint}")
        if count > 1 and not replace_all:
            at = [text[:m.start()].count("\n") + 1 for m in re.finditer(re.escape(old), text)]
            raise ToolError(f"The text appears {count} times (lines {', '.join(map(str, at[:10]))}). Include more "
                            "surrounding text so it's unique, or set replace_all to change every one.")
        self._save_version(file)
        first = text.index(old)
        updated = text.replace(old, new) if replace_all else text.replace(old, new, 1)
        file.write_text(updated, encoding="utf-8")
        self._written(file)
        line_no = updated[:first].count("\n") + 1
        all_lines = updated.splitlines()
        lo = max(1, line_no - CONTEXT)
        hi = min(len(all_lines), line_no + new.count("\n") + CONTEXT)
        snippet = "\n".join(f"{n}\t{all_lines[n - 1]}" for n in range(lo, hi + 1))
        return (f"Edited {self.show(file)}: {count if replace_all else 1} replacement(s). "
                f"Lines {lo}-{hi} now:\n{snippet}")

    def grep(self, pattern: str, glob: str = "", ignore_case: bool = False, context: int = 0) -> str:
        """Regex search across text files; `file:line: text`, with context lines marked by '-'."""
        if not pattern:
            raise ToolError("pattern is empty")
        try:
            regex = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
        except re.error as exc:
            raise ToolError(f"not a valid regular expression ({exc}); escape special characters like ( . *") from None
        context = max(0, min(context, 5))
        out, hits = [], 0
        for file in self._files():
            name = self.show(file)
            if glob and not (fnmatch.fnmatch(name, glob) or fnmatch.fnmatch(file.name, glob)):
                continue
            try:
                lines = file.read_text(encoding="utf-8").splitlines()
            except (UnicodeDecodeError, OSError):
                continue                                     # binary files are skipped
            for n, line in enumerate(lines, 1):
                if not regex.search(line):
                    continue
                hits += 1
                if hits > MAX_HITS:
                    return "\n".join(out) + f"\n[more than {MAX_HITS} matches: narrow the pattern or glob]"
                for c in range(max(1, n - context), n):
                    out.append(f"{name}-{c}- {lines[c - 1].strip()[:200]}")
                out.append(f"{name}:{n}: {line.strip()[:200]}")
                for c in range(n + 1, min(len(lines), n + context) + 1):
                    out.append(f"{name}-{c}- {lines[c - 1].strip()[:200]}")
        return "\n".join(out) or f"No line matches {pattern!r}" + (f" in files matching {glob!r}." if glob else ".")

    def undo_change(self, path: str) -> str:
        """Put back the version from before the last edit or overwrite."""
        file = self.resolve(path)
        folder = self.root / HISTORY / self.show(file)
        versions = sorted(folder.iterdir()) if folder.is_dir() else []
        if not versions:
            raise ToolError(f"No earlier version of {path!r} is saved.")
        previous = versions[-1]
        file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(previous, file)
        previous.unlink()
        self._written(file)
        return f"Restored {self.show(file)} to its version from before the last change " \
               f"({len(versions) - 1} older version(s) left)."


BASIC = {"list_files", "read_file", "write_file", "grep"}   # the tool set before precise editing


def build(root: Path, basic: bool = False) -> MCPServer:
    """`basic` offers only the original tools (no edit, find or undo), for
    comparing old and new in evals (SORA_WORKSPACE_TOOLS=basic)."""
    ws = Workspace(root)
    server = MCPServer("workspace", instructions=f"Files in the user's workspace folder {ws.root}.")

    @server.tool(annotations=READ)
    def list_files(path: str = ".") -> str:
        """List files and folders in the workspace (recursively), with sizes."""
        return ws.list_files(path)

    @server.tool(annotations=READ)
    def find_files(pattern: str = "**/*") -> str:
        """Find files by name pattern, newest first: '*.csv', 'notes/**/*.md', 'budget*'."""
        return ws.find_files(pattern)

    @server.tool(annotations=READ)
    def read_file(path: str, start_line: int = 1, lines: int = 0) -> str:
        """Read a text file, with line numbers ('12<TAB>text'). For long files give start_line and
        lines. The numbers are not part of the file: leave them out when quoting text to edit."""
        return ws.read_file(path, start_line, lines)

    @server.tool(annotations=LOCAL)
    def edit_file(path: str, old: str, new: str, replace_all: bool = False) -> str:
        """Change part of a file: replace the exact text `old` with `new`. Read the file first.
        `old` must appear exactly once (add surrounding text to make it unique) unless replace_all.
        Prefer this to rewriting a whole file: it's cheaper and can't disturb other lines."""
        return ws.edit_file(path, old, new, replace_all)

    @server.tool(annotations=LOCAL)
    def write_file(path: str, content: str, append: bool = False) -> str:
        """Create a file, or replace an existing one entirely (read it first). append=true adds to
        the end instead, e.g. for a running log. To change part of a file, use edit_file."""
        return ws.write_file(path, content, append)

    @server.tool(annotations=READ)
    def grep(pattern: str, glob: str = "", ignore_case: bool = False, context: int = 0) -> str:
        """Search file contents with a regular expression. Optional glob limits which files
        ('*.md'), context adds lines around each match (up to 5). Results: file:line: text."""
        return ws.grep(pattern, glob, ignore_case, context)

    @server.tool(annotations=LOCAL)
    def undo_change(path: str) -> str:
        """Restore a file to how it was before its last edit or overwrite (repeatable, 10 deep)."""
        return ws.undo_change(path)

    if basic:
        for name in ("edit_file", "find_files", "undo_change"):
            server.remove_tool(name)
    return server


def config(home: Path, basic: bool = False) -> dict:
    """The mcp.json entry for this server, used when the user hasn't configured one."""
    return {"command": sys.executable,
            "args": [str(Path(__file__).resolve()), str(home / "workspace"), *(["--basic"] if basic else [])]}


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit("usage: workspace_server.py WORKSPACE_FOLDER [--basic]")
    build(Path(sys.argv[1]), basic=sys.argv[2:] == ["--basic"]).run()
