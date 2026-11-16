"""run_code: Python that the model writes, run in a locked-down container.

Why a sandbox. The model writes the code, and what it read before (a web
page, a file someone sent) can steer what it writes. So the code runs where
it can't do harm, and every lock is set on each run, not trusted to the image:

    --network none         no internet: nothing can be sent out or fetched in
    -v workspace only      the only folder of your Mac that exists inside is
                           ~/.sora/workspace; .env, SSH keys, the rest: absent
    --read-only + /tmp     the system can't be changed; scratch space is a
                           small in-memory /tmp
    --user 1000            not root, --cap-drop ALL, no-new-privileges
    limits                 1 CPU, 1 GB memory (no swap), 128 processes,
                           a wall-clock timeout after which the container is
                           killed (killing `docker run` alone wouldn't stop it)
    output cap             what the code prints is clipped before the model
                           reads it

Each call is a fresh container: variables don't survive between calls, files
in the workspace do. Images the code saves (charts) are reported, so the
dashboard can show them.

This is sora's own capability (Docker on your Mac, any model). Anthropic's
server-side code execution is a separate option (SORA_CODE_EXECUTION).
"""

from __future__ import annotations

import shutil
import subprocess
import uuid
from pathlib import Path

from tools.registry import Tool, ToolError

IMAGE = "sora-sandbox:1"
TIMEOUT = 60
MAX_OUTPUT = 20_000
MAX_CODE = 100_000
PICTURES = {".png", ".jpg", ".jpeg", ".svg", ".gif", ".webp"}


def docker_command(workspace: Path, name: str, image: str = IMAGE) -> list[str]:
    """The exact `docker run` used for every call (tests assert each lock)."""
    return ["docker", "run", "--rm", "-i", "--name", name,
            "--network", "none",
            "--read-only", "--tmpfs", "/tmp:rw,nosuid,nodev,size=256m",
            "--user", "1000:1000", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--memory", "1g", "--memory-swap", "1g", "--cpus", "1", "--pids-limit", "128",
            "-v", f"{workspace.resolve()}:/workspace:rw", "-w", "/workspace",
            image, "python", "-I", "-"]           # -I: ignore env vars and the user site; code on stdin


def clip(text: str, limit: int = MAX_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n… [{len(text) - limit} characters cut] …\n{text[-half:]}"


def pictures(folder: Path) -> dict[str, float]:
    if not folder.is_dir():
        return {}
    return {str(p.relative_to(folder)): p.stat().st_mtime for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in PICTURES}


class Sandbox:
    def __init__(self, workspace: Path, image: str = IMAGE, timeout: int = TIMEOUT, run=subprocess.run):
        self.workspace, self.image, self.timeout, self.run = workspace, image, timeout, run
        self.last_pictures: list[str] = []

    def available(self) -> str | None:
        """Why the sandbox can't run, or None."""
        if not shutil.which("docker"):
            return "Docker isn't installed"
        done = self.run(["docker", "image", "inspect", self.image], capture_output=True, text=True, timeout=20)
        if done.returncode != 0:
            daemon = "Cannot connect" in (done.stderr or "") or "daemon" in (done.stderr or "").lower()
            return ("Docker isn't running (start Docker Desktop)" if daemon
                    else f"the sandbox image isn't built (run: uv run main.py sandbox build)")
        return None

    def run_code(self, code: str, language: str = "python") -> str:
        if language.lower() not in ("python", "py"):
            raise ToolError("only Python runs in the sandbox for now")
        if not code.strip():
            raise ToolError("code is empty")
        if len(code) > MAX_CODE:
            raise ToolError(f"code is over {MAX_CODE} characters; write it to a file in parts instead")
        problem = self.available()
        if problem:
            raise ToolError(f"The code sandbox can't run: {problem}. Work it out step by step instead, "
                            "and tell the user.")
        self.workspace.mkdir(parents=True, exist_ok=True)
        before = pictures(self.workspace)
        name = f"sora-sbx-{uuid.uuid4().hex[:10]}"
        try:
            done = self.run(docker_command(self.workspace, name, self.image), input=code, capture_output=True,
                            text=True, timeout=self.timeout)
        except subprocess.TimeoutExpired:
            self.run(["docker", "kill", name], capture_output=True, text=True, timeout=20)
            raise ToolError(f"Stopped after {self.timeout} seconds (the time limit). Make the code faster, "
                            "or split the work into smaller runs.") from None
        after = pictures(self.workspace)
        self.last_pictures = sorted(p for p, t in after.items() if before.get(p) != t)
        parts = [f"exit code {done.returncode}" + (" (out of memory: killed)" if done.returncode == 137 else "")]
        if done.stdout:
            parts.append("stdout:\n" + clip(done.stdout))
        if done.stderr:
            parts.append("stderr:\n" + clip(done.stderr, MAX_OUTPUT // 2))
        if self.last_pictures:
            parts.append("images saved in the workspace (shown to the user): " + ", ".join(self.last_pictures))
        if not done.stdout and not done.stderr and done.returncode == 0:
            parts.append("(no output: print() what you need to see)")
        return "\n".join(parts)

    def tool(self) -> Tool:
        return Tool(
            "run_code",
            "Run Python in a sandbox and get what it prints. Use it for exact arithmetic, statistics, dates, "
            "parsing or transforming files, and charts (matplotlib: save to a .png and it is shown to the "
            "user). The working folder is the user's workspace: read and write files there. pandas, numpy, "
            "openpyxl and matplotlib are installed. There is no internet. Each call starts fresh: variables "
            "don't carry over, so keep what you need in files. print() results; a bare expression shows nothing.",
            {"type": "object", "properties": {"code": {"type": "string"},
                                              "language": {"type": "string", "enum": ["python"]}},
             "required": ["code"]},
            self.run_code)


def build_image(out=print) -> int:
    folder = Path(__file__).resolve().parent.parent / "sandbox"
    done = subprocess.run(["docker", "build", "-t", IMAGE, str(folder)], text=True)
    out(f"built {IMAGE}" if done.returncode == 0 else "build failed (is Docker Desktop running?)")
    return done.returncode
