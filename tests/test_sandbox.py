"""The code sandbox: its locks (offline), and real escape attempts (live, with Docker)."""

import shutil
import subprocess
from types import SimpleNamespace as NS

import pytest

from tools.registry import ToolError
from tools.sandbox import IMAGE, Sandbox, clip, docker_command


def test_every_run_carries_every_lock(tmp_path):
    cmd = docker_command(tmp_path, "sora-sbx-x")
    flags = " ".join(cmd)
    for lock in ("--network none", "--read-only", "--user 1000:1000", "--cap-drop ALL",
                 "--security-opt no-new-privileges", "--memory 1g", "--memory-swap 1g", "--cpus 1",
                 "--pids-limit 128", "--rm"):
        assert lock in flags, lock
    mounts = [cmd[i + 1] for i, part in enumerate(cmd) if part == "-v"]
    assert mounts == [f"{tmp_path.resolve()}:/workspace:rw"]          # the workspace, and nothing else
    assert cmd[-4:] == [IMAGE, "python", "-I", "-"]                    # code arrives on stdin, never as an argument


class FakeDocker:
    def __init__(self, result=None, image_ok=True, timeout=False):
        self.calls, self.result, self.image_ok, self.timeout = [], result, image_ok, timeout

    def __call__(self, cmd, **kw):
        self.calls.append((cmd, kw))
        if cmd[:3] == ["docker", "image", "inspect"]:
            return NS(returncode=0 if self.image_ok else 1, stdout="", stderr="" if self.image_ok else "No such image")
        if cmd[:2] == ["docker", "kill"]:
            return NS(returncode=0, stdout="", stderr="")
        if self.timeout:
            raise subprocess.TimeoutExpired(cmd, 60)
        return self.result


def box(tmp_path, docker):
    monkey = Sandbox(tmp_path, run=docker)
    return monkey


def test_output_is_reported_and_clipped(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/local/bin/docker")
    docker = FakeDocker(NS(returncode=0, stdout="x" * 50_000, stderr=""))
    out = box(tmp_path, docker).run_code("print('x' * 50000)")
    assert out.startswith("exit code 0") and "characters cut" in out and len(out) < 25_000
    assert docker.calls[-1][1]["input"] == "print('x' * 50000)"
    assert clip("short") == "short"


def test_timeouts_kill_the_container(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/local/bin/docker")
    docker = FakeDocker(timeout=True)
    with pytest.raises(ToolError, match="time limit"):
        box(tmp_path, docker).run_code("while True: pass")
    run_cmd = next(c for c, _ in docker.calls if c[:2] == ["docker", "run"])
    assert docker.calls[-1][0] == ["docker", "kill", run_cmd[run_cmd.index("--name") + 1]]


def test_missing_docker_or_image_is_explained(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _: None)
    with pytest.raises(ToolError, match="isn't installed"):
        box(tmp_path, FakeDocker()).run_code("print(1)")
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/local/bin/docker")
    with pytest.raises(ToolError, match="sandbox build"):
        box(tmp_path, FakeDocker(image_ok=False)).run_code("print(1)")
    with pytest.raises(ToolError, match="only Python"):
        box(tmp_path, FakeDocker()).run_code("ls", "bash")


# ---- live: the real container must refuse every escape

def docker_ready() -> bool:
    if not shutil.which("docker"):
        return False
    return subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True).returncode == 0


live = pytest.mark.skipif(not docker_ready(), reason="needs Docker and the sora-sandbox image")


@live
def test_live_computes_and_writes_to_the_workspace(tmp_path):
    out = Sandbox(tmp_path).run_code(
        "import pandas as pd, matplotlib.pyplot as plt\n"
        "df = pd.DataFrame({'x': [1, 2, 3], 'y': [2, 4, 9]})\n"
        "print(df.y.sum(), round(df.y.mean(), 2))\n"
        "df.plot(x='x', y='y'); plt.savefig('chart.png')\n"
        "open('out.txt', 'w').write('hi')")
    assert "exit code 0" in out and "15 5.0" in out
    assert (tmp_path / "out.txt").read_text() == "hi" and (tmp_path / "chart.png").stat().st_size > 1000
    assert "images saved in the workspace (shown to the user): chart.png" in out


@live
@pytest.mark.parametrize("attack, blocked", [
    ("import urllib.request; urllib.request.urlopen('https://example.com', timeout=5)", "URLError"),
    ("import socket; socket.create_connection(('1.1.1.1', 53), timeout=5)", "OSError"),
    ("import os; print(os.listdir('/Users'))", "FileNotFoundError"),
    ("open('/etc/passwd_copy', 'w').write('x')", "Read-only file system"),
    ("import os; print(os.getuid())", "1000"),
    ("import os; os.setuid(0)", "PermissionError"),
])
def test_live_escapes_fail(tmp_path, attack, blocked):
    out = Sandbox(tmp_path).run_code(attack)
    assert blocked in out, out


@live
def test_live_secrets_on_the_mac_are_invisible(tmp_path):
    # Mac folders don't exist inside, and no secret is in any home or the workspace.
    # (A recursive glob from / also walks /proc and /sys and takes minutes.)
    out = Sandbox(tmp_path).run_code(
        "import os, glob\n"
        "print(sorted(os.listdir('/')))\n"
        "where = ['/workspace', '/home', '/root', '/tmp']\n"
        "print([f for w in where for f in glob.glob(w + '/**/.env', recursive=True) + glob.glob(w + '/**/id_rsa', recursive=True)])")
    roots = out.split("stdout:\n")[1].splitlines()[0]
    assert not any(mac in roots for mac in ("'Users'", "'Volumes'", "'Applications'", "'private'"))
    assert "\n[]" in out


@live
def test_live_a_fork_bomb_is_contained(tmp_path):
    out = Sandbox(tmp_path, timeout=30).run_code(
        "import os\n"
        "n = 0\n"
        "try:\n"
        "    while True:\n"
        "        if os.fork() == 0:\n"
        "            import time; time.sleep(30); os._exit(0)\n"
        "        n += 1\n"
        "except OSError as e:\n"
        "    print('stopped after', n, 'forks:', type(e).__name__)")
    assert "stopped after" in out and int(out.split("stopped after ")[1].split()[0]) < 128


@live
def test_live_runaway_code_is_killed(tmp_path):
    with pytest.raises(ToolError, match="time limit"):
        Sandbox(tmp_path, timeout=5).run_code("while True: pass")
    left = subprocess.run(["docker", "ps", "-q", "--filter", "name=sora-sbx-"], capture_output=True, text=True).stdout
    assert left.strip() == ""                                   # nothing left running



@live
def test_live_code_changes_can_be_undone(tmp_path):
    from tools.workspace_server import Workspace
    (tmp_path / "list.md").write_text("- eggs\n- rice\n")
    (tmp_path / "old.txt").write_text("keep me")
    out = Sandbox(tmp_path).run_code(
        "import os\nopen('list.md', 'w').write('- bread\\n')\nos.remove('old.txt')\nopen('new.txt', 'w').write('n')")
    assert "changed or deleted existing files (undo_change can restore them): list.md, old.txt" in out
    ws = Workspace(tmp_path)
    ws.undo_change("list.md")
    ws.undo_change("old.txt")
    assert (tmp_path / "list.md").read_text() == "- eggs\n- rice\n" and (tmp_path / "old.txt").read_text() == "keep me"
    assert "new.txt" not in out.split("undo_change")[0]          # new files aren't "changed"
