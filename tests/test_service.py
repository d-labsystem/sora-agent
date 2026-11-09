"""The launchd service, with launchctl replaced by a recorder: nothing is installed."""

import io
import plistlib
import subprocess
from pathlib import Path

import pytest

from harness.cli import main as cli
from ops import service


class FakeLaunchctl:
    def __init__(self, print_output="", fail=()):
        self.calls, self.print_output, self.fail = [], print_output, set(fail)

    def __call__(self, cmd):
        self.calls.append(cmd)
        verb = cmd[1]
        if verb in self.fail:
            return subprocess.CompletedProcess(cmd, 5, "", f"{verb} error")
        out = self.print_output if verb == "print" else ""
        return subprocess.CompletedProcess(cmd, 0 if verb != "print" or out else 113, out, "")


def test_plist_contents(tmp_path):
    plist = plistlib.loads(service.render(tmp_path, python="/venv/bin/python", project=Path("/p")).encode())
    assert plist["Label"] == "com.sora.agent"
    assert plist["ProgramArguments"] == ["/venv/bin/python", "/p/main.py", "serve"]
    assert plist["RunAtLoad"] and plist["KeepAlive"] and plist["ThrottleInterval"] == 10
    assert plist["EnvironmentVariables"]["SORA_HOME"] == str(tmp_path)
    assert plist["StandardErrorPath"] == str(tmp_path / "logs" / "service.err.log")
    assert "ANTHROPIC_API_KEY" not in service.render(tmp_path)      # the key stays in .env


def test_install_writes_and_bootstraps(tmp_path):
    run, target = FakeLaunchctl(), tmp_path / "LaunchAgents" / "com.sora.agent.plist"
    notes = service.install(tmp_path / "home", run, target)
    assert target.exists() and (tmp_path / "home" / "logs").is_dir()
    assert [c[1] for c in run.calls] == ["bootstrap"]
    assert any("Installed" in n for n in notes)
    service.install(tmp_path / "home", run, target)                  # reinstall stops the old one first
    assert [c[1] for c in run.calls] == ["bootstrap", "bootout", "bootstrap"]


def test_install_failure_is_reported(tmp_path):
    with pytest.raises(RuntimeError, match="bootstrap error"):
        service.install(tmp_path, FakeLaunchctl(fail={"bootstrap"}), tmp_path / "x.plist")


def test_uninstall(tmp_path):
    target = tmp_path / "x.plist"
    target.write_text("x")
    run = FakeLaunchctl()
    assert "removed" in service.uninstall(run, target)[0]
    assert not target.exists() and run.calls[0][1] == "bootout"
    assert service.uninstall(run, target) == ["The service was not installed."]


def test_status_reads_launchctl_and_probes(tmp_path):
    target = tmp_path / "x.plist"
    target.write_text("x")
    run = FakeLaunchctl(print_output="state = running\n\tpid = 4242\n\tlast exit code = 0\n")
    s = service.status(tmp_path, 8765, run, target, probe=lambda url: url.endswith("/api/status"))
    assert s == {**s, "installed": True, "loaded": True, "pid": 4242, "last_exit": "0", "answering": True}
    down = service.status(tmp_path, 8765, FakeLaunchctl(), tmp_path / "none", probe=lambda u: False)
    assert not down["installed"] and not down["loaded"] and not down["answering"]


def test_privacy_warning():
    assert "Desktop" in service.privacy_warning(Path.home() / "Desktop" / "sora")
    assert service.privacy_warning(Path.home() / "code" / "sora") is None
    assert service.privacy_warning(Path("/opt/sora")) is None


def test_cli_prints_plist_without_installing(tmp_path, monkeypatch):
    monkeypatch.setenv("SORA_HOME", str(tmp_path))
    out = io.StringIO()
    assert cli(["service", "plist"], out=out) == 0
    assert "<string>com.sora.agent</string>" in out.getvalue()
