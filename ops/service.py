"""Run sora as a macOS background service, so the dashboard is always on.

macOS starts and supervises background programs with launchd. You describe
the program in a small XML file (a "property list", .plist) in
~/Library/LaunchAgents, and launchd:

    RunAtLoad     starts it when you log in
    KeepAlive     restarts it if it crashes (ThrottleInterval: not more than
                  once every 10 seconds, so a crash loop can't spin the CPU)
    Std*Path      sends its output to log files in ~/.sora/logs

A LaunchAgent runs as you, only while you're logged in, with no admin rights,
which is all a personal assistant needs. (A LaunchDaemon would run as root at
boot: more power than sora should have.)

Commands, through the CLI:

    python main.py service plist       print the file, change nothing
    python main.py service install     write it and start the service
    python main.py service status      is it loaded, running, answering?
    python main.py service restart     after you change code or .env
    python main.py service uninstall   stop it and remove the file

The API key is NOT put in the plist: the service reads .env at startup like
any other run, so the key stays in one file you control.

macOS privacy note: background programs are not allowed to read the Desktop,
Documents or Downloads folders unless you grant access. If the project lives
there, the service may fail with "Operation not permitted". install() warns
about this; the fix is to move the project (e.g. to ~/code/sora) or to give
the Python interpreter access in System Settings → Privacy & Security.
"""

from __future__ import annotations

import os
import plistlib
import subprocess
import sys
import urllib.request
from collections.abc import Callable
from pathlib import Path

LABEL = "com.sora.agent"
PROJECT = Path(__file__).resolve().parent.parent
PROTECTED = ("Desktop", "Documents", "Downloads")

Runner = Callable[[list[str]], subprocess.CompletedProcess]


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def build_plist(home: Path, python: str = sys.executable, project: Path = PROJECT) -> dict:
    logs = home / "logs"
    return {
        "Label": LABEL,
        "ProgramArguments": [python, str(project / "main.py"), "serve"],
        "WorkingDirectory": str(project),
        "EnvironmentVariables": {
            "SORA_HOME": str(home),
            # launchd starts with a bare PATH; MCP servers launched by name
            # (uvx, npx) need the usual places.
            "PATH": os.pathsep.join([str(Path(python).parent), str(Path.home() / ".local/bin"),
                                     "/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin"]),
            "PYTHONUNBUFFERED": "1",
        },
        "RunAtLoad": True,
        "KeepAlive": True,
        "ThrottleInterval": 10,
        "StandardOutPath": str(logs / "service.out.log"),
        "StandardErrorPath": str(logs / "service.err.log"),
    }


def render(home: Path, **kwargs) -> str:
    return plistlib.dumps(build_plist(home, **kwargs)).decode()


def privacy_warning(project: Path = PROJECT) -> str | None:
    try:
        relative = project.relative_to(Path.home())
    except ValueError:
        return None
    if relative.parts and relative.parts[0] in PROTECTED:
        return (f"The project is in ~/{relative.parts[0]}, which macOS hides from background "
                f"programs. If the service log shows 'Operation not permitted', move the project "
                f"(e.g. to ~/code/sora) and reinstall, or allow {sys.executable} under System "
                f"Settings → Privacy & Security → Files and Folders.")
    return None


def _domain() -> str:
    return f"gui/{os.getuid()}"


def install(home: Path, run: Runner = _run, path: Path | None = None) -> list[str]:
    """Write the plist and start the service. Returns messages for the user."""
    path = path or plist_path()
    notes = []
    warning = privacy_warning()
    if warning:
        notes.append("Warning: " + warning)
    (home / "logs").mkdir(parents=True, exist_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():                              # reinstall: stop the old one first
        run(["launchctl", "bootout", f"{_domain()}/{LABEL}"])
    path.write_text(render(home), encoding="utf-8")
    result = run(["launchctl", "bootstrap", _domain(), str(path)])
    if result.returncode != 0:
        raise RuntimeError(f"launchctl bootstrap failed: {(result.stderr or result.stdout).strip()}")
    notes.append(f"Installed {path}")
    notes.append(f"Logs: {home / 'logs'}")
    return notes


def uninstall(run: Runner = _run, path: Path | None = None) -> list[str]:
    path = path or plist_path()
    run(["launchctl", "bootout", f"{_domain()}/{LABEL}"])   # fine if it wasn't running
    if path.exists():
        path.unlink()
        return [f"Stopped the service and removed {path}"]
    return ["The service was not installed."]


def restart(run: Runner = _run) -> list[str]:
    result = run(["launchctl", "kickstart", "-k", f"{_domain()}/{LABEL}"])
    if result.returncode != 0:
        raise RuntimeError(f"restart failed (is it installed?): {(result.stderr or result.stdout).strip()}")
    return ["Restarted."]


def status(home: Path, port: int, run: Runner = _run, path: Path | None = None,
           probe: Callable[[str], bool] | None = None) -> dict:
    """Three separate questions, because each fails differently: is the file
    there, has launchd loaded it (and with what pid), and does the server answer?"""
    path = path or plist_path()
    info = {"installed": path.exists(), "loaded": False, "pid": None, "last_exit": None,
            "answering": False, "url": f"http://127.0.0.1:{port}",
            "logs": str(home / "logs"), "warning": privacy_warning()}
    result = run(["launchctl", "print", f"{_domain()}/{LABEL}"])
    if result.returncode == 0:
        info["loaded"] = True
        for line in result.stdout.splitlines():
            key, _, value = line.strip().partition(" = ")
            if key == "pid":
                info["pid"] = int(value) if value.isdigit() else None
            elif key == "last exit code":
                info["last_exit"] = value
    info["answering"] = (probe or _answers)(f"{info['url']}/api/status")
    return info


def _answers(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as res:
            return res.status == 200
    except OSError:
        return False


def main(action: str, home: Path, port: int, out=print) -> int:
    if sys.platform != "darwin" and action != "plist":
        out("The service uses macOS launchd. On Linux, run `python main.py serve` under systemd.")
        return 1
    try:
        if action == "plist":
            out(render(home))
        elif action == "install":
            for line in install(home):
                out(line)
            out(f"Dashboard: http://127.0.0.1:{port} (give it a few seconds to start)")
        elif action == "uninstall":
            for line in uninstall():
                out(line)
        elif action == "restart":
            for line in restart():
                out(line)
        elif action == "status":
            s = status(home, port)
            out(f"installed  {'yes' if s['installed'] else 'no'}")
            out(f"loaded     {'yes' if s['loaded'] else 'no'}"
                + (f" (pid {s['pid']})" if s["pid"] else "")
                + (f", last exit {s['last_exit']}" if s["last_exit"] else ""))
            out(f"answering  {'yes, ' + s['url'] if s['answering'] else 'no'}")
            out(f"logs       {s['logs']}")
            if s["installed"] and not s["answering"]:
                out("Not answering: check service.err.log in the logs folder.")
                if s["warning"]:
                    out(s["warning"])
            return 0 if s["answering"] or not s["installed"] else 1
        else:
            out(f"unknown action {action}")
            return 2
    except RuntimeError as exc:
        out(f"Error: {exc}")
        return 1
    return 0
