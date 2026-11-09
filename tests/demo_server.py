"""The real dashboard against a scripted model: no API key, no cost.

    uv run tests/demo_server.py          # http://127.0.0.1:8790

Everything except the model is real: memory, tracing, the workspace MCP
server, reflection, evals. Try: "remember that I like masala chai",
"what do I like?", "save my todo list: milk, call Priya", "what time is it?",
"use the travel-planner skill" (fails, then sora writes a lesson).
"""

import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from fake import FakeClient, message, text, tool_use  # noqa: E402

from harness.app import Settings, Sora  # noqa: E402
from harness.web import serve  # noqa: E402


def small(request):
    prompt = request["messages"][0]["content"]
    if "You decide whether" in prompt:
        asked = prompt.rsplit("New message:", 1)[-1].strip().lower()
        return {"retrieve": "?" in asked, "query": asked, "reason": "scripted demo",
                "correction": asked.startswith("no")}
    if "review a turn" in prompt:
        return {"lessons": [{"situation": "the user names a skill",
                             "lesson": "Check the skills catalog for the exact name before calling load_skill",
                             "tool": "load_skill"}]}
    if "long-term memory of a personal" in prompt:
        return {"facts": [], "episode": "Tried out the sora dashboard demo"}
    return {"score": 4, "reason": "Scripted judge: clear and correct."}


def script(request):
    last = request["messages"][-1]["content"]
    if isinstance(last, list):                                   # tool results came back
        result = last[0]
        if result.get("is_error"):
            return message(text(f"That didn't work: {result['content'][:120]}"))
        return message(text(f"Done. {result['content'][:300]}"))
    words = last.lower()
    if words.startswith("remember"):
        fact = re.sub(r"^remember( that)?\s*", "", last, flags=re.I)
        return message(tool_use("remember", {"fact": f"Aryan says: {fact}"}), stop_reason="tool_use")
    if "time" in words:
        return message(tool_use("current_time", {}), stop_reason="tool_use")
    if "todo" in words or "save" in words:
        items = last.split(":", 1)[-1].split(",")
        body = "".join(f"- {i.strip()}\n" for i in items if i.strip())
        return message(tool_use("workspace__write_file", {"path": "todo.md", "content": body}),
                       stop_reason="tool_use")
    if "skill" in words:
        return message(tool_use("load_skill", {"name": "travel-planner"}), stop_reason="tool_use")
    notes = [line[2:] for line in request["system"].split("## Memory", 1)[-1].splitlines()
             if line.startswith("- ")] if "## Memory" in request["system"] else []
    reply = f"From memory: {notes[0]}" if notes else "Hi! (I'm the scripted demo model.)"
    return message(text(reply))


if __name__ == "__main__":
    home = Path(tempfile.mkdtemp(prefix="sora-demo-"))
    agent = Sora(Settings(home=home), FakeClient(script, small=small))
    serve(agent, 8790)
