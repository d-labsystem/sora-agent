"""The real dashboard against a scripted model: no API key, no cost.

    uv run tests/demo_server.py          # http://127.0.0.1:8790

Visuals: "chart", "table", "draw a diagram", "show markdown".

Everything except the model is real: memory, tracing, the workspace MCP
server, reflection, evals. Try: "remember that I like masala chai",
"what do I like?", "save my todo list: milk, call Priya", "what time is it?",
"use the travel-planner skill" (fails, then sora writes a lesson).
"""

import re
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from fake import FakeClient, message, text, tool_use  # noqa: E402

from harness.app import Settings, Sora  # noqa: E402
from harness.web import serve  # noqa: E402


PAUSE = 0.7   # seconds per model call, so the live diagram is visible


def small(request):
    time.sleep(PAUSE / 2)
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


CHART = {"type": "bar", "title": "Cost per turn, before and after caching", "labels": ["hi", "remember", "todo", "recall", "web"],
         "series": [{"name": "before", "values": [0.034, 0.067, 0.069, 0.034, 0.20]},
                    {"name": "after", "values": [0.042, 0.008, 0.009, 0.004, 0.147]}], "y_label": "USD"}
TABLE = {"title": "Memory types in sora", "columns": ["Memory", "Holds", "Written by", "Lifetime"],
         "rows": [["Working", "last 10 exchanges", "every turn", "this conversation"],
                  ["Semantic", "facts", "remember, consolidation", "until forgotten"],
                  ["Episodic", "dated summaries", "consolidation", "until deleted"],
                  ["Procedural", "skills, lessons", "you, reflection, save_skill", "until deleted"]]}
SVG = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 420 120"><rect width="420" height="120" fill="#f7f7f5"/>'
       '<g font-family="sans-serif" font-size="13" text-anchor="middle">'
       '<rect x="10" y="40" width="110" height="40" rx="8" fill="#e7ecfd" stroke="#3b5bdb"/><text x="65" y="65">Model</text>'
       '<rect x="155" y="40" width="110" height="40" rx="8" fill="#e7ecfd" stroke="#3b5bdb"/><text x="210" y="65">Tools</text>'
       '<rect x="300" y="40" width="110" height="40" rx="8" fill="#e7ecfd" stroke="#3b5bdb"/><text x="355" y="65">Reply</text>'
       '<path d="M120 60H155M265 60H300" stroke="#3b5bdb" stroke-width="2"/></g></svg>')


def script(request):
    time.sleep(PAUSE)
    last = request["messages"][-1]["content"]
    if isinstance(last, list):                                   # tool results came back
        result = last[0]
        if result.get("is_error"):
            return message(text(f"That didn't work: {result['content'][:120]}"))
        if "shown to the user" in result["content"]:
            return message(text("Here it is: **hover** the chart for values, or click a table header to sort."))
        return message(text(f"Done. {result['content'][:300]}"))
    words = last.lower()
    if "chart" in words:
        return message(tool_use("show_chart", CHART), stop_reason="tool_use")
    if "table" in words:
        return message(tool_use("show_table", TABLE), stop_reason="tool_use")
    if "draw" in words or "diagram" in words:
        return message(tool_use("show_diagram", {"title": "The loop", "svg": SVG}), stop_reason="tool_use")
    if "markdown" in words:
        return message(text("## Markdown works\n\n**Bold**, *italic* and `code`.\n\n| Model | $/M in |\n|---|---|\n"
                            "| Opus 5.5 | 4.00 |\n| Haiku 5.5 | 0.10 |\n\n- one\n- two\n\n<script>alert('not run')</script>"))
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
    system = "\n\n".join(b["text"] for b in request["system"])
    notes = [line[2:] for line in system.split("## Memory", 1)[-1].splitlines()
             if line.startswith("- ")] if "## Memory" in system else []
    reply = f"From memory: {notes[0]}" if notes else "Hi! (I'm the scripted demo model.)"
    return message(text(reply))


if __name__ == "__main__":
    home = Path(tempfile.mkdtemp(prefix="sora-demo-"))
    agent = Sora(Settings(home=home), FakeClient(script, small=small))
    serve(agent, 8790)
