"""The tool registry: every tool the model can call, in one place.

Three kinds of tool live here:
  local    a Python function in this project (tools/builtin.py)
  mcp      a tool from an MCP server, called through tools/mcp_client.py
  server   web_search and web_fetch, which Anthropic runs. They have no
           function here; loop/agent.py reports what they did.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

JSON_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool,
              "object": dict, "array": list}

# The _20260209 versions filter results with code before they reach the
# model, so fewer tokens come back. Nothing else is needed to turn them on.
WEB_TOOLS = [
    {"type": "web_search_20260209", "name": "web_search"},
    {"type": "web_fetch_20260209", "name": "web_fetch"},
]


class ToolError(Exception):
    """Raised by a tool to report a failure the model should read."""


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    function: Callable[..., str]
    source: str = "local"


def invalid_args(schema: dict[str, Any], args: Any) -> str | None:
    """Why `args` fails the schema, or None.

    Tool input is streamed (see loop/agent.py), and streamed input is not
    validated by the API, so this check runs before every tool."""
    if not isinstance(args, dict):
        return "arguments must be an object"
    missing = [k for k in schema.get("required", []) if k not in args]
    if missing:
        return f"missing {', '.join(missing)}"
    for key, spec in (schema.get("properties") or {}).items():
        expected = JSON_TYPES.get(spec.get("type")) if isinstance(spec, dict) else None
        if key in args and expected and not isinstance(args[key], expected):
            return f"{key} must be a {spec['type']}"
    return None


class ToolRegistry:
    def __init__(self, server_tools: list[dict] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        self.server_tools = list(server_tools or [])

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def describe(self) -> list[dict[str, str]]:
        """For the dashboard's tools page."""
        local = [{"name": t.name, "description": t.description, "source": t.source}
                 for t in self._tools.values()]
        server = [{"name": t["name"], "description": t["type"], "source": "anthropic"}
                  for t in self.server_tools]
        return local + server

    def schemas(self, notes: dict[str, list[str]] | None = None) -> list[dict[str, Any]]:
        """What the model sees on every call.

        `notes` are lessons learned about a tool (memory/learning.py). They are
        added to its description, so the model reads them as it picks a tool."""
        notes = notes or {}
        out = []
        for t in self._tools.values():
            description = t.description
            if notes.get(t.name):
                description += "\n\nNotes from past use:\n" + "\n".join(f"- {n}" for n in notes[t.name])
            out.append({"name": t.name, "description": description,
                        "input_schema": t.input_schema, "eager_input_streaming": True})
        return out + self.server_tools

    def execute(self, name: str, args: Any) -> tuple[str, bool]:
        """Run one tool call and return (output, is_error).

        A failure becomes text the model reads and can recover from, so a
        broken tool never crashes the turn."""
        tool = self._tools.get(name)
        if tool is None:
            return f"Unknown tool: {name}", True
        problem = invalid_args(tool.input_schema, args)
        if problem:
            return f"Invalid arguments: {problem}", True
        try:
            return str(tool.function(**args)), False
        except ToolError as exc:
            return str(exc), True
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}", True
