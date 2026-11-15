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


def web_tools(max_searches: int = 0, max_fetches: int = 0, fetch_max_tokens: int = 0) -> list[dict]:
    """The web tools with spending limits. 0 means no limit.

    Web research happens inside ONE model call: Anthropic's servers search,
    fetch and filter in a loop before the response comes back, so nothing in
    our loop can trim it. These limits act there instead:

      max_uses            searches / fetches allowed per request. Past it the
                          tool returns a max_uses_exceeded error and the model
                          answers with what it has.
      max_content_tokens  how much of each fetched page or PDF is read. A
                          research paper can be 20k+ tokens; the abstract,
                          introduction and results usually come first.

    Both trade completeness for cost, so they are settings, not constants."""
    search, fetch = dict(WEB_TOOLS[0]), dict(WEB_TOOLS[1])
    if max_searches:
        search["max_uses"] = max_searches
    if max_fetches:
        fetch["max_uses"] = max_fetches
    if fetch_max_tokens:
        fetch["max_content_tokens"] = fetch_max_tokens
    return [search, fetch]


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
    def __init__(self, server_tools: list[dict] | None = None, server_source: str = "anthropic") -> None:
        self._tools: dict[str, Tool] = {}
        self.server_tools = list(server_tools or [])   # in the provider's own format
        self.server_source = server_source

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def describe(self) -> list[dict[str, str]]:
        """For the dashboard's tools page."""
        local = [{"name": t.name, "description": t.description, "source": t.source}
                 for t in self._tools.values()]
        server = [{"name": t.get("name", t["type"]), "description": t["type"], "source": self.server_source}
                  for t in self.server_tools]
        return local + server

    def schemas(self) -> list[dict[str, Any]]:
        """What the model sees on every call.

        These must stay byte-for-byte the same from call to call: tools are
        the first thing in the cached prompt prefix, so any change here (a
        lesson appended to a description, say) invalidates the whole cache
        and the next call re-writes ~10k tokens at full price. Lessons about
        tools go in the system prompt's per-turn block instead (build_system)."""
        return [{"name": t.name, "description": t.description, "input_schema": t.input_schema,
                 "eager_input_streaming": True} for t in self._tools.values()] + self.server_tools

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
