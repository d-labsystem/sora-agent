"""MCP client: plug any MCP server's tools into the agent.

MCP (Model Context Protocol) is a standard way for a program to offer tools
to an AI app. A server for GitHub, your calendar or a folder of files works
with any client that speaks MCP, so we write the connector once and get every
server for free. This is the USB port of the agent.

Configuration is ~/.sora/mcp.json, in the same shape Claude Desktop uses, so
examples from server READMEs paste straight in:

    {"mcpServers": {
        "workspace": {"command": "{python}", "args": ["-m", "tools.workspace_server", "{home}/workspace"]},
        "remote":    {"url": "https://example.com/mcp", "headers": {"Authorization": "Bearer ..."}},
        "github":    {"command": "uvx", "args": ["..."], "disabled": true}
    }}

{home} becomes the sora home folder and {python} this Python interpreter.

How it runs. The MCP SDK is async, but the agent loop is plain synchronous
code. Rather than make everything async, MCP gets its own thread running an
asyncio event loop. Each server is one long-lived task on that loop that
connects, lists tools, then holds the connection open until shutdown. Tool
calls from the agent thread are handed to that loop and waited on with a
timeout. One slow or crashed server never blocks the others.

Tool names become `server__tool`, so two servers can both have a "search".
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import threading
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx2
from mcp import ClientSession, types
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client

from tools.registry import Tool, ToolError, ToolRegistry

log = logging.getLogger(__name__)

CONNECT_TIMEOUT = 30
CALL_TIMEOUT = 120
MAX_OUTPUT = 50_000
NAME_LIMIT = 64


def tool_name(server: str, tool: str) -> str:
    """`server__tool`, limited to what the Claude API accepts: letters,
    digits, _ and -, at most 64 characters."""
    clean = lambda s: re.sub(r"[^A-Za-z0-9_-]", "_", s)
    return f"{clean(server)}__{clean(tool)}"[:NAME_LIMIT]


def load_config(path: Path, home: Path) -> dict[str, dict]:
    """Read mcp.json and fill in placeholders. A missing file means no servers."""
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    servers = raw.get("mcpServers", raw)
    if not isinstance(servers, dict):
        raise ValueError(f"{path}: mcpServers must be an object")

    def fill(value: Any) -> Any:
        if isinstance(value, str):
            return value.replace("{home}", str(home)).replace("{python}", sys.executable)
        if isinstance(value, list):
            return [fill(v) for v in value]
        if isinstance(value, dict):
            return {k: fill(v) for k, v in value.items()}
        return value

    return {name: fill(spec) for name, spec in servers.items() if isinstance(spec, dict)}


def result_text(result: types.CallToolResult) -> str:
    """Turn MCP content blocks into the text the model reads."""
    parts = []
    for block in result.content:
        if block.type == "text":
            parts.append(block.text)
        elif block.type == "resource" and hasattr(block.resource, "text"):
            parts.append(block.resource.text)
        else:
            parts.append(f"[{block.type} content omitted]")
    if not parts and result.structured_content is not None:
        parts.append(json.dumps(result.structured_content, ensure_ascii=False))
    text = "\n".join(parts) or "(no output)"
    if len(text) > MAX_OUTPUT:
        text = text[:MAX_OUTPUT] + f"\n[truncated: {len(text) - MAX_OUTPUT} more characters]"
    return text


@dataclass
class ServerState:
    name: str
    spec: dict
    status: str = "starting"          # starting | connected | failed | disabled | stopped
    error: str = ""
    tools: list[types.Tool] = field(default_factory=list)
    session: ClientSession | None = None
    ready: threading.Event = field(default_factory=threading.Event)
    stop: asyncio.Event | None = None

    def summary(self) -> dict:
        return {"name": self.name, "status": self.status, "error": self.error,
                "transport": "http" if "url" in self.spec else "stdio",
                "tools": [t.name for t in self.tools]}


class MCPManager:
    def __init__(self, config: dict[str, dict], log_dir: Path):
        self.servers = {name: ServerState(name, spec) for name, spec in config.items()}
        self.log_dir = log_dir
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, name="mcp", daemon=True)

    # ---- lifecycle

    def start(self, timeout: float = CONNECT_TIMEOUT) -> None:
        """Connect to every enabled server in parallel, and wait until each
        has connected or failed (or the timeout passes)."""
        self.thread.start()
        for state in self.servers.values():
            if state.spec.get("disabled"):
                state.status = "disabled"
                state.ready.set()
            else:
                asyncio.run_coroutine_threadsafe(self._run(state), self.loop)
        for state in self.servers.values():
            if not state.ready.wait(timeout):
                state.status, state.error = "failed", f"no answer within {timeout:.0f}s"

    async def _run(self, state: ServerState) -> None:
        state.stop = asyncio.Event()
        try:
            async with AsyncExitStack() as stack:
                read, write = await stack.enter_async_context(self._transport(state, stack))
                session = await stack.enter_async_context(ClientSession(read, write))
                await asyncio.wait_for(session.initialize(), CONNECT_TIMEOUT)
                state.tools = await self._list_tools(session)
                state.session, state.status = session, "connected"
                state.ready.set()
                log.info("MCP %s connected with %d tools", state.name, len(state.tools))
                await state.stop.wait()
        except Exception as exc:
            state.error = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
            log.warning("MCP %s failed: %s", state.name, state.error)
        finally:
            state.session = None
            state.status = "stopped" if state.stop.is_set() else "failed"
            state.ready.set()

    def _transport(self, state: ServerState, stack: AsyncExitStack):
        spec = state.spec
        if "url" in spec:
            client = httpx2.AsyncClient(headers=spec.get("headers") or {}, timeout=CALL_TIMEOUT)
            stack.push_async_callback(client.aclose)
            return streamable_http_client(spec["url"], http_client=client)
        if "command" not in spec:
            raise ValueError("needs either 'command' (stdio) or 'url' (http)")
        # Server stderr goes to a log file: it's where their crash messages are.
        self.log_dir.mkdir(parents=True, exist_ok=True)
        errlog = stack.enter_context(open(self.log_dir / f"mcp-{state.name}.log", "a"))
        params = StdioServerParameters(command=spec["command"], args=spec.get("args", []),
                                       env=spec.get("env"), cwd=spec.get("cwd"))
        return stdio_client(params, errlog=errlog)

    @staticmethod
    async def _list_tools(session: ClientSession) -> list[types.Tool]:
        tools, cursor = [], None
        while True:
            params = types.PaginatedRequestParams(cursor=cursor) if cursor else None
            page = await session.list_tools(params=params)
            tools += page.tools
            cursor = page.next_cursor
            if not cursor:
                return tools

    def close(self, timeout: float = 10) -> None:
        if not self.thread.is_alive():
            return
        for state in self.servers.values():
            if state.stop is not None:
                self.loop.call_soon_threadsafe(state.stop.set)

        async def drain():
            pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            await asyncio.wait(pending, timeout=timeout) if pending else None

        try:
            asyncio.run_coroutine_threadsafe(drain(), self.loop).result(timeout + 1)
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout)
            if not self.thread.is_alive():
                self.loop.close()

    # ---- tools

    def call(self, server: str, tool: str, args: dict) -> str:
        """Called from the agent thread. Raises ToolError with readable text."""
        state = self.servers[server]
        if state.session is None:
            raise ToolError(f"MCP server {server!r} is not connected ({state.status}: {state.error})")
        future = asyncio.run_coroutine_threadsafe(
            state.session.call_tool(tool, args, read_timeout_seconds=CALL_TIMEOUT), self.loop)
        try:
            result = future.result(CALL_TIMEOUT + 5)
        except TimeoutError:
            future.cancel()
            raise ToolError(f"{server} {tool} took longer than {CALL_TIMEOUT}s") from None
        if not isinstance(result, types.CallToolResult):
            raise ToolError(f"{server} {tool} asked for input this agent can't give")
        text = result_text(result)
        if result.is_error:
            raise ToolError(text)
        return text

    def register(self, registry: ToolRegistry) -> list[str]:
        """Add every connected server's tools to the registry."""
        names = []
        for state in self.servers.values():
            for t in state.tools:
                name = tool_name(state.name, t.name)
                function = lambda _s=state.name, _t=t.name, **args: self.call(_s, _t, args)
                schema = t.input_schema if t.input_schema.get("type") == "object" \
                    else {"type": "object", "properties": {}}
                registry.register(Tool(name, t.description or f"{t.name} from {state.name}",
                                       schema, function, source=f"mcp:{state.name}"))
                names.append(name)
        return names

    def status(self) -> list[dict]:
        return [s.summary() for s in self.servers.values()]
