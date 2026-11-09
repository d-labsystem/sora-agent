"""A tiny MCP server for tests: started over stdio by tests/test_mcp_client.py."""

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

server = MCPServer("demo")


@server.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@server.tool()
def fail(reason: str) -> str:
    """Always fails, to test error handling."""
    raise ToolError(f"failed: {reason}")


if __name__ == "__main__":
    server.run()
