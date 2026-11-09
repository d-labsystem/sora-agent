"""MCP client against a real MCP server process (tests/mcp_demo_server.py)."""

import json
import sys
from pathlib import Path

import pytest

from tools.mcp_client import MCPManager, load_config, tool_name
from tools.registry import ToolRegistry

DEMO = str(Path(__file__).parent / "mcp_demo_server.py")


@pytest.fixture
def manager(tmp_path):
    config = {
        "demo": {"command": sys.executable, "args": [DEMO]},
        "broken": {"command": sys.executable, "args": ["-c", "import sys; sys.exit(1)"]},
        "off": {"command": "nothing", "disabled": True},
    }
    m = MCPManager(config, tmp_path / "logs")
    m.start(timeout=20)
    yield m
    m.close()


def test_tools_are_namespaced_and_callable(manager):
    registry = ToolRegistry()
    assert sorted(manager.register(registry)) == ["demo__add", "demo__fail"]
    assert registry.get("demo__add").source == "mcp:demo"
    assert registry.execute("demo__add", {"a": 2, "b": 3}) == ("5", False)


def test_server_errors_reach_the_model(manager):
    registry = ToolRegistry()
    manager.register(registry)
    output, error = registry.execute("demo__fail", {"reason": "disk full"})
    assert error and "disk full" in output


def test_one_broken_server_does_not_stop_others(manager):
    status = {s["name"]: s for s in manager.status()}
    assert status["demo"]["status"] == "connected" and status["demo"]["tools"] == ["add", "fail"]
    assert status["broken"]["status"] == "failed" and status["broken"]["error"]
    assert status["off"]["status"] == "disabled"


def test_config_placeholders_and_missing_file(tmp_path):
    assert load_config(tmp_path / "none.json", tmp_path) == {}
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"mcpServers": {"w": {"command": "{python}", "args": ["{home}/ws"]}}}))
    config = load_config(path, Path("/h"))
    assert config["w"] == {"command": sys.executable, "args": ["/h/ws"]}


def test_tool_names_are_api_safe():
    assert tool_name("my server", "do.thing") == "my_server__do_thing"
    assert len(tool_name("s" * 40, "t" * 40)) == 64
