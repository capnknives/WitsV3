"""Tests for MCP adapter startup timeouts and list_mcp_tools."""

import pytest

from core.mcp_adapter import (
    DEFAULT_STARTUP_TIMEOUT,
    NPX_STARTUP_TIMEOUT,
    startup_timeout_for_command,
)
from tools.list_mcp_tools import ListMCPToolsTool


def test_startup_timeout_npx():
    assert startup_timeout_for_command(["npx", "-y", "pkg"]) == NPX_STARTUP_TIMEOUT


def test_startup_timeout_uvx():
    assert startup_timeout_for_command(["uvx", "pkg"]) == NPX_STARTUP_TIMEOUT


def test_startup_timeout_node():
    assert startup_timeout_for_command(["node", "index.js"]) == DEFAULT_STARTUP_TIMEOUT


@pytest.mark.asyncio
async def test_list_mcp_tools_from_registry():
    from types import SimpleNamespace

    class FakeRegistry:
        tools = {
            "mcp_search_code": SimpleNamespace(
                description="Search code graph",
                server_name="io.github.DeusData/codebase-memory-mcp",
                mcp_tool=SimpleNamespace(name="search_code"),
            ),
            "web_search": SimpleNamespace(description="web", server_name="builtin"),
        }

    tool = ListMCPToolsTool()
    tool.set_dependencies(None, tool_registry=FakeRegistry())
    result = await tool.execute()
    assert result["success"] is True
    assert result["count"] == 1
    assert result["tools"][0]["registered_name"] == "mcp_search_code"


@pytest.mark.asyncio
async def test_list_mcp_tools_filter_server():
    from types import SimpleNamespace

    class FakeRegistry:
        tools = {
            "mcp_a": SimpleNamespace(
                description="a",
                server_name="server-a",
                mcp_tool=SimpleNamespace(name="a"),
            ),
        }

    tool = ListMCPToolsTool()
    tool.set_dependencies(None, tool_registry=FakeRegistry())
    result = await tool.execute(server_name="server-b")
    assert result["count"] == 0
    assert "No connected MCP tools" in result["message"]
