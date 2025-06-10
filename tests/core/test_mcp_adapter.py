"""
Test module for the MCP adapter in WitsV3.
Tests the MCP adapter for connecting to MCP servers and calling tools.
"""

import os
import json
import asyncio
import logging
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from typing import Dict, Any, List

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core.mcp_adapter import MCPAdapter, MCPServer, MCPTool, StdioMCPClient
from core.schemas import ToolCall, ToolResult

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def mcp_server_config():
    """Fixture for a mock MCP server configuration"""
    return MCPServer(
        name="test_server",
        command=["echo", "test"],
        args=None,
        env=None,
        working_directory=None
    )


@pytest.fixture
def mock_mcp_client():
    """Fixture for a mock MCP client"""
    client = AsyncMock(spec=StdioMCPClient)
    client.connect = AsyncMock(return_value=True)
    client.disconnect = AsyncMock()
    client.list_tools = AsyncMock(return_value=[
        MCPTool(
            name="test_tool",
            description="A test tool",
            input_schema={"type": "object", "properties": {"arg1": {"type": "string"}}},
            server_name="test_server"
        )
    ])
    client.call_tool = AsyncMock(return_value={"result": "success"})
    return client


@pytest.fixture
def mcp_adapter():
    """Fixture for an MCP adapter"""
    return MCPAdapter()


@pytest.mark.asyncio
async def test_add_server(mcp_adapter, mcp_server_config, mock_mcp_client):
    """Test adding a server to the MCP adapter"""
    with patch("core.mcp_adapter.StdioMCPClient", return_value=mock_mcp_client):
        result = await mcp_adapter.add_server(mcp_server_config)

        assert result is True
        assert mcp_server_config.name in mcp_adapter.clients
        assert len(mcp_adapter.tools) == 1
        assert "test_tool" in mcp_adapter.tools
        mock_mcp_client.connect.assert_called_once()
        mock_mcp_client.list_tools.assert_called_once()


@pytest.mark.asyncio
async def test_remove_server(mcp_adapter, mcp_server_config, mock_mcp_client):
    """Test removing a server from the MCP adapter"""
    with patch("core.mcp_adapter.StdioMCPClient", return_value=mock_mcp_client):
        await mcp_adapter.add_server(mcp_server_config)
        await mcp_adapter.remove_server(mcp_server_config.name)

        assert mcp_server_config.name not in mcp_adapter.clients
        assert len(mcp_adapter.tools) == 0
        mock_mcp_client.disconnect.assert_called_once()


@pytest.mark.asyncio
async def test_list_available_tools(mcp_adapter, mcp_server_config, mock_mcp_client):
    """Test listing available tools from the MCP adapter"""
    with patch("core.mcp_adapter.StdioMCPClient", return_value=mock_mcp_client):
        await mcp_adapter.add_server(mcp_server_config)
        tools = await mcp_adapter.list_available_tools()

        assert len(tools) == 1
        assert tools[0].name == "test_tool"
        assert tools[0].description == "A test tool"
        assert tools[0].server_name == "test_server"


@pytest.mark.asyncio
async def test_call_tool(mcp_adapter, mcp_server_config, mock_mcp_client):
    """Test calling a tool through the MCP adapter"""
    with patch("core.mcp_adapter.StdioMCPClient", return_value=mock_mcp_client):
        await mcp_adapter.add_server(mcp_server_config)

        tool_call = ToolCall(
            call_id="test_call_id",
            tool_name="test_tool",
            arguments={"arg1": "test_value"}
        )

        result = await mcp_adapter.call_tool(tool_call)

        assert result.success is True
        assert result.call_id == "test_call_id"
        assert result.error is None
        mock_mcp_client.call_tool.assert_called_once_with(
            "test_tool", {"arg1": "test_value"}
        )


@pytest.mark.asyncio
async def test_call_nonexistent_tool(mcp_adapter):
    """Test calling a tool that doesn't exist"""
    tool_call = ToolCall(
        call_id="test_call_id",
        tool_name="nonexistent_tool",
        arguments={"arg1": "test_value"}
    )

    result = await mcp_adapter.call_tool(tool_call)

    assert result.success is False
    assert result.call_id == "test_call_id"
    assert "not found" in result.error


@pytest.mark.asyncio
async def test_mcp_adapter_shutdown(mcp_adapter, mcp_server_config, mock_mcp_client):
    """Test shutting down the MCP adapter"""
    with patch("core.mcp_adapter.StdioMCPClient", return_value=mock_mcp_client):
        await mcp_adapter.add_server(mcp_server_config)
        await mcp_adapter.shutdown()

        assert len(mcp_adapter.clients) == 0
        assert len(mcp_adapter.tools) == 0
        mock_mcp_client.disconnect.assert_called_once()


# Integration test with real MCP server - only run if environment variable is set
@pytest.mark.skipif(not os.environ.get("WITSV3_RUN_INTEGRATION_TESTS"),
                    reason="Integration tests are disabled")
@pytest.mark.asyncio
async def test_filesystem_mcp_server_integration():
    """Test integration with the filesystem MCP server"""
    # Find the filesystem MCP server
    config_path = "data/mcp_tools.json"

    if not os.path.exists(config_path):
        pytest.skip(f"MCP configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Find the filesystem server
    filesystem_server = None
    for server in config.get("servers", []):
        if "filesystem" in server.get("name", "").lower():
            filesystem_server = server
            break

    if not filesystem_server:
        pytest.skip("Filesystem MCP server not found in configuration")

    # Create the MCP adapter
    mcp_adapter = MCPAdapter()

    # Create server config
    server_config = MCPServer(
        name=filesystem_server["name"],
        command=filesystem_server["command"] if isinstance(filesystem_server["command"], list)
                else filesystem_server["command"].split(),
        working_directory=filesystem_server.get("working_directory")
    )

    try:
        # Connect to server
        success = await mcp_adapter.add_server(server_config)
        assert success, "Failed to connect to filesystem MCP server"

        # List available tools
        tools = await mcp_adapter.list_available_tools()
        assert len(tools) > 0, "No tools found from filesystem MCP server"

        # Find the readFile tool
        read_file_tool = None
        for tool in tools:
            if "read" in tool.name.lower() and "file" in tool.name.lower():
                read_file_tool = tool
                break

        if not read_file_tool:
            pytest.skip("readFile tool not found in filesystem MCP server")

        # Call the readFile tool
        tool_call = ToolCall(
            call_id="test_read_file",
            tool_name=read_file_tool.name,
            arguments={"path": config_path}  # Read the mcp_tools.json file
        )

        result = await mcp_adapter.call_tool(tool_call)

        assert result.success, f"Failed to call readFile tool: {result.error}"
        assert isinstance(result.result, dict), "Expected result to be a dictionary"
        assert "content" in result.result, "Expected 'content' in result"

        content = result.result["content"]
        assert "servers" in content, "Expected 'servers' in file content"

    finally:
        # Clean up
        await mcp_adapter.shutdown()


if __name__ == "__main__":
    asyncio.run(test_mcp_adapter_shutdown(MCPAdapter(),
                                          MCPServer(name="test", command=["echo", "test"]),
                                          AsyncMock()))
