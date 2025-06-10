"""
Test module for the MCP tool registry in WitsV3.
Tests the MCP tool registry for loading MCP tool configurations and registering tools.
"""

import os
import json
import asyncio
import tempfile
import logging
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from typing import Dict, Any, List

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tools.mcp_tool_registry import MCPToolRegistry
from tools.mcp_tool import MCPTool
from core.mcp_adapter import MCPAdapter, MCPServer, MCPTool as MCPToolInfo
from core.enhanced_mcp_adapter import EnhancedMCPAdapter
from core.tool_registry import ToolRegistry
from core.schemas import ToolCall, ToolResult

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def mock_tool_registry():
    """Fixture for a mock tool registry"""
    registry = MagicMock(spec=ToolRegistry)
    registry.register_tool = MagicMock()
    return registry


@pytest.fixture
def mock_mcp_adapter():
    """Fixture for a mock MCP adapter"""
    adapter = AsyncMock(spec=EnhancedMCPAdapter)

    # Mock add_server method
    adapter.add_server = AsyncMock(return_value=True)

    # Mock list_available_tools method
    adapter.list_available_tools = AsyncMock(return_value=[
        MCPToolInfo(
            name="test_tool",
            description="A test tool",
            input_schema={"type": "object", "properties": {"arg1": {"type": "string"}}},
            server_name="test_server"
        )
    ])

    return adapter


@pytest.fixture
def temp_config_file():
    """Fixture for a temporary MCP tools config file"""
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json') as f:
        config = {
            "auto_connect": True,
            "servers": [
                {
                    "name": "test_server",
                    "command": ["echo", "test"],
                    "working_directory": "."
                }
            ]
        }
        json.dump(config, f)
        f.flush()
        yield f.name

    # Clean up
    os.unlink(f.name)


@pytest.mark.asyncio
async def test_initialize(mock_tool_registry, mock_mcp_adapter, temp_config_file):
    """Test initializing the MCP tool registry"""
    # Create registry with temp config file
    registry = MCPToolRegistry(config_path=temp_config_file)

    # Mock EnhancedMCPAdapter creation
    with patch('tools.mcp_tool_registry.EnhancedMCPAdapter', return_value=mock_mcp_adapter):
        success = await registry.initialize(mock_tool_registry)

        assert success is True
        assert registry.mcp_adapter is mock_mcp_adapter
        mock_mcp_adapter.add_server.assert_called_once()
        mock_mcp_adapter.list_available_tools.assert_called_once()
        mock_tool_registry.register_tool.assert_called_once()


@pytest.mark.asyncio
async def test_load_config_file_not_found():
    """Test loading a non-existent config file"""
    registry = MCPToolRegistry(config_path="nonexistent_file.json")
    success = await registry._load_config()
    assert success is False


@pytest.mark.asyncio
async def test_connect_to_servers(mock_mcp_adapter, temp_config_file):
    """Test connecting to MCP servers"""
    registry = MCPToolRegistry(config_path=temp_config_file)

    # Load config
    await registry._load_config()

    # Set MCP adapter
    registry.mcp_adapter = mock_mcp_adapter

    # Connect to servers
    await registry._connect_to_servers()

    mock_mcp_adapter.add_server.assert_called_once()


@pytest.mark.asyncio
async def test_register_tools_with_registry(mock_tool_registry, mock_mcp_adapter):
    """Test registering MCP tools with the main tool registry"""
    registry = MCPToolRegistry()
    registry.mcp_adapter = mock_mcp_adapter

    await registry._register_tools_with_registry(mock_tool_registry)

    mock_mcp_adapter.list_available_tools.assert_called_once()
    mock_tool_registry.register_tool.assert_called_once()


@pytest.mark.asyncio
async def test_shutdown(mock_mcp_adapter):
    """Test shutting down the MCP tool registry"""
    registry = MCPToolRegistry()
    registry.mcp_adapter = mock_mcp_adapter

    await registry.shutdown()

    mock_mcp_adapter.shutdown.assert_called_once()


@pytest.mark.skipif(not os.environ.get("WITSV3_RUN_INTEGRATION_TESTS"),
                    reason="Integration tests are disabled")
@pytest.mark.asyncio
async def test_mcp_tool_registry_integration():
    """Test integration with actual MCP servers"""
    # Create real tool registry
    tool_registry = ToolRegistry()

    # Create MCP tool registry with real config file
    mcp_registry = MCPToolRegistry(config_path="data/mcp_tools.json")

    try:
        # Initialize registry
        success = await mcp_registry.initialize(tool_registry)
        assert success, "Failed to initialize MCP tool registry"

        # Check if tools were registered
        tools = tool_registry.list_all_tools()
        mcp_tools = [tool for tool in tools if tool.name.startswith("mcp_")]
        assert len(mcp_tools) > 0, "No MCP tools were registered"

        # Verify tool schemas
        for tool in mcp_tools:
            schema = tool.get_schema()
            assert schema is not None, f"Tool {tool.name} has no schema"

            # If it's a read file tool, try to execute it
            if "read" in tool.name.lower() and "file" in tool.name.lower():
                result = await tool.execute(path="data/mcp_tools.json")
                assert result is not None, "Failed to execute read file tool"

    finally:
        # Clean up
        if mcp_registry.mcp_adapter:
            await mcp_registry.shutdown()


if __name__ == "__main__":
    asyncio.run(test_mcp_tool_registry_integration())
