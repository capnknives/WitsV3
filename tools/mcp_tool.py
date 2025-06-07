"""
MCP Tool for WitsV3.
Provides integration with MCP servers and tools.
"""

import logging
import json
from typing import Any, Dict, List, Optional

from core.base_tool import BaseTool
from core.mcp_adapter import MCPAdapter, MCPTool as MCPToolInfo
from core.enhanced_mcp_adapter import EnhancedMCPAdapter
from core.schemas import ToolCall

logger = logging.getLogger(__name__)


class MCPTool(BaseTool):
    """Tool that wraps MCP tools for use in WitsV3."""
    
    def __init__(self, name: str, description: str, mcp_tool: MCPToolInfo, mcp_adapter: MCPAdapter):
        """Initialize the MCP tool wrapper."""
        super().__init__(name=name, description=description)
        self.mcp_tool = mcp_tool
        self.mcp_adapter = mcp_adapter
        self.input_schema = mcp_tool.input_schema
        self.server_name = mcp_tool.server_name
        
    async def execute(self, **kwargs) -> Any:
        """Execute the MCP tool."""
        try:
            # Create a tool call for the MCP adapter
            tool_call = ToolCall(
                call_id=f"mcp_{self.name}_{id(kwargs)}",
                tool_name=self.mcp_tool.name,
                arguments=kwargs
            )
            
            # Call the MCP tool
            result = await self.mcp_adapter.call_tool(tool_call)
            
            if not result.success:
                self.logger.error(f"MCP tool {self.name} failed: {result.error}")
                return f"Error: {result.error}"
            
            return result.result
            
        except Exception as e:
            self.logger.error(f"Error executing MCP tool {self.name}: {e}")
            return f"Error: {str(e)}"
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the tool's schema for LLM consumption."""
        return self.input_schema
