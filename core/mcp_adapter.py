"""
MCP (Model Context Protocol) Adapter for WitsV3
Provides integration with MCP tools and servers
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .schemas import ToolCall, ToolResult, StreamData

logger = logging.getLogger(__name__)


@dataclass
class MCPServer:
    """Configuration for an MCP server connection"""
    name: str
    command: List[str]
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None
    working_directory: Optional[str] = None


@dataclass
class MCPTool:
    """Represents an MCP tool"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    server_name: str


class MCPClient(ABC):
    """Abstract base class for MCP client implementations"""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the MCP server"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the MCP server"""
        pass
    
    @abstractmethod
    async def list_tools(self) -> List[MCPTool]:
        """List available tools from the MCP server"""
        pass
    
    @abstractmethod
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on the MCP server"""
        pass


class StdioMCPClient(MCPClient):
    """MCP client that communicates via stdio with an MCP server process"""
    
    def __init__(self, server_config: MCPServer):
        self.server_config = server_config
        self.process: Optional[asyncio.subprocess.Process] = None
        self.is_connected = False
        self.request_id = 0
        self._tools_cache: List[MCPTool] = []
        
    async def connect(self) -> bool:
        """Start the MCP server process and establish connection"""
        try:
            # Start the MCP server process
            cmd = self.server_config.command
            if self.server_config.args:
                cmd.extend(self.server_config.args)
                
            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self.server_config.env,
                cwd=self.server_config.working_directory
            )
            
            # Send initialize request
            initialize_request = {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "clientInfo": {
                        "name": "WitsV3",
                        "version": "1.0.0"
                    }
                }
            }
            
            await self._send_request(initialize_request)
            response = await self._read_response()
            
            if response and response.get("result"):
                self.is_connected = True
                logger.info(f"Connected to MCP server: {self.server_config.name}")
                
                # Send initialized notification
                initialized_notification = {
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized"
                }
                await self._send_request(initialized_notification)
                
                return True
            else:
                logger.error(f"Failed to initialize MCP server: {self.server_config.name}")
                return False
                
        except Exception as e:
            logger.error(f"Error connecting to MCP server {self.server_config.name}: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from the MCP server"""
        if self.process:
            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
            self.process = None
        self.is_connected = False
        logger.info(f"Disconnected from MCP server: {self.server_config.name}")
    
    async def list_tools(self) -> List[MCPTool]:
        """List available tools from the MCP server"""
        if not self.is_connected:
            return []
            
        try:
            request = {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "tools/list"
            }
            
            await self._send_request(request)
            response = await self._read_response()
            
            if response and "result" in response:
                tools = []
                for tool_data in response["result"].get("tools", []):
                    tool = MCPTool(
                        name=tool_data["name"],
                        description=tool_data.get("description", ""),
                        input_schema=tool_data.get("inputSchema", {}),
                        server_name=self.server_config.name
                    )
                    tools.append(tool)
                
                self._tools_cache = tools
                return tools
            else:
                logger.error(f"Failed to list tools from {self.server_config.name}")
                return []
                
        except Exception as e:
            logger.error(f"Error listing tools from {self.server_config.name}: {e}")
            return []
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on the MCP server"""
        if not self.is_connected:
            return {"error": "Not connected to MCP server"}
            
        try:
            request = {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "tools/call",
                "params": {
                    "name": name,
                    "arguments": arguments
                }
            }
            
            await self._send_request(request)
            response = await self._read_response()
            
            if response and "result" in response:
                return response["result"]
            else:
                error_msg = response.get("error", {}).get("message", "Unknown error") if response else "No response"
                return {"error": f"Tool call failed: {error_msg}"}
                
        except Exception as e:
            logger.error(f"Error calling tool {name} on {self.server_config.name}: {e}")
            return {"error": str(e)}
    
    def _next_id(self) -> int:
        """Generate next request ID"""
        self.request_id += 1
        return self.request_id
    
    async def _send_request(self, request: Dict[str, Any]) -> None:
        """Send a JSON-RPC request to the MCP server"""
        if not self.process or not self.process.stdin:
            raise Exception("No active process to send request to")
            
        message = json.dumps(request) + "\n"
        self.process.stdin.write(message.encode())
        await self.process.stdin.drain()
    
    async def _read_response(self) -> Optional[Dict[str, Any]]:
        """Read a JSON-RPC response from the MCP server"""
        if not self.process or not self.process.stdout:
            return None
            
        try:
            line = await asyncio.wait_for(self.process.stdout.readline(), timeout=10.0)
            if line:
                return json.loads(line.decode().strip())
            return None
        except (asyncio.TimeoutError, json.JSONDecodeError) as e:
            logger.error(f"Error reading response: {e}")
            return None


class MCPAdapter:
    """Main adapter for managing MCP server connections and tool calls"""
    
    def __init__(self):
        self.clients: Dict[str, MCPClient] = {}
        self.tools: Dict[str, MCPTool] = {}
        
    async def add_server(self, server_config: MCPServer) -> bool:
        """Add and connect to an MCP server"""
        client = StdioMCPClient(server_config)
        
        if await client.connect():
            self.clients[server_config.name] = client
            
            # Load tools from this server
            tools = await client.list_tools()
            for tool in tools:
                self.tools[tool.name] = tool
                
            logger.info(f"Added MCP server {server_config.name} with {len(tools)} tools")
            return True
        else:
            logger.error(f"Failed to add MCP server {server_config.name}")
            return False
    
    async def remove_server(self, server_name: str) -> None:
        """Remove and disconnect from an MCP server"""
        if server_name in self.clients:
            await self.clients[server_name].disconnect()
            del self.clients[server_name]
            
            # Remove tools from this server
            tools_to_remove = [name for name, tool in self.tools.items() 
                             if tool.server_name == server_name]
            for tool_name in tools_to_remove:
                del self.tools[tool_name]
                
            logger.info(f"Removed MCP server {server_name}")
    
    async def list_available_tools(self) -> List[MCPTool]:
        """List all available tools from all connected servers"""
        return list(self.tools.values())
    
    async def call_tool(self, tool_call: ToolCall) -> ToolResult:
        """Call an MCP tool and return the result"""
        tool_name = tool_call.name
        
        if tool_name not in self.tools:
            return ToolResult(
                call_id=tool_call.call_id,
                success=False,
                result=None,
                error=f"Tool '{tool_name}' not found"
            )
        
        tool = self.tools[tool_name]
        client = self.clients.get(tool.server_name)
        
        if not client:
            return ToolResult(
                call_id=tool_call.call_id,
                success=False,
                result=None,
                error=f"Server '{tool.server_name}' not connected"
            )
        
        try:
            result = await client.call_tool(tool_name, tool_call.arguments)
            
            if "error" in result:
                return ToolResult(
                    call_id=tool_call.call_id,
                    success=False,
                    result=None,
                    error=result["error"]
                )
            else:
                return ToolResult(
                    call_id=tool_call.call_id,
                    success=True,
                    result=result,
                    error=None
                )
                
        except Exception as e:
            return ToolResult(
                call_id=tool_call.call_id,
                success=False,
                result=None,
                error=str(e)
            )
    
    async def shutdown(self) -> None:
        """Shutdown all MCP server connections"""
        for server_name in list(self.clients.keys()):
            await self.remove_server(server_name)
        logger.info("MCP adapter shutdown complete")


# Test function
async def test_mcp_adapter():
    """Test the MCP adapter functionality"""
    adapter = MCPAdapter()
    
    # Example server configuration (would need actual MCP server)
    server_config = MCPServer(
        name="filesystem",
        command=["npx", "-y", "@modelcontextprotocol/server-filesystem"],
        args=["/tmp"]
    )
    
    try:
        # Test adding server (will fail without actual MCP server)
        success = await adapter.add_server(server_config)
        print(f"Server connection: {'Success' if success else 'Failed'}")
        
        if success:
            # Test listing tools
            tools = await adapter.list_available_tools()
            print(f"Available tools: {[tool.name for tool in tools]}")
            
            # Test tool call
            if tools:
                tool_call = ToolCall(
                    call_id="test_1",
                    name=tools[0].name,
                    arguments={}
                )
                result = await adapter.call_tool(tool_call)
                print(f"Tool call result: {result}")
        
    finally:
        await adapter.shutdown()


if __name__ == "__main__":
    asyncio.run(test_mcp_adapter())
