"""
MCP (Model Context Protocol) Adapter for WitsV3
Provides integration with MCP tools and servers
"""

import asyncio
import json
import logging
import os
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from .schemas import ToolCall, ToolResult

logger = logging.getLogger(__name__)

# npx/uvx cold start can take 60–120s on first run.
DEFAULT_IO_TIMEOUT = 60.0
NPX_STARTUP_TIMEOUT = 120.0
DEFAULT_STARTUP_TIMEOUT = 30.0


def startup_timeout_for_command(command: list[str]) -> float:
    """How long to wait for the MCP server process to answer initialize."""
    if not command:
        return DEFAULT_STARTUP_TIMEOUT
    exe = command[0].lower()
    joined = " ".join(command).lower()
    if "npx" in exe or "uvx" in exe or "npx" in joined or "uvx" in joined:
        return NPX_STARTUP_TIMEOUT
    return DEFAULT_STARTUP_TIMEOUT


@dataclass
class MCPServer:
    """Configuration for an MCP server connection"""

    name: str
    command: list[str]
    args: list[str] | None = None
    env: dict[str, str] | None = None
    working_directory: str | None = None


@dataclass
class MCPTool:
    """Represents an MCP tool"""

    name: str
    description: str
    input_schema: dict[str, Any]
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
    async def list_tools(self) -> list[MCPTool]:
        """List available tools from the MCP server"""
        pass

    @abstractmethod
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call a tool on the MCP server"""
        pass


class StdioMCPClient(MCPClient):
    """MCP client that communicates via stdio with an MCP server process"""

    def __init__(self, server_config: MCPServer):
        self.server_config = server_config
        self.process: asyncio.subprocess.Process | None = None
        self.is_connected = False
        self.handshake_complete = False
        self.request_id = 0
        self._tools_cache: list[MCPTool] = []
        self._stderr_task: asyncio.Task | None = None
        self._io_timeout = DEFAULT_IO_TIMEOUT

    def _start_stderr_drain(self) -> None:
        """Log MCP server stderr without blocking the JSON-RPC handshake."""

        async def _drain() -> None:
            if not self.process or not self.process.stderr:
                return
            while True:
                line = await self.process.stderr.readline()
                if not line:
                    break
                text = line.decode(errors="replace").strip()
                if not text:
                    continue
                lowered = text.lower()
                if "error" in lowered and "level=info" not in lowered:
                    logger.warning(
                        "MCP %s stderr: %s",
                        self.server_config.name,
                        text[:800],
                    )
                else:
                    logger.debug("MCP %s stderr: %s", self.server_config.name, text[:500])

        self._stderr_task = asyncio.create_task(_drain())

    async def _cleanup_failed_connect(self) -> None:
        if self._stderr_task and not self._stderr_task.done():
            self._stderr_task.cancel()
        self._stderr_task = None
        if self.process and self.process.returncode is None:
            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
        self.process = None
        self.is_connected = False
        self.handshake_complete = False

    async def connect(self) -> bool:
        """Connect to the MCP server"""
        if self.is_connected and self.handshake_complete:
            return True

        try:
            logger.info(f"Connecting to MCP server: {self.server_config.name}")

            cmd = list(self.server_config.command)
            if self.server_config.args:
                cmd.extend(self.server_config.args)

            env = os.environ.copy()
            if self.server_config.env:
                env.update(self.server_config.env)

            resolved_cmd = shutil.which(cmd[0])
            if resolved_cmd:
                cmd[0] = resolved_cmd

            if isinstance(cmd[0], str) and (cmd[0].endswith(".js") or cmd[0].endswith(".ts")):
                if self.server_config.working_directory and not os.path.exists(
                    self.server_config.working_directory
                ):
                    logger.error(
                        f"MCP server working directory does not exist: {self.server_config.working_directory}"
                    )
                    return False
                if self.server_config.working_directory:
                    full_cmd_path = os.path.join(self.server_config.working_directory, cmd[0])
                    if not os.path.exists(full_cmd_path) and not os.path.exists(
                        full_cmd_path.replace("\\", "/")
                    ):
                        logger.error(f"MCP server command not found: {full_cmd_path}")
                        return False

            startup_timeout = startup_timeout_for_command(cmd)
            logger.info(
                f"Starting MCP server process: {cmd} (startup timeout {startup_timeout:.0f}s)"
            )
            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=self.server_config.working_directory,
            )
            self._start_stderr_drain()

            request = {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "WitsV3", "version": "3.0"},
                },
            }
            await self._send_request(request)
            response = await self._read_response(timeout=startup_timeout)

            if not response or "result" not in response:
                error_msg = (
                    response.get("error", {}).get("message", "No response")
                    if response
                    else f"No initialize response within {startup_timeout:.0f}s"
                )
                logger.error(f"MCP initialize failed for {self.server_config.name}: {error_msg}")
                await self._cleanup_failed_connect()
                return False

            await self._send_request(
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                }
            )
            self.is_connected = True
            self.handshake_complete = True
            logger.info(f"Successfully connected to MCP server: {self.server_config.name}")
            return True

        except Exception as e:
            logger.error(f"Error connecting to MCP server {self.server_config.name}: {e}")
            await self._cleanup_failed_connect()
            return False

    async def disconnect(self) -> None:
        """Disconnect from the MCP server"""
        if self._stderr_task and not self._stderr_task.done():
            self._stderr_task.cancel()
        self._stderr_task = None
        if self.process:
            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
            self.process = None
        self.is_connected = False
        self.handshake_complete = False
        logger.info(f"Disconnected from MCP server: {self.server_config.name}")

    async def list_tools(self) -> list[MCPTool]:
        """List available tools from the MCP server"""
        if not self.is_connected or not self.handshake_complete:
            return []

        try:
            request = {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "tools/list",
            }

            await self._send_request(request)
            response = await self._read_response(timeout=self._io_timeout)

            if response and "result" in response:
                tools = []
                for tool_data in response["result"].get("tools", []):
                    tool = MCPTool(
                        name=tool_data["name"],
                        description=tool_data.get("description", ""),
                        input_schema=tool_data.get("inputSchema", {}),
                        server_name=self.server_config.name,
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

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call a tool on the MCP server"""
        if not self.is_connected:
            return {"error": "Not connected to MCP server"}

        try:
            request = {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            }

            await self._send_request(request)
            response = await self._read_response(timeout=self._io_timeout)

            if response and "result" in response:
                return response["result"]
            else:
                error_msg = (
                    response.get("error", {}).get("message", "Unknown error")
                    if response
                    else "No response"
                )
                return {"error": f"Tool call failed: {error_msg}"}

        except Exception as e:
            logger.error(f"Error calling tool {name} on {self.server_config.name}: {e}")
            return {"error": str(e)}

    def _next_id(self) -> int:
        """Generate next request ID"""
        self.request_id += 1
        return self.request_id

    async def _send_request(self, request: dict[str, Any]) -> None:
        """Send a JSON-RPC request to the MCP server"""
        if not self.process or not self.process.stdin:
            raise Exception("No active process to send request to")

        message = json.dumps(request) + "\n"
        self.process.stdin.write(message.encode())
        await self.process.stdin.drain()

    async def _read_response(self, timeout: float | None = None) -> dict[str, Any] | None:
        """Read a JSON-RPC response from the MCP server, skipping notifications"""
        if not self.process or not self.process.stdout:
            return None

        deadline = asyncio.get_event_loop().time() + (timeout or self._io_timeout)
        try:
            while True:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    raise asyncio.TimeoutError()
                line = await asyncio.wait_for(self.process.stdout.readline(), timeout=remaining)
                if not line:
                    return None
                message = json.loads(line.decode().strip())
                if "id" in message and ("result" in message or "error" in message):
                    return message
        except (asyncio.TimeoutError, json.JSONDecodeError) as e:
            logger.error(f"Error reading MCP response from {self.server_config.name}: {e}")
            return None


class MCPAdapter:
    """Main adapter for managing MCP server connections and tool calls"""

    def __init__(self):
        self.clients: dict[str, MCPClient] = {}
        self.tools: dict[str, MCPTool] = {}

    async def add_server(self, server_config: MCPServer) -> bool:
        """Add and connect to an MCP server"""
        client = StdioMCPClient(server_config)

        if await client.connect():
            self.clients[server_config.name] = client

            tools = await client.list_tools()
            if not tools:
                logger.warning(
                    "MCP server %s connected but reported 0 tools",
                    server_config.name,
                )
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
            tools_to_remove = [
                name for name, tool in self.tools.items() if tool.server_name == server_name
            ]
            for tool_name in tools_to_remove:
                del self.tools[tool_name]

            logger.info(f"Removed MCP server {server_name}")

    async def list_available_tools(self) -> list[MCPTool]:
        """List all available tools from all connected servers"""
        return list(self.tools.values())

    async def call_tool(self, tool_call: ToolCall) -> ToolResult:
        """Call an MCP tool and return the result"""
        tool_name = tool_call.tool_name

        if tool_name not in self.tools:
            return ToolResult(
                call_id=tool_call.call_id,
                success=False,
                result=None,
                error=f"Tool '{tool_name}' not found",
            )

        tool = self.tools[tool_name]
        client = self.clients.get(tool.server_name)

        if not client:
            return ToolResult(
                call_id=tool_call.call_id,
                success=False,
                result=None,
                error=f"Server '{tool.server_name}' not connected",
            )

        try:
            result = await client.call_tool(tool_name, tool_call.arguments)

            if "error" in result:
                return ToolResult(
                    call_id=tool_call.call_id, success=False, result=None, error=result["error"]
                )
            else:
                return ToolResult(
                    call_id=tool_call.call_id, success=True, result=result, error=None
                )

        except Exception as e:
            return ToolResult(call_id=tool_call.call_id, success=False, result=None, error=str(e))

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
        args=["/tmp"],
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
                tool_call = ToolCall(call_id="test_1", tool_name=tools[0].name, arguments={})
                result = await adapter.call_tool(tool_call)
                print(f"Tool call result: {result}")

    finally:
        await adapter.shutdown()


if __name__ == "__main__":
    asyncio.run(test_mcp_adapter())
