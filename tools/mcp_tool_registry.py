"""
MCP Tool Registry for WitsV3.
Handles loading and registering MCP tools with the main ToolRegistry.
"""

import logging
import json
import os
from typing import Any, Dict, List, Optional

from core.mcp_adapter import MCPAdapter, MCPServer
from core.enhanced_mcp_adapter import EnhancedMCPAdapter
from core.tool_registry import ToolRegistry
from tools.mcp_tool import MCPTool

logger = logging.getLogger(__name__)


class MCPToolRegistry:
    """
    Registry for MCP tools.
    Handles loading MCP server configurations and registering MCP tools with the main ToolRegistry.
    """

    def __init__(self, config_path: str = "data/mcp_tools.json"):
        """
        Initialize the MCP tool registry.

        Args:
            config_path: Path to MCP tools configuration file
        """
        self.config_path = config_path
        self.mcp_adapter: Optional[EnhancedMCPAdapter] = None
        self.logger = logging.getLogger("WitsV3.MCPToolRegistry")
        self.config: Dict[str, Any] = {}

    async def initialize(self, tool_registry: ToolRegistry) -> bool:
        """
        Initialize the MCP tool registry and connect to MCP servers.

        Args:
            tool_registry: Main tool registry to register MCP tools with

        Returns:
            True if initialization was successful, False otherwise
        """
        try:
            # Load configuration
            if not await self._load_config():
                return False

            # Create MCP adapter
            self.mcp_adapter = EnhancedMCPAdapter(self.config_path)

            # Connect to MCP servers
            if self.config.get("auto_connect", True):
                await self._connect_to_servers()

            # Register MCP tools with the main tool registry
            await self._register_tools_with_registry(tool_registry)

            self.logger.info("MCP tool registry initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize MCP tool registry: {e}")
            return False

    async def _load_config(self) -> bool:
        """
        Load MCP configuration from file.

        Returns:
            True if configuration was loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(self.config_path):
                self.logger.error(f"MCP configuration file not found: {self.config_path}")
                return False

            with open(self.config_path, 'r') as f:
                self.config = json.load(f)

            self.logger.info(f"Loaded MCP configuration from {self.config_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load MCP configuration: {e}")
            return False

    async def _connect_to_servers(self) -> None:
        """Connect to MCP servers defined in the configuration."""
        if not self.mcp_adapter:
            self.logger.error("MCP adapter not initialized")
            return

        servers = self.config.get("servers", [])
        self.logger.info(f"Connecting to {len(servers)} MCP servers")

        for server_config in servers:
            try:
                server_name = server_config.get("name", "unknown")

                # Skip servers that don't have the required configurations
                if not ("command" in server_config or
                        ("type" in server_config and server_config["type"] == "github" and "clone_url" in server_config)):
                    self.logger.error(f"Invalid server configuration for {server_name}: missing required fields")
                    continue

                # Check if working directory exists for command-based configs
                if "command" in server_config and "working_directory" in server_config:
                    working_dir = server_config["working_directory"]
                    if not os.path.exists(working_dir):
                        self.logger.warning(f"Working directory does not exist: {working_dir}")

                        # Try to find the directory by converting relative paths
                        if working_dir.startswith("mcp_servers"):
                            abs_path = os.path.join(os.getcwd(), working_dir)
                            if os.path.exists(abs_path):
                                server_config["working_directory"] = abs_path
                                self.logger.info(f"Updated working directory to absolute path: {abs_path}")
                            else:
                                self.logger.error(f"Could not find working directory: {working_dir}")
                                continue
                        else:
                            self.logger.error(f"Working directory not found: {working_dir}")
                            continue

                # Handle GitHub repository configuration
                if server_config.get("type") == "github" and "clone_url" in server_config:
                    # Use install_tool method for GitHub repositories
                    success = await self.mcp_adapter.install_tool(server_config["clone_url"])

                    if success:
                        self.logger.info(f"Connected to MCP server via GitHub: {server_name}")
                    else:
                        # Try fallback approach for GitHub repos
                        if "working_directory" in server_config and "command" in server_config:
                            self.logger.info(f"Trying direct connection for {server_name}")

                            # Create MCPServer object
                            mcp_server = MCPServer(
                                name=server_name,
                                command=server_config["command"].split() if isinstance(server_config["command"], str)
                                       else server_config["command"],
                                args=server_config.get("args"),
                                env=server_config.get("env"),
                                working_directory=server_config.get("working_directory")
                            )

                            # Connect to server
                            direct_success = await self.mcp_adapter.add_server(mcp_server)
                            if direct_success:
                                self.logger.info(f"Successfully connected to MCP server using direct method: {server_name}")
                            else:
                                self.logger.warning(f"Failed to connect to MCP server via any method: {server_name}")
                        else:
                            self.logger.warning(f"Failed to connect to MCP server via GitHub: {server_name}")

                # Otherwise, use the direct command approach
                elif "command" in server_config:
                    # Create MCPServer object
                    mcp_server = MCPServer(
                        name=server_name,
                        command=server_config["command"].split() if isinstance(server_config["command"], str)
                               else server_config["command"],
                        args=server_config.get("args"),
                        env=server_config.get("env"),
                        working_directory=server_config.get("working_directory")
                    )

                    # Connect to server
                    success = await self.mcp_adapter.add_server(mcp_server)

                    if success:
                        self.logger.info(f"Connected to MCP server: {server_name}")
                    else:
                        self.logger.warning(f"Failed to connect to MCP server: {server_name}")
                else:
                    self.logger.error(f"Invalid server configuration for {server_name}: missing command or GitHub information")

            except Exception as e:
                self.logger.error(f"Error connecting to MCP server: {e}")

                # Provide more diagnostic information
                if "working_directory" in server_config:
                    working_dir = server_config["working_directory"]
                    if not os.path.exists(working_dir):
                        self.logger.error(f"Working directory does not exist: {working_dir}")
                    else:
                        self.logger.info(f"Directory contents: {os.listdir(working_dir)}")

                        # Check for command file
                        if "command" in server_config and isinstance(server_config["command"], str):
                            cmd_parts = server_config["command"].split()
                            if cmd_parts and os.path.isfile(os.path.join(working_dir, cmd_parts[0])):
                                self.logger.info(f"Command file exists: {cmd_parts[0]}")
                            else:
                                self.logger.error(f"Command file not found: {cmd_parts[0] if cmd_parts else 'unknown'}")

                                # List all .js and .ts files to help debugging
                                js_files = [f for f in os.listdir(working_dir) if f.endswith('.js') or f.endswith('.ts')]
                                if js_files:
                                    self.logger.info(f"Available JS/TS files: {js_files}")

    async def _register_tools_with_registry(self, tool_registry: ToolRegistry) -> None:
        """
        Register MCP tools with the main tool registry.

        Args:
            tool_registry: Main tool registry to register tools with
        """
        if not self.mcp_adapter:
            self.logger.error("MCP adapter not initialized")
            return

        try:
            # Get all available tools from MCP adapter
            mcp_tools = await self.mcp_adapter.list_available_tools()

            self.logger.info(f"Registering {len(mcp_tools)} MCP tools with tool registry")

            for mcp_tool in mcp_tools:
                # Create MCPTool wrapper
                tool = MCPTool(
                    name=f"mcp_{mcp_tool.name}",
                    description=mcp_tool.description,
                    mcp_tool=mcp_tool,
                    mcp_adapter=self.mcp_adapter
                )

                # Register with main tool registry
                tool_registry.register_tool(tool)
                self.logger.info(f"Registered MCP tool: {tool.name}")

        except Exception as e:
            self.logger.error(f"Error registering MCP tools: {e}")

    async def shutdown(self) -> None:
        """Shutdown all MCP server connections."""
        if self.mcp_adapter:
            await self.mcp_adapter.shutdown()
            self.logger.info("MCP tool registry shutdown complete")
