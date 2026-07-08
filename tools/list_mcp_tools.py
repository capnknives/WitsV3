"""
List MCP tools currently connected to WITS.

Use when the user asks what MCP servers/tools are available, whether a
particular server is connected, or which mcp_* tools the orchestrator can call.
"""

import logging
from typing import Any

from core.base_tool import BaseTool

logger = logging.getLogger(__name__)


class ListMCPToolsTool(BaseTool):
    """Report MCP servers and live mcp_* tools registered with WITS."""

    def __init__(self):
        super().__init__(
            name="list_mcp_tools",
            description=(
                "List MCP servers and tools currently connected to WITS. "
                "Returns each live mcp_* tool name, its server, and description. "
                "Use when the user asks about installed/connected MCP tools, "
                "whether WITS can use a specific MCP server, or what MCP "
                "capabilities are available right now. For discovering NEW servers "
                "to install, use search_mcp_tools instead."
            ),
        )
        self.tool_registry = None

    def set_dependencies(self, config, llm_interface=None, memory_manager=None, **kwargs) -> None:
        self.tool_registry = kwargs.get("tool_registry")

    async def execute(self, server_name: str | None = None) -> dict[str, Any]:
        if not self.tool_registry:
            return {
                "success": False,
                "error": "Tool registry unavailable",
                "servers": [],
                "tools": [],
            }

        tools: list[dict[str, Any]] = []
        by_server: dict[str, list[dict[str, Any]]] = {}

        for name, tool in sorted(self.tool_registry.tools.items()):
            if not name.startswith("mcp_"):
                continue
            server = getattr(tool, "server_name", "unknown")
            if server_name and server_name.lower() not in server.lower():
                continue
            entry = {
                "registered_name": name,
                "mcp_tool_name": getattr(
                    getattr(tool, "mcp_tool", None), "name", name.removeprefix("mcp_")
                ),
                "server": server,
                "description": (tool.description or "")[:500],
            }
            tools.append(entry)
            by_server.setdefault(server, []).append(entry)

        servers = [
            {"name": srv, "tool_count": len(entries), "tools": entries}
            for srv, entries in sorted(by_server.items())
        ]

        if server_name and not tools:
            return {
                "success": True,
                "count": 0,
                "servers": [],
                "tools": [],
                "message": (
                    f"No connected MCP tools match '{server_name}'. "
                    "Open /mcp, connect the server (first npx install can take 1–2 min), "
                    "then retry. Use search_mcp_tools to find servers not yet installed."
                ),
            }

        return {
            "success": True,
            "count": len(tools),
            "servers": servers,
            "tools": tools,
            "message": (
                f"{len(tools)} MCP tool(s) live across {len(servers)} server(s). "
                "Call them by registered_name (e.g. mcp_search_code) in the orchestrator."
                if tools
                else "No MCP tools connected. Install and Connect on the /mcp page — "
                "npx servers need up to 2 minutes on first connect."
            ),
        }

    def get_schema(self) -> dict[str, Any]:
        return {
            "name": "list_mcp_tools",
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "server_name": {
                        "type": "string",
                        "description": "Optional filter (substring match on server name)",
                    },
                },
            },
        }
