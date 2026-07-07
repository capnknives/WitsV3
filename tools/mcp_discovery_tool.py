"""
MCP discovery tool for WitsV3.

Gives WITS a way to *discover* new capabilities it doesn't currently have: when
the user asks for something no installed tool covers, WITS can search the
official MCP registry for a server that provides it.

Deliberately read-only. Searching is safe (it only queries the registry API),
but installing and connecting an MCP server downloads and runs third-party code
on this machine — so that stays a human action on the /mcp web page. This tool
tells the user what's available and how to add it; it never self-installs.
"""

import logging
from typing import Any, Dict, List, Optional

from core.base_tool import BaseTool
from core.mcp_registry_search import DEFAULT_REGISTRY_URL, search_registry

logger = logging.getLogger(__name__)


class SearchMCPToolsTool(BaseTool):
    """Search the MCP registry for servers that provide a needed capability."""

    def __init__(self):
        super().__init__(
            name="search_mcp_tools",
            description=(
                "Search the Model Context Protocol registry for installable MCP "
                "servers that would give WITS a NEW capability it doesn't already "
                "have (e.g. 'send email', 'query postgres', 'control spotify'). "
                "Returns matching servers with what they do and how to install "
                "them. This only searches — the user installs a server themselves "
                "from the MCP page. Use it when no existing tool can do the job."
            ),
        )
        self.config = None

    def set_dependencies(self, config, llm_interface=None, memory_manager=None) -> None:
        self.config = config

    def _registry_url(self) -> str:
        try:
            return self.config.tool_system.mcp_registry_url or DEFAULT_REGISTRY_URL
        except Exception:
            return DEFAULT_REGISTRY_URL

    async def execute(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        query = (query or "").strip()
        if not query:
            return {"success": False, "error": "Provide a capability to search for", "results": []}

        try:
            entries = await search_registry(
                query, limit=max(1, int(max_results)), registry_url=self._registry_url()
            )
        except Exception as e:
            logger.warning("search_mcp_tools failed: %s", e)
            return {"success": False, "error": f"MCP registry search failed: {e}", "results": []}

        results: List[Dict[str, Any]] = []
        for entry in entries:
            install = entry.get("install")
            required_env = [
                ev["name"]
                for ev in (install.get("env_vars") if install else [])
                if ev.get("required")
            ]
            results.append(
                {
                    "name": entry["name"],
                    "description": entry["description"],
                    "repository": entry["repository"],
                    "installable": install is not None,
                    "command": " ".join(install["command"]) if install else None,
                    "required_env": required_env,
                    "note": (
                        None
                        if install
                        else "Remote or non-stdio server — set up manually on the MCP page."
                    ),
                }
            )

        installable = sum(1 for r in results if r["installable"])
        return {
            "success": True,
            "count": len(results),
            "results": results,
            "message": (
                f"Found {len(results)} MCP server(s), {installable} installable locally. "
                "To add one, open the MCP page (/mcp) in the WITS web UI, use "
                "'Discover servers', and click Install — installing runs the "
                "server's code on this machine, so it needs your confirmation."
            ),
        }

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": "search_mcp_tools",
            "description": (
                "Search the MCP registry for installable servers that add a new "
                "capability to WITS. Read-only; the user installs from the MCP page."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The capability to look for, e.g. 'send slack message'",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of servers to return",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        }
