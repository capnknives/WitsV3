"""MCP management routes for the WitsV3 web UI."""

import json
import logging
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from web.schemas import MCPRegistryInstall, MCPServerAdd, MCPToolInvoke

logger = logging.getLogger("WitsV3.WebUI")


def register_mcp_routes(app: FastAPI, system) -> None:
    """Register all MCP-related API routes on the FastAPI app."""

    def _mcp_config_path() -> Path:
        return Path(system.config.tool_system.mcp_tool_definitions_path)

    def _load_mcp_config() -> dict[str, Any]:
        path = _mcp_config_path()
        if not path.exists():
            return {"auto_connect": True, "servers": []}
        return json.loads(path.read_text(encoding="utf-8"))

    def _save_mcp_config(config: dict[str, Any]) -> None:
        path = _mcp_config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    def _mcp_adapter(create: bool = False):
        """The live MCP adapter, creating a registry on the system lazily."""
        registry = getattr(system, "mcp_registry", None)
        if registry is not None and registry.mcp_adapter is not None:
            return registry.mcp_adapter
        if not create:
            return None
        from core.enhanced_mcp_adapter import EnhancedMCPAdapter
        from tools.mcp_tool_registry import MCPToolRegistry

        registry = MCPToolRegistry(str(_mcp_config_path()))
        registry.config = _load_mcp_config()
        registry.mcp_adapter = EnhancedMCPAdapter(str(_mcp_config_path()))
        system.mcp_registry = registry  # system.shutdown() disconnects it
        return registry.mcp_adapter

    def _unregister_mcp_tools_for_server(server_name: str) -> int:
        """Remove mcp_* wrappers for a disconnected server from the live registry."""
        removed = 0
        for name, tool in list(system.tool_registry.tools.items()):
            if not name.startswith("mcp_"):
                continue
            server = getattr(tool, "server_name", None)
            if server == server_name:
                system.tool_registry.unregister_tool(name)
                removed += 1
        return removed

    def _register_mcp_tools_for_server(adapter, server_name: str) -> list:
        """Register connected MCP tools with the live tool registry."""
        from tools.mcp_tool import MCPTool as MCPToolWrapper

        registered = []
        tools = [t for t in adapter.tools.values() if t.server_name == server_name]
        for mcp_tool in tools:
            wrapper_name = f"mcp_{mcp_tool.name}"
            system.tool_registry.register_tool(
                MCPToolWrapper(
                    name=wrapper_name,
                    description=mcp_tool.description,
                    mcp_tool=mcp_tool,
                    mcp_adapter=adapter,
                )
            )
            registered.append(mcp_tool)
        return registered

    async def _connect_mcp_server_entry(name: str, entry: dict[str, Any]):
        """Connect one configured MCP server; returns (ok, tools_or_error)."""
        import asyncio

        from core.mcp_adapter import MCPServer, startup_timeout_for_command

        if "command" not in entry:
            return False, "server entry has no command"

        adapter = _mcp_adapter(create=True)
        if name in adapter.clients:
            await adapter.remove_server(name)
            _unregister_mcp_tools_for_server(name)

        command = entry["command"]
        cmd_list = command.split() if isinstance(command, str) else list(command)
        server = MCPServer(
            name=name,
            command=cmd_list,
            args=entry.get("args"),
            env=entry.get("env"),
            working_directory=entry.get("working_directory"),
        )
        connect_timeout = startup_timeout_for_command(cmd_list) + 15.0
        try:
            ok = await asyncio.wait_for(adapter.add_server(server), timeout=connect_timeout)
        except asyncio.TimeoutError:
            return False, (
                f"connection timed out ({connect_timeout:.0f}s). "
                "npx/uvx servers can take 1–2 minutes on first run — try Reconnect."
            )
        if not ok:
            return False, "connection failed — check logs/witsv3.log for details"

        tools = _register_mcp_tools_for_server(adapter, name)
        logger.info(f"MCP server connected: {name} ({len(tools)} tools)")
        return True, tools

    @app.get("/api/mcp/status")
    async def mcp_status():
        adapter = _mcp_adapter()
        config_servers = _load_mcp_config().get("servers", [])
        connected = list(adapter.clients.keys()) if adapter else []
        tool_count = len(adapter.tools) if adapter else 0
        return {
            "configured_servers": len(config_servers),
            "connected_servers": len(connected),
            "connected": connected,
            "tool_count": tool_count,
            "registry_url": system.config.tool_system.mcp_registry_url,
        }

    @app.get("/api/mcp/servers")
    async def mcp_servers():
        adapter = _mcp_adapter()
        connected = set(adapter.clients.keys()) if adapter else set()
        tools_by_server: dict[str, int] = {}
        if adapter:
            for tool in await adapter.list_available_tools():
                tools_by_server[tool.server_name] = tools_by_server.get(tool.server_name, 0) + 1
        servers = []
        for entry in _load_mcp_config().get("servers", []):
            name = entry.get("name", "unknown")
            servers.append(
                {
                    "name": name,
                    "command": entry.get("command"),
                    "working_directory": entry.get("working_directory"),
                    "connected": name in connected,
                    "tool_count": tools_by_server.get(name, 0),
                }
            )
        return {"servers": servers}

    @app.post("/api/mcp/servers")
    async def mcp_add_server(body: MCPServerAdd):
        name = body.name.strip()
        command = body.command.strip()
        if not name or not command:
            return JSONResponse({"detail": "name and command are required"}, status_code=400)

        config = _load_mcp_config()
        if any(s.get("name") == name for s in config.get("servers", [])):
            return JSONResponse({"detail": f"server '{name}' already exists"}, status_code=409)

        entry: dict[str, Any] = {"name": name, "command": command}
        if body.working_directory:
            entry["working_directory"] = body.working_directory
        if body.args:
            entry["args"] = body.args
        config.setdefault("servers", []).append(entry)
        _save_mcp_config(config)
        logger.info(f"MCP server added via web UI: {name}")
        return {"added": True, "server": entry}

    @app.delete("/api/mcp/servers/{name:path}")
    async def mcp_remove_server(name: str):
        adapter = _mcp_adapter()
        if adapter and name in adapter.clients:
            await adapter.remove_server(name)
        _unregister_mcp_tools_for_server(name)

        config = _load_mcp_config()
        before = len(config.get("servers", []))
        config["servers"] = [s for s in config.get("servers", []) if s.get("name") != name]
        if len(config["servers"]) == before:
            return JSONResponse({"detail": "unknown server"}, status_code=404)
        _save_mcp_config(config)
        return {"removed": True}

    @app.post("/api/mcp/servers/{name:path}/connect")
    async def mcp_connect_server(name: str):
        entry = next(
            (s for s in _load_mcp_config().get("servers", []) if s.get("name") == name), None
        )
        if entry is None:
            return JSONResponse({"detail": "unknown server"}, status_code=404)

        ok, payload = await _connect_mcp_server_entry(name, entry)
        if not ok:
            status = 409 if payload == "already connected" else 502
            return JSONResponse({"detail": payload}, status_code=status)

        tools = payload
        tool_list = [{"name": t.name, "description": t.description} for t in tools]
        result = {"connected": True, "tools": tool_list, "tool_count": len(tool_list)}
        if not tool_list:
            result["warning"] = (
                "Server handshake succeeded but no tools were listed. "
                "Try Reconnect — npx servers often need a second attempt after cache warm-up."
            )
        return result

    @app.post("/api/mcp/servers/{name:path}/reconnect")
    async def mcp_reconnect_server(name: str):
        """Disconnect (if needed) and connect again — fixes stale 0-tool sessions."""
        entry = next(
            (s for s in _load_mcp_config().get("servers", []) if s.get("name") == name), None
        )
        if entry is None:
            return JSONResponse({"detail": "unknown server"}, status_code=404)
        adapter = _mcp_adapter()
        if adapter and name in adapter.clients:
            await adapter.remove_server(name)
            _unregister_mcp_tools_for_server(name)
        ok, payload = await _connect_mcp_server_entry(name, entry)
        if not ok:
            return JSONResponse({"detail": payload}, status_code=502)
        tools = payload
        tool_list = [{"name": t.name, "description": t.description} for t in tools]
        return {
            "reconnected": True,
            "tools": tool_list,
            "tool_count": len(tool_list),
        }

    @app.post("/api/mcp/servers/{name:path}/disconnect")
    async def mcp_disconnect_server(name: str):
        adapter = _mcp_adapter()
        if not adapter or name not in adapter.clients:
            return JSONResponse({"detail": "server not connected"}, status_code=409)
        await adapter.remove_server(name)
        removed = _unregister_mcp_tools_for_server(name)
        logger.info(f"MCP server disconnected via web UI: {name} ({removed} tools unregistered)")
        return {"disconnected": True, "tools_removed": removed}

    @app.get("/api/mcp/servers/{name:path}/tools")
    async def mcp_server_tools(name: str):
        adapter = _mcp_adapter()
        if not adapter or name not in adapter.clients:
            return JSONResponse({"detail": "server not connected"}, status_code=409)
        tools = [t for t in await adapter.list_available_tools() if t.server_name == name]
        return {
            "tools": [
                {
                    "name": t.name,
                    "registered_name": f"mcp_{t.name}",
                    "description": t.description,
                    "input_schema": t.input_schema,
                }
                for t in tools
            ]
        }

    @app.get("/api/mcp/tools")
    async def mcp_all_tools():
        """All tools from connected MCP servers (for the playground)."""
        adapter = _mcp_adapter()
        if not adapter:
            return {"tools": []}
        return {
            "tools": [
                {
                    "name": t.name,
                    "registered_name": f"mcp_{t.name}",
                    "server": t.server_name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                }
                for t in await adapter.list_available_tools()
            ]
        }

    @app.post("/api/mcp/tools/{tool_name:path}/invoke")
    async def mcp_invoke_tool(tool_name: str, body: MCPToolInvoke):
        """Run a connected MCP tool from the web UI (human testing)."""
        adapter = _mcp_adapter()
        if not adapter:
            return JSONResponse({"detail": "no MCP servers connected"}, status_code=409)

        bare = tool_name.removeprefix("mcp_")
        tool = adapter.tools.get(bare)
        if tool is None:
            return JSONResponse({"detail": f"unknown MCP tool: {tool_name}"}, status_code=404)

        from core.schemas import ToolCall

        try:
            result = await adapter.call_tool(
                ToolCall(
                    call_id=f"web_ui_{uuid.uuid4().hex[:8]}",
                    tool_name=bare,
                    arguments=body.arguments,
                )
            )
        except Exception as e:
            return JSONResponse({"detail": str(e)}, status_code=502)

        if not result.success:
            return JSONResponse({"success": False, "error": result.error}, status_code=502)
        return {"success": True, "result": result.result}

    @app.get("/api/mcp/registry/search")
    async def mcp_registry_search(q: str = "", limit: int = 10):
        from core.mcp_registry_search import search_registry

        try:
            entries = await search_registry(
                q,
                limit=max(1, min(limit, 25)),
                registry_url=system.config.tool_system.mcp_registry_url,
            )
        except Exception as e:
            logger.warning(f"MCP registry search failed: {e}")
            return JSONResponse({"detail": f"registry search failed: {e}"}, status_code=502)

        existing = {s.get("name") for s in _load_mcp_config().get("servers", [])}
        for entry in entries:
            entry["already_added"] = entry["name"] in existing
        return {"results": entries}

    @app.post("/api/mcp/registry/install")
    async def mcp_registry_install(body: MCPRegistryInstall):
        name = body.name.strip()
        if not name or not body.command:
            return JSONResponse({"detail": "name and command are required"}, status_code=400)

        config = _load_mcp_config()
        if any(s.get("name") == name for s in config.get("servers", [])):
            return JSONResponse({"detail": f"server '{name}' already exists"}, status_code=409)

        entry: dict[str, Any] = {"name": name, "command": body.command, "source": "registry"}
        if body.working_directory:
            entry["working_directory"] = body.working_directory
        if body.env:
            # Drop blank values so we don't feed empty required vars to the server.
            env = {k: v for k, v in body.env.items() if v}
            if env:
                entry["env"] = env
        config.setdefault("servers", []).append(entry)
        _save_mcp_config(config)
        logger.info(f"MCP server installed from registry via web UI: {name}")

        connect_result = None
        if body.connect:
            ok, payload = await _connect_mcp_server_entry(name, entry)
            if ok:
                connect_result = {
                    "connected": True,
                    "tools": [{"name": t.name, "description": t.description} for t in payload],
                }
            else:
                connect_result = {"connected": False, "detail": payload}

        return {"installed": True, "server": entry, "connect": connect_result}
