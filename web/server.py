# web/server.py
"""
WitsV3 Web UI server.

FastAPI app streaming agent responses over SSE, with side-panel endpoints
for tools, memory search, and document RAG. Mobile-first frontend lives in
web/static (installable as a PWA on Android).

The app is built by create_app(system) where `system` is an initialized
WitsV3System (or any object exposing the same surface: config,
control_center, tool_registry, memory_manager, session_histories) — tests
pass a lightweight fake.
"""

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core.schemas import ConversationHistory

logger = logging.getLogger("WitsV3.WebUI")

STATIC_DIR = Path(__file__).parent / "static"

# Paths that never require auth (the shell page + PWA assets load before the
# user can enter a token; every /api/* call is protected).
PUBLIC_PATHS = {"/", "/personality", "/settings", "/mcp",
                "/manifest.webmanifest", "/icon.svg", "/app.js", "/style.css"}


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class SettingsUpdate(BaseModel):
    """Runtime-adjustable settings from the /settings page. All optional."""
    history_window: Optional[int] = None
    default_temperature: Optional[float] = None
    max_iterations: Optional[int] = None
    default_model: Optional[str] = None
    orchestrator_model: Optional[str] = None
    escalation_enabled: Optional[bool] = None
    escalation_model: Optional[str] = None
    escalation_max_tokens: Optional[int] = None


class MCPServerAdd(BaseModel):
    name: str
    command: str
    working_directory: Optional[str] = None
    args: Optional[list[str]] = None


class EscalationDecision(BaseModel):
    session_id: Optional[str] = None


class PersonalityAnswers(BaseModel):
    """Questionnaire answers from the /personality page. All optional —
    only the fields the user filled in are written to the overrides file."""
    identity_label: Optional[str] = None
    default_role: Optional[str] = None
    tone: Optional[str] = None
    language_level: Optional[str] = None
    verbosity: Optional[str] = None
    structure_preference: Optional[str] = None
    humor: Optional[str] = None
    default_persona: Optional[str] = None
    core_directives: Optional[list[str]] = None


def create_app(system) -> FastAPI:
    """Build the FastAPI app around an initialized WitsV3System."""
    app = FastAPI(title="WitsV3 Web UI", docs_url=None, redoc_url=None)
    web_token = os.getenv("WITSV3_WEB_TOKEN", "")
    require_auth = bool(system.config.web_ui.require_auth and web_token)

    if system.config.web_ui.require_auth and not web_token:
        logger.warning(
            "web_ui.require_auth is true but WITSV3_WEB_TOKEN is not set - "
            "the web UI is UNPROTECTED on your network. Add WITSV3_WEB_TOKEN to .env."
        )

    # ------------------------------------------------------------- auth
    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        if require_auth and request.url.path.startswith("/api/"):
            header = request.headers.get("authorization", "")
            token = header.removeprefix("Bearer ").strip() if header.startswith("Bearer ") else ""
            if not token:
                token = request.query_params.get("token", "")
            if token != web_token:
                return JSONResponse({"detail": "unauthorized"}, status_code=401)
        return await call_next(request)

    # ------------------------------------------------------------- pages
    @app.get("/")
    async def index():
        # no-store so browsers always pick up frontend updates
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/personality")
    async def personality_page():
        return FileResponse(STATIC_DIR / "personality.html", headers={"Cache-Control": "no-store"})

    @app.get("/settings")
    async def settings_page():
        return FileResponse(STATIC_DIR / "settings.html", headers={"Cache-Control": "no-store"})

    @app.get("/mcp")
    async def mcp_page():
        return FileResponse(STATIC_DIR / "mcp.html", headers={"Cache-Control": "no-store"})

    @app.get("/manifest.webmanifest")
    async def manifest():
        return FileResponse(STATIC_DIR / "manifest.webmanifest", media_type="application/manifest+json")

    @app.get("/icon.svg")
    async def icon():
        return FileResponse(STATIC_DIR / "icon.svg", media_type="image/svg+xml")

    @app.get("/app.js")
    async def app_js():
        return FileResponse(STATIC_DIR / "app.js", media_type="text/javascript",
                            headers={"Cache-Control": "no-cache"})

    @app.get("/style.css")
    async def style_css():
        return FileResponse(STATIC_DIR / "style.css", media_type="text/css",
                            headers={"Cache-Control": "no-cache"})

    # ------------------------------------------------------------- chat
    @app.post("/api/chat")
    async def chat(body: ChatRequest):
        session_id = body.session_id or str(uuid.uuid4())

        if session_id not in system.session_histories:
            system.session_histories[session_id] = ConversationHistory(session_id=session_id)
        conversation = system.session_histories[session_id]
        conversation.add_message("user", body.message)

        async def event_stream():
            def sse(event: str, data: Dict[str, Any]) -> str:
                return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

            yield sse("session", {"session_id": session_id})
            result_parts = []
            try:
                async for stream_data in system.control_center.run(
                    user_input=body.message,
                    conversation_history=conversation,
                    session_id=session_id,
                ):
                    payload = {
                        "type": stream_data.type,
                        "content": stream_data.content,
                        "source": stream_data.source,
                    }
                    yield sse("stream", payload)
                    if stream_data.type in ("result", "error"):
                        result_parts.append(stream_data.content)

                final_text = "\n".join(result_parts) or "(no response)"
                conversation.add_message("assistant", final_text)
                yield sse("done", {"final": final_text})

            except Exception as e:
                logger.error(f"Chat stream failed: {e}", exc_info=True)
                error_msg = f"Error processing request: {e}"
                conversation.add_message("assistant", error_msg)
                yield sse("stream", {"type": "error", "content": error_msg, "source": "web"})
                yield sse("done", {"final": error_msg})

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ------------------------------------------------------------- info
    @app.get("/api/status")
    async def status():
        cfg = system.config
        return {
            "project": cfg.project_name,
            "version": cfg.version,
            "models": {
                "default": cfg.ollama_settings.default_model,
                "orchestrator": cfg.ollama_settings.orchestrator_model,
                "embedding": cfg.ollama_settings.embedding_model,
            },
            "tool_count": len(system.tool_registry.tools),
            "active_sessions": len(system.session_histories),
        }

    @app.get("/api/tools")
    async def tools():
        return {
            "tools": [
                {"name": t.name, "description": t.description}
                for t in system.tool_registry.tools.values()
            ]
        }

    @app.get("/api/memory/search")
    async def memory_search(q: str, limit: int = 5):
        segments = await system.memory_manager.search_memory(query_text=q, limit=limit)
        return {
            "results": [
                {
                    "type": s.type,
                    "source": s.source,
                    "text": (s.content.text or s.content.tool_output or "")[:500],
                    "relevance": round(s.relevance_score or 0.0, 3),
                }
                for s in segments
            ]
        }

    # --------------------------------------------------------- settings
    async def _list_ollama_models() -> list:
        """Best-effort list of locally available Ollama models."""
        import asyncio
        import urllib.request

        def fetch():
            url = system.config.ollama_settings.url.rstrip("/") + "/api/tags"
            with urllib.request.urlopen(url, timeout=3) as resp:
                return json.loads(resp.read())

        try:
            data = await asyncio.to_thread(fetch)
            return sorted(m["name"] for m in data.get("models", []))
        except Exception:
            return []

    @app.get("/api/settings")
    async def get_settings():
        from core.escalation import MODEL_PRICES, get_escalation_manager

        cfg = system.config
        return {
            "history_window": cfg.agents.history_window,
            "default_temperature": cfg.agents.default_temperature,
            "max_iterations": cfg.agents.max_iterations,
            "default_model": cfg.ollama_settings.default_model,
            "orchestrator_model": cfg.ollama_settings.orchestrator_model,
            "available_models": await _list_ollama_models(),
            "escalation_enabled": cfg.escalation.enabled,
            "escalation_model": cfg.escalation.model,
            "escalation_max_tokens": cfg.escalation.max_tokens,
            "escalation_models": {
                name: {"input_per_mtok": p[0], "output_per_mtok": p[1]}
                for name, p in MODEL_PRICES.items()
            },
            "anthropic_key_configured": get_escalation_manager().api_key_configured(),
        }

    @app.post("/api/settings")
    async def save_settings(update: SettingsUpdate):
        from core.config import save_local_overrides

        cfg = system.config
        overrides: Dict[str, Any] = {}

        def apply(section, field, value, override_section):
            if value is None:
                return
            setattr(section, field, value)  # validates via pydantic
            overrides.setdefault(override_section, {})[field] = getattr(section, field)

        try:
            apply(cfg.agents, "history_window", update.history_window, "agents")
            apply(cfg.agents, "default_temperature", update.default_temperature, "agents")
            apply(cfg.agents, "max_iterations", update.max_iterations, "agents")
            apply(cfg.ollama_settings, "default_model", update.default_model, "ollama_settings")
            apply(cfg.ollama_settings, "orchestrator_model", update.orchestrator_model, "ollama_settings")
            apply(cfg.escalation, "enabled", update.escalation_enabled, "escalation")
            apply(cfg.escalation, "model", update.escalation_model, "escalation")
            apply(cfg.escalation, "max_tokens", update.escalation_max_tokens, "escalation")
        except Exception as e:
            return JSONResponse({"detail": f"invalid value: {e}"}, status_code=400)

        if not overrides:
            return JSONResponse({"detail": "no settings provided"}, status_code=400)

        path = save_local_overrides(overrides)
        logger.info(f"Settings updated via web UI: {list(overrides)} -> {path}")
        return {"saved": True, "applied_live": True}

    # ------------------------------------------------------ escalations
    @app.get("/api/escalations")
    async def list_escalations():
        from core.escalation import get_escalation_manager
        return {"requests": get_escalation_manager().list()}

    @app.post("/api/escalations/{request_id}/approve")
    async def approve_escalation(request_id: str, decision: Optional[EscalationDecision] = None):
        from core.escalation import get_escalation_manager

        manager = get_escalation_manager()
        try:
            request = await manager.approve(request_id)
        except KeyError:
            return JSONResponse({"detail": "unknown escalation"}, status_code=404)
        except ValueError as e:
            return JSONResponse({"detail": str(e)}, status_code=409)

        # Put Claude's answer into the conversation so WITS sees it next turn
        session_id = decision.session_id if decision else None
        if request.status == "answered" and session_id in system.session_histories:
            system.session_histories[session_id].add_message(
                "assistant",
                f"[Claude ({request.model}) answered the escalated question]\n{request.answer}",
            )
        return request.to_dict()

    @app.post("/api/escalations/{request_id}/deny")
    async def deny_escalation(request_id: str):
        from core.escalation import get_escalation_manager

        if not get_escalation_manager().deny(request_id):
            return JSONResponse({"detail": "not a pending escalation"}, status_code=404)
        return {"denied": True}

    # -------------------------------------------------------------- mcp
    def _mcp_config_path() -> Path:
        return Path(system.config.tool_system.mcp_tool_definitions_path)

    def _load_mcp_config() -> Dict[str, Any]:
        path = _mcp_config_path()
        if not path.exists():
            return {"auto_connect": True, "servers": []}
        return json.loads(path.read_text(encoding="utf-8"))

    def _save_mcp_config(config: Dict[str, Any]) -> None:
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

    @app.get("/api/mcp/servers")
    async def mcp_servers():
        adapter = _mcp_adapter()
        connected = set(adapter.clients.keys()) if adapter else set()
        tools_by_server: Dict[str, int] = {}
        if adapter:
            for tool in await adapter.list_available_tools():
                tools_by_server[tool.server_name] = tools_by_server.get(tool.server_name, 0) + 1
        servers = []
        for entry in _load_mcp_config().get("servers", []):
            name = entry.get("name", "unknown")
            servers.append({
                "name": name,
                "command": entry.get("command"),
                "working_directory": entry.get("working_directory"),
                "connected": name in connected,
                "tool_count": tools_by_server.get(name, 0),
            })
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

        entry: Dict[str, Any] = {"name": name, "command": command}
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

        config = _load_mcp_config()
        before = len(config.get("servers", []))
        config["servers"] = [s for s in config.get("servers", []) if s.get("name") != name]
        if len(config["servers"]) == before:
            return JSONResponse({"detail": "unknown server"}, status_code=404)
        _save_mcp_config(config)
        return {"removed": True}

    @app.post("/api/mcp/servers/{name:path}/connect")
    async def mcp_connect_server(name: str):
        import asyncio
        from core.mcp_adapter import MCPServer
        from tools.mcp_tool import MCPTool as MCPToolWrapper

        entry = next((s for s in _load_mcp_config().get("servers", []) if s.get("name") == name), None)
        if entry is None:
            return JSONResponse({"detail": "unknown server"}, status_code=404)
        if "command" not in entry:
            return JSONResponse({"detail": "server entry has no command"}, status_code=400)

        adapter = _mcp_adapter(create=True)
        if name in adapter.clients:
            return JSONResponse({"detail": "already connected"}, status_code=409)

        command = entry["command"]
        server = MCPServer(
            name=name,
            command=command.split() if isinstance(command, str) else command,
            args=entry.get("args"),
            env=entry.get("env"),
            working_directory=entry.get("working_directory"),
        )
        try:
            ok = await asyncio.wait_for(adapter.add_server(server), timeout=45)
        except asyncio.TimeoutError:
            return JSONResponse({"detail": "connection timed out (45s)"}, status_code=502)
        if not ok:
            return JSONResponse({"detail": "connection failed - check the server log"}, status_code=502)

        # Register this server's tools with the live tool registry
        tools = [t for t in await adapter.list_available_tools() if t.server_name == name]
        for mcp_tool in tools:
            system.tool_registry.register_tool(MCPToolWrapper(
                name=f"mcp_{mcp_tool.name}",
                description=mcp_tool.description,
                mcp_tool=mcp_tool,
                mcp_adapter=adapter,
            ))
        logger.info(f"MCP server connected via web UI: {name} ({len(tools)} tools)")
        return {
            "connected": True,
            "tools": [{"name": t.name, "description": t.description} for t in tools],
        }

    @app.get("/api/mcp/servers/{name:path}/tools")
    async def mcp_server_tools(name: str):
        adapter = _mcp_adapter()
        if not adapter or name not in adapter.clients:
            return JSONResponse({"detail": "server not connected"}, status_code=409)
        tools = [t for t in await adapter.list_available_tools() if t.server_name == name]
        return {"tools": [{"name": t.name, "description": t.description} for t in tools]}

    # ------------------------------------------------------ personality
    @app.get("/api/personality")
    async def get_personality():
        from core.personality_manager import PersonalityManager

        pm = PersonalityManager(config=system.config)
        profile = pm.personality_profile or {}
        comm = profile.get("communication", {})
        personas = [
            r.get("name") for r in profile.get("persona_layers", {}).get("available_roles", [])
            if isinstance(r, dict) and r.get("name")
        ]
        return {
            "identity_label": profile.get("identity_label", ""),
            "default_role": profile.get("default_role", ""),
            "tone": comm.get("tone", ""),
            "language_level": comm.get("language_level", ""),
            "verbosity": comm.get("verbosity", ""),
            "structure_preference": comm.get("structure_preference", ""),
            "humor": comm.get("humor", ""),
            "default_persona": profile.get("persona_layers", {}).get("default_persona", ""),
            "core_directives": profile.get("core_directives", []),
            "available_personas": personas,
            "system_prompt": pm.get_system_prompt(),
        }

    @app.post("/api/personality")
    async def save_personality(answers: PersonalityAnswers):
        import yaml
        from core.personality_manager import PersonalityManager, reload_personality_manager

        overrides: Dict[str, Any] = {}
        comm: Dict[str, Any] = {}

        def clean(value: Optional[str]) -> Optional[str]:
            return value.strip() if value and value.strip() else None

        if clean(answers.identity_label):
            overrides["identity_label"] = clean(answers.identity_label)
        if clean(answers.default_role):
            overrides["default_role"] = clean(answers.default_role)
        for field in ("tone", "language_level", "verbosity", "structure_preference", "humor"):
            value = clean(getattr(answers, field))
            if value:
                comm[field] = value
        if comm:
            overrides["communication"] = comm
        if clean(answers.default_persona):
            overrides["persona_layers"] = {"default_persona": clean(answers.default_persona)}
        if answers.core_directives is not None:
            directives = [d.strip() for d in answers.core_directives if d.strip()]
            if directives:
                overrides["core_directives"] = directives

        if not overrides:
            return JSONResponse({"detail": "no answers provided"}, status_code=400)

        overrides_path = Path(PersonalityManager.overrides_path_for(system.config.personality.profile_path))
        overrides_path.parent.mkdir(parents=True, exist_ok=True)
        header = (
            "# Written by the WITS web UI personality questionnaire (/personality).\n"
            "# Merged over config/wits_personality.yaml at load - delete this file\n"
            "# to revert to the base profile.\n"
        )
        overrides_path.write_text(header + yaml.safe_dump(overrides, sort_keys=False, allow_unicode=True),
                                  encoding="utf-8")

        # Apply immediately: agents fetch the personality manager per call
        pm = reload_personality_manager(config=system.config)
        logger.info(f"Personality overrides saved to {overrides_path}")
        return {"saved": True, "system_prompt": pm.get_system_prompt()}

    @app.delete("/api/personality")
    async def reset_personality():
        from core.personality_manager import PersonalityManager, reload_personality_manager

        overrides_path = Path(PersonalityManager.overrides_path_for(system.config.personality.profile_path))
        existed = overrides_path.exists()
        if existed:
            overrides_path.unlink()
        pm = reload_personality_manager(config=system.config)
        return {"reset": existed, "system_prompt": pm.get_system_prompt()}

    # -------------------------------------------------------- documents
    @app.get("/api/documents")
    async def documents():
        docs_dir = Path(system.config.document_rag.documents_path)
        chunks_by_file: Dict[str, int] = {}
        segments = await system.memory_manager.get_recent_memory(
            limit=1_000_000, filter_dict={"type": "DOCUMENT_CHUNK"}
        )
        for seg in segments:
            fp = seg.metadata.get("file_path")
            if fp:
                chunks_by_file[fp] = chunks_by_file.get(fp, 0) + 1

        files = []
        if docs_dir.exists():
            for path in sorted(docs_dir.rglob("*")):
                if path.is_file() and path.name != ".gitkeep":
                    rel = path.relative_to(docs_dir).as_posix()
                    files.append({
                        "name": rel,
                        "size": path.stat().st_size,
                        "chunks": chunks_by_file.get(rel, 0),
                    })
        return {"files": files}

    @app.post("/api/documents/upload")
    async def upload_document(file: UploadFile = File(...)):
        docs_dir = Path(system.config.document_rag.documents_path)
        docs_dir.mkdir(parents=True, exist_ok=True)

        # Keep only the file name — no path traversal from client input
        safe_name = Path(file.filename or "upload.txt").name
        target = docs_dir / safe_name
        target.write_bytes(await file.read())

        ingest_tool = system.tool_registry.get_tool("ingest_documents")
        summary = await ingest_tool.execute() if ingest_tool else {"success": False, "error": "ingest tool unavailable"}
        return {"saved": safe_name, "ingest": summary}

    return app
