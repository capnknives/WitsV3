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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from core.schemas import ConversationHistory, StreamData
from web.owner_controls import (
    owner_auth_detail,
    owner_auth_failure,
    owner_command_reply,
    parse_owner_command,
    schedule_owner_action,
)
from web.routes_mcp import register_mcp_routes
from web.routes_personality import register_personality_routes
from web.schemas import (
    ChatRequest,
    DocumentDeleteRequest,
    EscalationDecision,
    ExportRequest,
    MemoryPruneRequest,
    OwnerControlRequest,
    SettingsUpdate,
)
from web.user_errors import format_chat_error

logger = logging.getLogger("WitsV3.WebUI")

STATIC_DIR = Path(__file__).parent / "static"

# Paths that never require auth (the shell page + PWA assets load before the
# user can enter a token; every /api/* call is protected).
PUBLIC_PATHS = {
    "/",
    "/personality",
    "/settings",
    "/mcp",
    "/manifest.webmanifest",
    "/icon.svg",
    "/app.js",
    "/style.css",
    "/mcp.css",
}


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
        return FileResponse(
            STATIC_DIR / "manifest.webmanifest", media_type="application/manifest+json"
        )

    @app.get("/icon.svg")
    async def icon():
        return FileResponse(STATIC_DIR / "icon.svg", media_type="image/svg+xml")

    @app.get("/app.js")
    async def app_js():
        return FileResponse(
            STATIC_DIR / "app.js",
            media_type="text/javascript",
            headers={"Cache-Control": "no-cache"},
        )

    @app.get("/style.css")
    async def style_css():
        return FileResponse(
            STATIC_DIR / "style.css", media_type="text/css", headers={"Cache-Control": "no-cache"}
        )

    @app.get("/mcp.css")
    async def mcp_css():
        return FileResponse(
            STATIC_DIR / "mcp.css", media_type="text/css", headers={"Cache-Control": "no-cache"}
        )

    # ------------------------------------------------------------- chat
    def _stream_payload(stream_data: StreamData) -> dict[str, Any]:
        """Serialize a StreamData event for SSE, with friendly error copy."""
        payload: dict[str, Any] = {
            "type": stream_data.type,
            "content": stream_data.content,
            "source": stream_data.source,
        }
        if stream_data.error_details:
            payload["error_details"] = stream_data.error_details
        if stream_data.type == "error":
            combined = stream_data.content
            if stream_data.error_details:
                combined = f"{combined}\n{stream_data.error_details}"
            fmt = format_chat_error(combined, system.config.ollama_settings.url)
            if fmt["code"] != "generic":
                payload["content"] = fmt["message"]
            payload["user_error"] = fmt
        return payload

    @app.post("/api/chat")
    async def chat(body: ChatRequest, request: Request):
        session_id = body.session_id or str(uuid.uuid4())

        if session_id not in system.session_histories:
            system.session_histories[session_id] = ConversationHistory(session_id=session_id)
        conversation = system.session_histories[session_id]
        conversation.add_message("user", body.message)

        owner_action = parse_owner_command(body.message)

        async def event_stream():
            def sse(event: str, data: dict[str, Any]) -> str:
                return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

            yield sse("session", {"session_id": session_id})

            # Exact slash commands bypass the agent so the owner can kill a hung UI.
            if owner_action:
                denied = owner_auth_detail(request)
                if denied is not None:
                    conversation.add_message("assistant", denied)
                    yield sse(
                        "stream",
                        {"type": "error", "content": denied, "source": "web"},
                    )
                    yield sse("done", {"final": denied})
                    return

                reply = owner_command_reply(owner_action)
                conversation.add_message("assistant", reply)
                yield sse(
                    "stream",
                    {
                        "type": "result",
                        "content": reply,
                        "source": "web",
                        "owner_action": owner_action,
                    },
                )
                yield sse("done", {"final": reply, "owner_action": owner_action})
                try:
                    schedule_owner_action(owner_action, delay_seconds=1.0)
                except Exception as e:
                    logger.error(f"Failed to schedule owner {owner_action}: {e}", exc_info=True)
                return

            result_parts = []
            try:
                async for stream_data in system.control_center.run(
                    user_input=body.message,
                    conversation_history=conversation,
                    session_id=session_id,
                ):
                    payload = _stream_payload(stream_data)
                    yield sse("stream", payload)
                    if stream_data.type in ("result", "error"):
                        result_parts.append(payload["content"])

                final_text = "\n".join(result_parts) or "(no response)"
                conversation.add_message("assistant", final_text)
                yield sse("done", {"final": final_text})

            except Exception as e:
                logger.error(f"Chat stream failed: {e}", exc_info=True)
                fmt = format_chat_error(e, system.config.ollama_settings.url)
                error_msg = fmt["message"]
                if fmt.get("hint"):
                    error_msg = f"{fmt['message']}\n\n{fmt['hint']}"
                conversation.add_message("assistant", error_msg)
                yield sse(
                    "stream",
                    {
                        "type": "error",
                        "content": fmt["message"],
                        "source": "web",
                        "user_error": fmt,
                    },
                )
                yield sse("done", {"final": error_msg})

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    def _format_session_export(conversation: ConversationHistory) -> str:
        if not conversation.messages:
            return "Conversation history is empty."
        lines = []
        for msg in conversation.messages:
            lines.append(f"{msg.role.upper()}: {msg.content}")
        return "\n\n".join(lines)

    @app.post("/api/export")
    async def export_conversation(body: ExportRequest):
        """Write the current session transcript to exports/ (one-click export)."""
        session_id = body.session_id
        if not session_id or session_id not in system.session_histories:
            return JSONResponse({"detail": "no active session"}, status_code=400)

        conversation = system.session_histories[session_id]
        content = _format_session_export(conversation)
        if not content.strip():
            return JSONResponse({"detail": "conversation is empty"}, status_code=400)

        export_dir = Path("exports")
        export_dir.mkdir(parents=True, exist_ok=True)
        if body.file_path:
            rel = Path(body.file_path.replace("\\", "/"))
            if rel.is_absolute() or ".." in rel.parts or len(rel.parts) != 1:
                return JSONResponse({"detail": "invalid file_path"}, status_code=400)
            out_path = export_dir / rel.name
        else:
            stamp = uuid.uuid4().hex[:8]
            out_path = export_dir / f"chat_export_{stamp}.txt"

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8")

        return {
            "success": True,
            "file_path": str(out_path).replace("\\", "/"),
            "message_count": len(conversation.messages),
            "message": f"Exported {len(conversation.messages)} messages to {out_path}",
        }

    # ---------------------------------------------------- owner controls
    @app.post("/api/owner/shutdown")
    async def owner_shutdown(request: Request, body: OwnerControlRequest | None = None):
        """Force-stop the web process. Requires WITSV3_WEB_TOKEN + confirm SHUTDOWN."""
        denied = owner_auth_failure(request)
        if denied is not None:
            return denied
        confirm = (body.confirm if body else "") or ""
        if confirm.strip().upper() != "SHUTDOWN":
            return JSONResponse(
                {"detail": "missing/incorrect confirm (type SHUTDOWN)"},
                status_code=400,
            )
        delay = body.delay_seconds if body and body.delay_seconds is not None else 1.0
        return schedule_owner_action("shutdown", delay_seconds=delay)

    @app.post("/api/owner/restart")
    async def owner_restart(request: Request, body: OwnerControlRequest | None = None):
        """Relaunch the web process. Requires WITSV3_WEB_TOKEN + confirm RESTART."""
        denied = owner_auth_failure(request)
        if denied is not None:
            return denied
        confirm = (body.confirm if body else "") or ""
        if confirm.strip().upper() != "RESTART":
            return JSONResponse(
                {"detail": "missing/incorrect confirm (type RESTART)"},
                status_code=400,
            )
        delay = body.delay_seconds if body and body.delay_seconds is not None else 1.0
        return schedule_owner_action("restart", delay_seconds=delay)

    # ------------------------------------------------------------- info
    @app.get("/api/status")
    async def status():
        cfg = system.config
        ollama_available = None
        llm = getattr(system, "llm_interface", None)
        if llm is not None and hasattr(llm, "is_service_available"):
            try:
                ollama_available = await llm.is_service_available()
            except Exception:
                ollama_available = False
        return {
            "project": cfg.project_name,
            "version": cfg.version,
            "models": {
                "default": cfg.ollama_settings.default_model,
                "orchestrator": cfg.ollama_settings.orchestrator_model,
                "embedding": cfg.ollama_settings.embedding_model,
            },
            "ollama": {
                "url": cfg.ollama_settings.url,
                "available": ollama_available,
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

    @app.get("/api/memory/recent")
    async def memory_recent(
        limit: int = 20,
        offset: int = 0,
        segment_type: str | None = None,
        source: str | None = None,
    ):
        """
        List recent memory segments (recency-sorted, newest first).

        offset is counted from the newest segment (client-side "pagination").
        """
        limit = max(1, min(limit, 100))
        offset = max(0, offset)

        filter_dict: dict[str, Any] = {}
        if segment_type:
            filter_dict["type"] = segment_type
        if source:
            filter_dict["source"] = source

        fetch_limit = min(limit + offset + 1, 1000)  # cap to avoid abuse
        segments = await system.memory_manager.get_recent_memory(
            limit=fetch_limit, filter_dict=filter_dict or None
        )

        items = segments[offset : offset + limit]

        def preview(seg: Any) -> str:
            content = getattr(seg, "content", None)
            if content is None:
                return ""
            return (getattr(content, "text", None) or getattr(content, "tool_output", None) or "")[
                :500
            ]

        results = [
            {
                "id": getattr(s, "id", None),
                "timestamp": getattr(getattr(s, "timestamp", None), "isoformat", lambda: None)(),
                "type": getattr(s, "type", None),
                "source": getattr(s, "source", None),
                "tool_name": (
                    getattr(getattr(s, "content", None), "tool_name", None)
                    or getattr(s, "metadata", {}).get("tool_name")
                ),
                "text": preview(s),
            }
            for s in items
        ]

        return {
            "results": results,
            "offset": offset,
            "limit": limit,
            "has_more": len(segments) > offset + limit,
        }

    @app.post("/api/memory/prune")
    async def memory_prune(body: MemoryPruneRequest):
        # Confirm safeguard to prevent accidental wipes from the UI.
        if body.confirm != "PRUNE":
            return JSONResponse(
                {"detail": "missing/incorrect confirm (type PRUNE)"}, status_code=400
            )

        if not body.filter_dict:
            return JSONResponse({"detail": "filter_dict must not be empty"}, status_code=400)

        removed = await system.memory_manager.delete_segments(body.filter_dict)
        return {"success": True, "removed": removed}

    @app.get("/api/search/providers")
    async def search_providers():
        """Which web search providers are configured (no secrets exposed)."""
        ws = system.config.web_search
        return {
            "provider_mode": ws.provider,
            "tavily_configured": bool(ws.tavily_api_key or os.getenv("TAVILY_API_KEY")),
            "brave_configured": bool(ws.brave_api_key or os.getenv("BRAVE_SEARCH_API_KEY")),
            "fallback": "duckduckgo",
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
        mr = cfg.model_routing
        return {
            "history_window": cfg.agents.history_window,
            "default_temperature": cfg.agents.default_temperature,
            "max_iterations": cfg.agents.max_iterations,
            "default_model": cfg.ollama_settings.default_model,
            "orchestrator_model": cfg.ollama_settings.orchestrator_model,
            "available_models": await _list_ollama_models(),
            "model_routing": {
                "enabled": mr.enabled,
                "trivial_model": mr.trivial_model,
                "code_model": mr.code_model,
                "complex_model": mr.complex_model,
                "trivial_max_chars": mr.trivial_max_chars,
            },
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
        overrides: dict[str, Any] = {}

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
            apply(
                cfg.ollama_settings,
                "orchestrator_model",
                update.orchestrator_model,
                "ollama_settings",
            )
            apply(cfg.model_routing, "enabled", update.routing_enabled, "model_routing")
            apply(cfg.model_routing, "trivial_model", update.routing_trivial_model, "model_routing")
            apply(cfg.model_routing, "code_model", update.routing_code_model, "model_routing")
            apply(cfg.model_routing, "complex_model", update.routing_complex_model, "model_routing")
            apply(
                cfg.model_routing,
                "trivial_max_chars",
                update.routing_trivial_max_chars,
                "model_routing",
            )
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
    async def approve_escalation(request_id: str, decision: EscalationDecision | None = None):
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

    # -------------------------------------------------------- documents
    @app.get("/api/documents")
    async def documents():
        docs_dir = Path(system.config.document_rag.documents_path)
        chunks_by_file: dict[str, int] = {}
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
                    stat = path.stat()
                    files.append(
                        {
                            "name": rel,
                            "size": stat.st_size,
                            "chunks": chunks_by_file.get(rel, 0),
                            "ext": path.suffix.lower().lstrip("."),
                            "modified": datetime.fromtimestamp(
                                stat.st_mtime, tz=timezone.utc
                            ).isoformat(),
                        }
                    )
        total_chunks = sum(f["chunks"] for f in files)
        return {"files": files, "count": len(files), "total_chunks": total_chunks}

    @app.post("/api/documents/upload")
    async def upload_document(file: UploadFile = File(...)):
        docs_dir = Path(system.config.document_rag.documents_path)
        docs_dir.mkdir(parents=True, exist_ok=True)

        # Keep only the file name — no path traversal from client input
        safe_name = Path(file.filename or "upload.txt").name
        target = docs_dir / safe_name
        target.write_bytes(await file.read())

        ingest_tool = system.tool_registry.get_tool("ingest_documents")
        summary = (
            await ingest_tool.execute()
            if ingest_tool
            else {"success": False, "error": "ingest tool unavailable"}
        )
        return {"saved": safe_name, "ingest": summary}

    @app.post("/api/documents/delete")
    async def delete_document(body: DocumentDeleteRequest):
        docs_dir = Path(system.config.document_rag.documents_path).resolve()
        # Resolve within docs_dir; reject traversal or the dir itself.
        target = (docs_dir / body.name).resolve()
        if not target.is_relative_to(docs_dir) or target == docs_dir:
            return JSONResponse({"detail": "invalid document name"}, status_code=400)

        rel = target.relative_to(docs_dir).as_posix()
        existed = target.is_file()
        if existed:
            target.unlink()

        removed = await system.memory_manager.delete_segments(
            {"type": "DOCUMENT_CHUNK", "file_path": rel}
        )

        if not existed and not removed:
            return JSONResponse({"detail": "document not found"}, status_code=404)

        return {"deleted": rel, "removed_chunks": removed, "file_removed": existed}

    @app.post("/api/documents/reindex")
    async def reindex_documents():
        ingest_tool = system.tool_registry.get_tool("ingest_documents")
        summary = (
            await ingest_tool.execute()
            if ingest_tool
            else {"success": False, "error": "ingest tool unavailable"}
        )
        return {"ingest": summary}

    register_mcp_routes(app, system)
    register_personality_routes(app, system)

    return app
