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
PUBLIC_PATHS = {"/", "/manifest.webmanifest", "/icon.svg", "/app.js", "/style.css"}


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


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
