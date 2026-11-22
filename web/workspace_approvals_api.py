"""Owner routes and Inbox page for workspace approvals (contract sections 5, 6 and 8).

``register(app)`` is called by ``web.workspace_routes.register_workspace``. Every route that
returns or changes data depends on ``require_owner``. The page and its script are static and
carry no data, so they are served without a token (the script sends the owner's token itself).
"""

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict

from core.workspace import approvals
from core.workspace.store import WorkspaceError
from core.workspace.text import clean_text
from web.workspace_routes import get_db, require_owner

logger = logging.getLogger("WitsV3.Workspace.ApprovalsAPI")

NOTE_MAX = 2000
_STATIC_DIR = Path(__file__).parent / "static"
# Reason: the Inbox must never be served stale after an update, and owner pages are not cached.
_NO_STORE = {"Cache-Control": "no-store"}


class ResolveBody(BaseModel):
    """JSON body for ``POST /api/workspace/approvals/{id}/resolve``."""

    model_config = ConfigDict(extra="forbid")

    choice: str
    note: str | None = None


def _error(exc: WorkspaceError) -> HTTPException:
    return HTTPException(status_code=400, detail={"code": exc.code, "message": exc.message})


async def _list_pending() -> dict:
    return {"approvals": await approvals.list_pending(get_db())}


async def _resolve(approval_id: str, body: ResolveBody) -> dict:
    try:
        # Reason: the note is owner text that later agents may read, so it gets the same
        # cleaning as agent text (section 8); empty after cleaning means no note.
        note = None if body.note is None else clean_text(body.note, NOTE_MAX, required=False)
        result = await approvals.resolve(get_db(), approval_id, body.choice, note or None)
    except WorkspaceError as exc:
        raise _error(exc) from None
    if result is None:
        raise HTTPException(
            status_code=409, detail="approval is missing, already resolved or expired"
        )
    return result


async def _inbox_page() -> FileResponse:
    return FileResponse(_STATIC_DIR / "inbox.html", headers=_NO_STORE)


async def _inbox_script() -> FileResponse:
    return FileResponse(
        _STATIC_DIR / "inbox.js", media_type="application/javascript", headers=_NO_STORE
    )


async def _inbox_link_script() -> FileResponse:
    # Reason: the main page always loads this script; it is only served when the workspace is on,
    # so a 404 (workspace off) leaves the link hidden.
    return FileResponse(
        _STATIC_DIR / "inbox-link.js", media_type="application/javascript", headers=_NO_STORE
    )


def register(app: FastAPI) -> None:
    """Add the owner approval routes and the Inbox page to the app.

    Args:
        app: The FastAPI application.
    """
    router = APIRouter(prefix="/api/workspace/approvals", dependencies=[Depends(require_owner)])
    router.add_api_route("", _list_pending, methods=["GET"])
    router.add_api_route("/{approval_id}/resolve", _resolve, methods=["POST"])
    app.include_router(router)
    app.add_api_route("/inbox", _inbox_page, methods=["GET"], include_in_schema=False)
    app.add_api_route("/inbox.js", _inbox_script, methods=["GET"], include_in_schema=False)
    app.add_api_route(
        "/inbox-link.js", _inbox_link_script, methods=["GET"], include_in_schema=False
    )
    logger.info("workspace approval routes registered")
