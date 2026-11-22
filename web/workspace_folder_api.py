"""Owner routes for folder grants, and the live-grants hook (folder access contract, section 5).

``register(app)`` is called by ``web.workspace_routes.register_workspace``, which only runs when
the workspace is enabled. It adds the owner's list and revoke routes and a start-up context that
points ``path_access`` at the live grants in the workspace database; with the workspace off this
module never loads and ``path_access`` keeps its "no grants" provider.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, FastAPI, HTTPException

from core import runtime_paths
from core.workspace import folder_access
from web import workspace_routes
from web.workspace_routes import add_lifespan, get_db, require_owner

logger = logging.getLogger("WitsV3.Workspace.FolderAPI")


async def _list_grants() -> dict:
    base = workspace_routes.load_config().folders.base
    base_folders = [
        {"name": "inbox", "path": str(runtime_paths.inbox_dir(base)), "mode": "index"},
        {"name": "documents", "path": str(runtime_paths.documents_dir(base)), "mode": "index"},
        {"name": "outbox", "path": str(runtime_paths.outbox_dir(base)), "mode": "read_write"},
        {"name": "projects", "path": str(runtime_paths.projects_dir(base)), "mode": "read_write"},
    ]
    return {"grants": await folder_access.list_grants(get_db()), "base_folders": base_folders}


async def _revoke_grant(grant_id: str) -> dict:
    if not await folder_access.revoke(get_db(), grant_id):
        raise HTTPException(status_code=404, detail="no live grant with that id")
    return {"id": grant_id, "revoked": True}


@asynccontextmanager
async def _live_grants() -> AsyncIterator[None]:
    # Reason: entered after the database opens and exited before it closes, so the provider
    # never reads a closed connection.
    folder_access.activate(get_db())
    try:
        yield
    finally:
        folder_access.deactivate()


def register(app: FastAPI) -> None:
    """Add the owner folder-grant routes and install the live-grants provider at start-up.

    Args:
        app: The FastAPI application.
    """
    router = APIRouter(prefix="/api/workspace/folder-grants", dependencies=[Depends(require_owner)])
    router.add_api_route("", _list_grants, methods=["GET"])
    router.add_api_route("/{grant_id}/revoke", _revoke_grant, methods=["POST"])
    app.include_router(router)
    add_lifespan(_live_grants)
    logger.info("workspace folder-grant routes registered")
