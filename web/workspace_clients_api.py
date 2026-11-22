"""Owner routes for workspace clients (contract section 5).

``register(app)`` is called by ``web.workspace_routes.register_workspace``. Every route depends on
``require_owner``. The create route returns the token once; the list never returns a token, its
hash or its prefix.
"""

import logging

from fastapi import APIRouter, Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from core.workspace import clients
from core.workspace.store import WorkspaceError
from web.workspace_routes import get_db, require_owner

logger = logging.getLogger("WitsV3.Workspace.ClientsAPI")


class CreateClientBody(BaseModel):
    """JSON body for ``POST /api/workspace/clients``."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=clients.NAME_MAX)
    scopes: list[str] = Field(min_length=1, max_length=3)
    projects: list[str] = Field(default_factory=lambda: ["*"], min_length=1, max_length=100)
    rate_per_min: int = Field(default=60, ge=1, le=10_000)
    expires_at: int | None = Field(default=None, ge=0)


def _error(exc: WorkspaceError) -> HTTPException:
    status = 409 if exc.code == "name_taken" else 400
    return HTTPException(status_code=status, detail={"code": exc.code, "message": exc.message})


async def _list_clients() -> dict:
    return {"clients": await clients.list_clients(get_db())}


async def _create_client(body: CreateClientBody) -> JSONResponse:
    try:
        identity, token = await clients.create_client(
            get_db(), body.name, body.scopes, body.projects, body.rate_per_min, body.expires_at
        )
    except WorkspaceError as exc:
        raise _error(exc) from None
    payload = {
        "client": {
            "id": identity.id,
            "name": identity.name,
            "scopes": sorted(identity.scopes),
            "projects": sorted(identity.projects),
        },
        "token": token,
    }
    # Reason: the only response that ever carries a token must not be cached anywhere.
    return JSONResponse(payload, status_code=201, headers={"Cache-Control": "no-store"})


async def _revoke_client(client_id: str) -> dict:
    if not await clients.revoke_client(get_db(), client_id):
        raise HTTPException(status_code=404, detail="no active client with that id")
    return {"id": client_id, "revoked": True}


def register(app: FastAPI) -> None:
    """Add the owner client routes to the app.

    Args:
        app: The FastAPI application.
    """
    router = APIRouter(prefix="/api/workspace/clients", dependencies=[Depends(require_owner)])
    router.add_api_route("", _list_clients, methods=["GET"])
    router.add_api_route("", _create_client, methods=["POST"], status_code=201)
    router.add_api_route("/{client_id}/revoke", _revoke_client, methods=["POST"])
    app.include_router(router)
    logger.info("workspace client routes registered")
