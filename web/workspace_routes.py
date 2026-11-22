"""Gateway shell for the shared agent workspace (contract sections 5, 7 and 9).

This module only wires things together: it opens the workspace database, wraps the app's
lifespan, loads the optional route modules and provides the owner check. Client, approval and
MCP routes live in their own modules and register themselves through ``register(app)``.
"""

from __future__ import annotations

import asyncio
import hmac
import importlib
import logging
import os
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException, Request

from core.config import WorkspaceSettings, load_config
from core.runtime_paths import project_root
from web.owner_controls import configured_web_token, extract_bearer_token

if TYPE_CHECKING:
    # Reason: imported lazily at startup so web.server imports without aiosqlite (absent from
    # some runtime environments) while the workspace is disabled.
    from core.workspace.db import Database

logger = logging.getLogger("WitsV3.Workspace")

# Reason: order matters (clients before approvals before the MCP mount); modules that are not
# built yet are skipped so packages can land one at a time.
_OPTIONAL_MODULES: tuple[str, ...] = (
    "web.workspace_clients_api",
    "web.workspace_approvals_api",
    "web.workspace_folder_api",
    "web.workspace_mcp",
    "core.workspace.notify",
)

_db: Database | None = None
_lifespan_factories: list[Callable[[], AbstractAsyncContextManager]] = []


def get_db() -> Database:
    """Return the open workspace database.

    Returns:
        The Database opened at startup.

    Raises:
        RuntimeError: If the workspace is disabled or the app has not started.
    """
    if _db is None:
        raise RuntimeError("workspace database is not open (workspace disabled or not started)")
    return _db


def add_lifespan(factory: Callable[[], AbstractAsyncContextManager]) -> None:
    """Register a context manager to enter at startup and exit at shutdown.

    Args:
        factory: Called once at startup; the context it returns is held open until shutdown
            (for example the MCP session manager's ``run()``).
    """
    _lifespan_factories.append(factory)


async def require_owner(request: Request) -> None:
    """FastAPI dependency: allow only the owner's ``WITSV3_WEB_TOKEN`` bearer token.

    Args:
        request: The incoming request.

    Raises:
        HTTPException: 401 when the token is missing, empty, wrong or not encodable.
    """
    expected = configured_web_token()
    presented = extract_bearer_token(request)
    # Reason: an empty presented token must never reach the comparison, and an unset server
    # token must fail closed rather than match an empty header.
    if not presented or not expected:
        raise HTTPException(status_code=401, detail="owner token required")
    try:
        match = hmac.compare_digest(presented.encode("utf-8"), expected.encode("utf-8"))
    except UnicodeEncodeError:
        match = False
    if not match:
        raise HTTPException(status_code=401, detail="owner token required")


def _resolve_db_path(settings: WorkspaceSettings) -> Path:
    path = Path(settings.db_path)
    return path if path.is_absolute() else project_root() / path


def _load_optional_modules(app: FastAPI) -> None:
    for name in _OPTIONAL_MODULES:
        try:
            module = importlib.import_module(name)
        except ModuleNotFoundError as exc:
            # Reason: only the module itself being absent means "not built yet"; a missing
            # dependency inside an existing module is a real bug and must surface.
            if exc.name != name:
                raise
            logger.info("workspace module %s not built yet; skipping", name)
            continue
        module.register(app)


def _wrap_lifespan(app: FastAPI, settings: WorkspaceSettings) -> None:
    original = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(inner_app: FastAPI) -> AsyncIterator:
        global _db
        from core.workspace.db import open_db

        async with original(inner_app) as state:
            db_path = _resolve_db_path(settings)
            # Reason: mkdir is blocking file I/O, so keep it off the event loop.
            await asyncio.to_thread(db_path.parent.mkdir, parents=True, exist_ok=True)
            db = await open_db(db_path)
            _db = db
            try:
                async with AsyncExitStack() as stack:
                    for factory in list(_lifespan_factories):
                        await stack.enter_async_context(factory())
                    yield state
            finally:
                _db = None
                await db.close()

    app.router.lifespan_context = lifespan


def register_workspace(app: FastAPI) -> None:
    """Mount the workspace gateway on the app when ``workspace.enabled`` is true.

    Does nothing when disabled (the default): no routes, no lifespan wrapper, no database file.
    Otherwise wraps (never replaces) the app's lifespan to open the database and enter contexts
    added with ``add_lifespan``, then loads each optional route module and calls its
    ``register(app)``.

    Args:
        app: The FastAPI application.

    Raises:
        RuntimeError: If enabled while ``WITSV3_WEB_TOKEN`` is unset or empty, or while
            ``web_ui.require_auth`` is false.
    """
    config = load_config()
    settings = config.workspace
    if not settings.enabled:
        return
    # Reason: with require_auth off the existing middleware opens every other /api/* route to
    # any caller on the network, including the agent hosts the workspace admits.
    if not config.web_ui.require_auth:
        raise RuntimeError(
            "workspace.enabled is true but web_ui.require_auth is false; "
            "set web_ui.require_auth to true before enabling the workspace"
        )
    # Reason: without a token the existing middleware treats every caller as owner, so owner
    # routes would be open to the network.
    if not os.getenv("WITSV3_WEB_TOKEN", "").strip():
        raise RuntimeError(
            "workspace.enabled is true but WITSV3_WEB_TOKEN is unset or empty; "
            "set it in .env before enabling the workspace"
        )
    _lifespan_factories.clear()
    _wrap_lifespan(app, settings)
    _load_optional_modules(app)
