"""Access log helpers: annotate HTTP lines with owner/guest identity."""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import Request

from core.guest_access import GuestRegistry, enrich_guest_payload
from web.guest_auth import resolve_auth

ACCESS_LOGGER = logging.getLogger("uvicorn.access")


def owner_display_name(config: Any) -> str:
    """Label for owner sessions in access logs (env overrides config)."""
    env_name = os.getenv("WITSV3_OWNER_NAME", "").strip()
    if env_name:
        return env_name[:40]
    web_ui = getattr(config, "web_ui", None)
    if web_ui is not None:
        name = getattr(web_ui, "owner_display_name", None)
        if name:
            return str(name).strip()[:40] or "Owner"
    return "Owner"


def resolve_caller_label(
    request: Request,
    config: Any,
    registry: GuestRegistry | None = None,
) -> str:
    """Return join-page display name for guests, owner label for owner, else anon/-."""
    preset = getattr(request.state, "caller_label", None)
    if isinstance(preset, str) and preset.strip():
        return preset.strip()[:40]

    reg = registry or GuestRegistry()
    state_role = getattr(request.state, "auth_role", None)
    state_guest = getattr(request.state, "guest", None)
    if state_role == "owner":
        return owner_display_name(config)
    if state_role == "guest" and state_guest:
        enriched = enrich_guest_payload(state_guest, reg)
        return (enriched or {}).get("display_name") or "Guest"

    auth = resolve_auth(request, config)
    if auth.get("role") == "owner":
        return owner_display_name(config)
    if auth.get("role") == "guest" and auth.get("guest"):
        enriched = enrich_guest_payload(auth["guest"], reg)
        return (enriched or {}).get("display_name") or "Guest"

    path = request.url.path
    if not path.startswith("/api/"):
        return "-"
    return "anon"


def log_http_access(
    request: Request,
    status_code: int,
    config: Any,
    registry: GuestRegistry | None = None,
) -> None:
    """Emit a uvicorn-style access line with caller label."""
    client = request.client
    host = client.host if client else "?"
    port = client.port if client else "?"
    label = resolve_caller_label(request, config, registry)
    path = request.url.path
    if request.url.query:
        path = f"{path}?{request.url.query}"
    ACCESS_LOGGER.info(
        '%s:%s [%s] - "%s %s HTTP/1.1" %s',
        host,
        port,
        label,
        request.method,
        path,
        status_code,
    )


def install_access_log_middleware(
    app,
    config: Any,
    guest_registry: GuestRegistry | None = None,
) -> None:
    """Log each HTTP request with owner/guest label (replaces uvicorn access_log)."""
    registry = guest_registry or GuestRegistry()

    @app.middleware("http")
    async def labeled_access_log(request: Request, call_next):
        response = await call_next(request)
        log_http_access(request, response.status_code, config, registry)
        return response
