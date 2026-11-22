"""HTTP helpers for guest / family-tester auth on the web UI."""

from __future__ import annotations

from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

from core.guest_access import (
    GUEST_PUBLIC_PATHS,
    guest_access_enabled,
    validate_guest_token,
)
from web.owner_controls import configured_web_token, extract_bearer_token

# Paths guests may call (exact). Prefix rules handled separately.
_GUEST_EXACT_PATHS = frozenset(
    {
        "/api/chat",
        "/api/status",
        "/api/export",
        "/api/tools",
        "/api/commands",
        "/api/search/providers",
    }
)


def guest_may_call(path: str) -> bool:
    if path.startswith("/api/guest/admin/"):
        return False
    if path in _GUEST_EXACT_PATHS:
        return True
    if path.startswith("/api/guest/"):
        return True
    if path == "/api/sessions" or path.startswith("/api/sessions/"):
        return True
    return False


def resolve_auth(request: Request, config: Any) -> dict[str, Any]:
    """Classify the caller for /api/* middleware.

    Returns: role ('owner'|'guest'|None), guest (payload|None), allow (bool).
    """
    path = request.url.path
    token = extract_bearer_token(request)
    web_token = configured_web_token()
    guests_on = guest_access_enabled(config)

    # Status is always public so /join can show "disabled" without a token.
    if path == "/api/guest/status":
        return {"role": None, "guest": None, "token": token, "allow": True}

    if path in GUEST_PUBLIC_PATHS and guests_on:
        return {"role": None, "guest": None, "token": token, "allow": True}

    if web_token and token == web_token:
        return {"role": "owner", "guest": None, "token": token, "allow": True}

    if guests_on and token:
        payload = validate_guest_token(token)
        if payload:
            return {
                "role": "guest",
                "guest": payload,
                "token": token,
                "allow": guest_may_call(path),
            }

    # Whenever guests are enabled, owner needs the owner token (RT-5): with no
    # token configured, non-guests are refused rather than treated as owner.
    require_auth = guests_on or bool(
        getattr(getattr(config, "web_ui", None), "require_auth", True) and web_token
    )
    if not require_auth:
        return {"role": "owner", "guest": None, "token": token, "allow": True}

    return {"role": None, "guest": None, "token": token, "allow": False}


def owner_locked_warning() -> str:
    """Start-up warning text: guests enabled but no owner token configured."""
    return (
        "Guest access is enabled but WITSV3_WEB_TOKEN is not set - the owner UI "
        "is LOCKED (all non-guest /api calls get 401) until you add "
        "WITSV3_WEB_TOKEN to .env."
    )


def guest_forbidden_response() -> JSONResponse:
    return JSONResponse(
        {"detail": "This action is only available to the owner."},
        status_code=403,
    )


def unauthorized_response() -> JSONResponse:
    return JSONResponse({"detail": "unauthorized"}, status_code=401)
