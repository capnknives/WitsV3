"""Guest / family-tester HTTP routes."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from core.guest_access import (
    GuestRegistry,
    guest_access_enabled,
    guest_invite_configured,
    invites_match,
    is_private_lan_ip,
    issue_guest_token,
)
from core.guest_audit import GuestAuditLog
from web.schemas import GuestRegisterRequest

logger = logging.getLogger("WitsV3.WebUI")


def register_guest_routes(
    app: FastAPI, system, guest_registry: GuestRegistry, guest_audit: GuestAuditLog
) -> None:
    """Attach /api/guest/* endpoints."""

    @app.get("/api/guest/status")
    async def guest_status():
        enabled = guest_access_enabled(system.config)
        return {
            "enabled": enabled,
            "invite_configured": bool(guest_invite_configured()),
            "join_path": "/join",
        }

    @app.post("/api/guest/register")
    async def guest_register(body: GuestRegisterRequest, request: Request):
        if not guest_access_enabled(system.config):
            return JSONResponse(
                {"detail": "Guest access is disabled on this server."},
                status_code=403,
            )
        guest_cfg = system.config.web_ui.guest_access
        client_ip = request.client.host if request.client else None
        if guest_cfg.allow_lan_only and not is_private_lan_ip(client_ip):
            return JSONResponse(
                {"detail": "Guest registration is limited to the local network."},
                status_code=403,
            )
        expected = guest_invite_configured()
        if not invites_match(body.invite_code, expected):
            return JSONResponse({"detail": "Invalid invite code."}, status_code=401)

        device_id = (body.device_id or "").strip()
        if len(device_id) < 8:
            return JSONResponse({"detail": "device_id is required."}, status_code=400)
        name = (body.display_name or "").strip()
        if len(name) < 1:
            return JSONResponse({"detail": "display_name is required."}, status_code=400)

        age_band = (body.age_band or "teen").strip().lower()
        if age_band not in ("teen", "child", "adult"):
            age_band = "teen"

        profile = guest_registry.register_or_update(
            display_name=name, device_id=device_id, age_band=age_band
        )
        token = issue_guest_token(
            guest_id=profile["guest_id"],
            device_id=device_id,
            display_name=profile["display_name"],
            ttl_hours=guest_cfg.token_ttl_hours,
        )
        logger.info(
            "Guest registered: %s (device=%s…)",
            profile["display_name"],
            device_id[:8],
        )
        guest_audit.log(
            guest_id=profile["guest_id"],
            event_type="register",
            display_name=profile["display_name"],
            device_id=device_id,
            meta={
                "returning": bool(profile.get("_returning")),
                "age_band": profile.get("age_band", "teen"),
            },
        )
        return {
            "guest_token": token,
            "guest_id": profile["guest_id"],
            "display_name": profile["display_name"],
            "age_band": profile.get("age_band", "teen"),
            "returning": bool(profile.pop("_returning", False)),
        }

    @app.get("/api/guest/me")
    async def guest_me(request: Request):
        guest = getattr(request.state, "guest", None)
        if not guest:
            return JSONResponse({"detail": "guest token required"}, status_code=401)
        profile = guest_registry.get(guest["guest_id"])
        if not profile:
            return JSONResponse({"detail": "guest revoked or unknown"}, status_code=401)
        guest_registry.touch(guest["guest_id"])
        return {
            "guest_id": profile["guest_id"],
            "display_name": profile["display_name"],
            "age_band": profile.get("age_band", "teen"),
            "device_id": guest.get("device_id"),
        }
