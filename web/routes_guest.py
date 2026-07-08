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
    normalize_age_band,
)
from core.guest_audit import GuestAuditLog
from core.guest_user_profile import GuestUserProfileStore
from web.access_log import owner_display_name
from web.schemas import (
    GuestMergeRequest,
    GuestRegisterRequest,
    GuestRevokeRequest,
    GuestSetAgeBandRequest,
)

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

        default_band = normalize_age_band(
            guest_cfg.default_guest_age_band, default="teen"
        )

        profile = guest_registry.register_or_update(
            display_name=name,
            device_id=device_id,
            default_age_band=default_band,
        )
        request.state.caller_label = profile["display_name"]
        request.state.auth_role = "guest"
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

    @app.patch("/api/guest/admin/age-band")
    async def owner_set_guest_age_band(body: GuestSetAgeBandRequest, request: Request):
        """Owner-only: assign child / teen / adult protection tier for a guest."""
        if getattr(request.state, "auth_role", None) != "owner":
            return JSONResponse(
                {"detail": "Only the owner can change guest age bands."},
                status_code=403,
            )
        if not body.guest_id and not body.display_name:
            return JSONResponse(
                {"detail": "guest_id or display_name is required."},
                status_code=400,
            )
        band = normalize_age_band(body.age_band)
        if body.guest_id:
            profile = guest_registry.set_age_band(body.guest_id.strip(), band)
        else:
            profile = guest_registry.set_age_band_by_name(
                (body.display_name or "").strip(), band
            )
        if not profile:
            return JSONResponse({"detail": "Guest not found."}, status_code=404)

        guest_audit.log(
            guest_id=profile["guest_id"],
            event_type="age_band_set",
            display_name=profile.get("display_name"),
            meta={"age_band": band, "set_by": "owner"},
        )
        logger.info(
            "Owner set guest %s age_band=%s",
            profile.get("display_name"),
            band,
        )
        return {
            "guest_id": profile["guest_id"],
            "display_name": profile["display_name"],
            "age_band": profile.get("age_band", band),
        }

    @app.get("/api/guest/admin/accounts")
    async def owner_list_guest_accounts(request: Request):
        """Owner-only: roster + interest profile summaries."""
        if getattr(request.state, "auth_role", None) != "owner":
            return JSONResponse(
                {"detail": "Only the owner can view guest accounts."},
                status_code=403,
            )
        store = GuestUserProfileStore()
        summaries = store.list_profile_summaries(guest_registry)
        summary_by_name = {p["display_name"].lower(): p for p in summaries}
        guests = []
        for acct in guest_registry.list_active_guests():
            gid = acct["guest_id"]
            name = acct.get("display_name", "Guest")
            prof = summary_by_name.get(name.lower(), {})
            dupes = guest_registry.find_all_by_display_name(name)
            guests.append(
                {
                    "guest_id": gid,
                    "display_name": name,
                    "age_band": acct.get("age_band", "teen"),
                    "first_seen": acct.get("first_seen"),
                    "last_seen": acct.get("last_seen"),
                    "device_count": len(acct.get("device_ids") or []),
                    "turn_count": prof.get("turn_count", 0),
                    "top_interests": prof.get("top_interests", []),
                    "fact_count": prof.get("fact_count", 0),
                    "profile_updated_at": prof.get("updated_at"),
                    "duplicate_name": len(dupes) > 1,
                    "merge_candidates": [
                        {"guest_id": g["guest_id"], "last_seen": g.get("last_seen")}
                        for g in dupes
                        if g["guest_id"] != gid
                    ],
                }
            )
        return {
            "enabled": guest_access_enabled(system.config),
            "invite_configured": bool(guest_invite_configured()),
            "owner_display_name": owner_display_name(system.config),
            "duplicate_names": list(guest_registry.duplicate_display_names().keys()),
            "guests": guests,
        }

    @app.get("/api/guest/admin/profile")
    async def owner_get_guest_profile(request: Request, guest_id: str = "", display_name: str = ""):
        """Owner-only: full JSON interest profile for one guest."""
        if getattr(request.state, "auth_role", None) != "owner":
            return JSONResponse(
                {"detail": "Only the owner can view guest profiles."},
                status_code=403,
            )
        acct = None
        if guest_id:
            acct = guest_registry.get(guest_id.strip())
        elif display_name:
            acct = guest_registry.find_by_display_name(display_name.strip())
        if not acct:
            return JSONResponse({"detail": "Guest not found."}, status_code=404)
        store = GuestUserProfileStore()
        profile = store.load_merged_for_display_name(
            acct.get("display_name", "Guest"), guest_registry
        ) or store.load(acct["guest_id"], acct.get("display_name", "Guest"))
        return {
            "account": {
                "guest_id": acct["guest_id"],
                "display_name": acct.get("display_name"),
                "age_band": acct.get("age_band", "teen"),
                "last_seen": acct.get("last_seen"),
                "duplicate_accounts": len(
                    guest_registry.find_all_by_display_name(acct.get("display_name", ""))
                ),
            },
            "profile": profile,
            "summary": store.format_owner_summary(
                display_name=acct.get("display_name"),
                registry=guest_registry,
            ),
        }

    @app.delete("/api/guest/admin/account")
    async def owner_revoke_guest(body: GuestRevokeRequest, request: Request):
        """Owner-only: revoke a guest account and remove its profile file."""
        if getattr(request.state, "auth_role", None) != "owner":
            return JSONResponse(
                {"detail": "Only the owner can revoke guest accounts."},
                status_code=403,
            )
        guest_id = body.guest_id.strip()
        acct = guest_registry.get(guest_id)
        if not acct:
            return JSONResponse({"detail": "Guest not found."}, status_code=404)
        name = acct.get("display_name", "Guest")
        if not guest_registry.revoke(guest_id):
            return JSONResponse({"detail": "Guest not found."}, status_code=404)
        store = GuestUserProfileStore()
        remaining = guest_registry.find_all_by_display_name(name)
        if len(remaining) <= 1:
            store.delete_profile(guest_id)
            if remaining:
                store.consolidate_for_display_name(guest_registry, name)
        else:
            store.delete_profile(guest_id)
        guest_audit.log(
            guest_id=guest_id,
            event_type="revoked",
            display_name=name,
            meta={"revoked_by": "owner"},
        )
        logger.info("Owner revoked guest %s (%s)", name, guest_id[:8])
        return {"revoked": True, "guest_id": guest_id, "display_name": name}

    @app.post("/api/guest/admin/merge")
    async def owner_merge_guests(body: GuestMergeRequest, request: Request):
        """Owner-only: merge duplicate guest accounts into one."""
        if getattr(request.state, "auth_role", None) != "owner":
            return JSONResponse(
                {"detail": "Only the owner can merge guest accounts."},
                status_code=403,
            )
        target_id = body.target_guest_id.strip()
        source_id = body.source_guest_id.strip()
        merged_acct = guest_registry.merge_guests(
            target_guest_id=target_id, source_guest_id=source_id
        )
        if not merged_acct:
            return JSONResponse({"detail": "Could not merge guests."}, status_code=400)
        store = GuestUserProfileStore()
        name = merged_acct.get("display_name", "Guest")
        store.merge_profiles(
            target_guest_id=target_id,
            source_guest_id=source_id,
            display_name=name,
        )
        store.consolidate_for_display_name(guest_registry, name)
        guest_audit.log(
            guest_id=target_id,
            event_type="accounts_merged",
            display_name=name,
            meta={"merged_from": source_id},
        )
        logger.info("Owner merged guest %s into %s", source_id[:8], target_id[:8])
        return {
            "merged": True,
            "guest_id": target_id,
            "display_name": name,
            "device_count": len(merged_acct.get("device_ids") or []),
        }
