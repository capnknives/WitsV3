"""Guest / family-tester access: registry, tokens, and tool allowlist.

See planning/roadmap/guest-tester-access-2026-07.md.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger("WitsV3.GuestAccess")

# Safe tools for guest chat. Everything else is blocked in the orchestrator.
GUEST_ALLOWED_TOOLS = frozenset(
    {
        "web_search",
        "math_operations",
        "datetime",
        "read_conversation_history",
        "analyze_conversation",
        "enhanced_reasoning",
    }
)

# API path prefixes guests may call (exact or prefix match).
GUEST_ALLOWED_API_PREFIXES = (
    "/api/guest/",
    "/api/chat",
    "/api/status",
    "/api/export",
    "/api/tools",
    "/api/search/providers",
)

# Register / me are public (invite-gated); other /api/guest/* need a guest token.
GUEST_PUBLIC_PATHS = frozenset(
    {
        "/api/guest/register",
        "/api/guest/status",
    }
)

DEFAULT_PROFILES_PATH = Path("data/guest_profiles.json")


def guest_invite_configured() -> str:
    return os.getenv("WITSV3_GUEST_INVITE", "").strip()


def guest_signing_secret() -> str:
    """HMAC key for guest tokens. Prefer WITSV3_GUEST_SECRET; else derive."""
    explicit = os.getenv("WITSV3_GUEST_SECRET", "").strip()
    if explicit:
        return explicit
    invite = guest_invite_configured()
    web = os.getenv("WITSV3_WEB_TOKEN", "").strip()
    if invite and web:
        return hashlib.sha256(f"wits-guest|{web}|{invite}".encode()).hexdigest()
    if invite:
        return hashlib.sha256(f"wits-guest-invite|{invite}".encode()).hexdigest()
    return ""


def guest_access_enabled(config: Any) -> bool:
    """True when config enables guests and invite + signing material exist."""
    web_ui = getattr(config, "web_ui", None)
    guest = getattr(web_ui, "guest_access", None) if web_ui else None
    if guest is None or not getattr(guest, "enabled", False):
        return False
    return bool(guest_invite_configured() and guest_signing_secret())


def is_private_lan_ip(ip: str | None) -> bool:
    if not ip:
        return False
    ip = ip.strip()
    # FastAPI TestClient reports host as "testclient"
    if ip in ("127.0.0.1", "::1", "localhost", "testclient"):
        return True
    if ip.startswith("10."):
        return True
    if ip.startswith("192.168."):
        return True
    if ip.startswith("172."):
        try:
            second = int(ip.split(".")[1])
            return 16 <= second <= 31
        except (IndexError, ValueError):
            return False
    return False


def guest_session_key(guest_id: str, session_id: str) -> str:
    return f"guest:{guest_id}:{session_id}"


class GuestRegistry:
    """File-backed guest profiles keyed by device_id."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else DEFAULT_PROFILES_PATH
        self._data: dict[str, Any] = {"guests": {}}
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            self._data = json.loads(self.path.read_text(encoding="utf-8"))
            if "guests" not in self._data:
                self._data = {"guests": {}}
        except Exception as e:
            logger.warning("Failed to load guest profiles from %s: %s", self.path, e)
            self._data = {"guests": {}}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    def find_by_device(self, device_id: str) -> dict[str, Any] | None:
        for guest in self._data["guests"].values():
            if device_id in guest.get("device_ids", []) and not guest.get("revoked"):
                return guest
        return None

    def get(self, guest_id: str) -> dict[str, Any] | None:
        guest = self._data["guests"].get(guest_id)
        if guest and not guest.get("revoked"):
            return guest
        return None

    def display_name_for(self, guest_id: str, fallback: str | None = None) -> str:
        """Join-page display name from registry (source of truth for logs/UI)."""
        profile = self.get(guest_id)
        if profile and profile.get("display_name"):
            return str(profile["display_name"]).strip()[:80]
        return (fallback or "Guest").strip()[:80]

    def find_by_display_name(self, display_name: str) -> dict[str, Any] | None:
        target = display_name.strip().lower()
        for guest in self._data["guests"].values():
            if guest.get("revoked"):
                continue
            if (guest.get("display_name") or "").strip().lower() == target:
                return guest
        return None

    def register_or_update(
        self, *, display_name: str, device_id: str, age_band: str = "teen"
    ) -> dict[str, Any]:
        now = time.time()
        existing = self.find_by_device(device_id)
        if existing:
            existing["display_name"] = display_name.strip()[:80] or existing["display_name"]
            existing["last_seen"] = now
            if age_band:
                existing["age_band"] = age_band
            self._save()
            out = dict(existing)
            out["_returning"] = True
            return out

        guest_id = str(uuid.uuid4())
        profile = {
            "guest_id": guest_id,
            "display_name": display_name.strip()[:80] or "Guest",
            "device_ids": [device_id],
            "age_band": age_band or "teen",
            "first_seen": now,
            "last_seen": now,
            "revoked": False,
        }
        self._data["guests"][guest_id] = profile
        self._save()
        out = dict(profile)
        out["_returning"] = False
        return out

    def touch(self, guest_id: str) -> None:
        guest = self._data["guests"].get(guest_id)
        if not guest:
            return
        guest["last_seen"] = time.time()
        self._save()

    def revoke(self, guest_id: str) -> bool:
        guest = self._data["guests"].get(guest_id)
        if not guest:
            return False
        guest["revoked"] = True
        self._save()
        return True

    def list_guests(self) -> list[dict[str, Any]]:
        return list(self._data["guests"].values())

    def list_active_guests(self) -> list[dict[str, Any]]:
        return [g for g in self.list_guests() if not g.get("revoked")]


def format_active_guest_accounts(
    registry: GuestRegistry | None = None,
    *,
    include_revoked: bool = False,
) -> str:
    """Human-readable roster of /join guest accounts for the owner."""
    from datetime import datetime, timezone

    reg = registry or GuestRegistry()
    active = reg.list_active_guests()
    revoked = [g for g in reg.list_guests() if g.get("revoked")]

    if not active and not (include_revoked and revoked):
        return (
            "No active guest accounts yet. Family testers appear here after they "
            "register at /join with the invite code."
        )

    def _fmt_ts(ts: float | int | None) -> str:
        if not ts:
            return "unknown"
        return datetime.fromtimestamp(float(ts), timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [f"Active guest accounts: {len(active)}"]
    for g in sorted(active, key=lambda x: (x.get("display_name") or "").lower()):
        devices = len(g.get("device_ids") or [])
        lines.append(
            f"- {g.get('display_name', 'Guest')} "
            f"(id {g['guest_id'][:8]}…, age_band={g.get('age_band', 'teen')}, "
            f"devices={devices}, first={_fmt_ts(g.get('first_seen'))}, "
            f"last_seen={_fmt_ts(g.get('last_seen'))})"
        )

    if include_revoked and revoked:
        lines.append(f"\nRevoked guests: {len(revoked)}")
        for g in sorted(revoked, key=lambda x: (x.get("display_name") or "").lower()):
            lines.append(f"- {g.get('display_name', 'Guest')} (id {g['guest_id'][:8]}…)")

    return "\n".join(lines)


def _b64url(data: bytes) -> str:
    import base64

    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    import base64

    pad = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + pad)


def enrich_guest_payload(
    payload: dict[str, Any] | None,
    registry: GuestRegistry | None = None,
) -> dict[str, Any] | None:
    """Merge registry display_name into a validated guest token payload."""
    if not payload or not payload.get("guest_id"):
        return payload
    reg = registry or GuestRegistry()
    name = reg.display_name_for(payload["guest_id"], fallback=payload.get("display_name"))
    merged = dict(payload)
    merged["display_name"] = name
    return merged


def issue_guest_token(
    *,
    guest_id: str,
    device_id: str,
    display_name: str,
    ttl_hours: int = 720,
) -> str:
    secret = guest_signing_secret()
    if not secret:
        raise RuntimeError("Guest signing secret is not configured")
    now = int(time.time())
    payload = {
        "role": "guest",
        "guest_id": guest_id,
        "device_id": device_id,
        "display_name": display_name,
        "iat": now,
        "exp": now + max(1, ttl_hours) * 3600,
        "jti": secrets.token_hex(8),
    }
    body = _b64url(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode())
    sig = _b64url(hmac.new(secret.encode(), body.encode(), hashlib.sha256).digest())
    return f"{body}.{sig}"


def validate_guest_token(token: str) -> dict[str, Any] | None:
    secret = guest_signing_secret()
    if not secret or not token or "." not in token:
        return None
    body, _, sig = token.partition(".")
    expected = _b64url(hmac.new(secret.encode(), body.encode(), hashlib.sha256).digest())
    if not hmac.compare_digest(sig, expected):
        return None
    try:
        payload = json.loads(_b64url_decode(body))
    except Exception:
        return None
    if payload.get("role") != "guest":
        return None
    if int(payload.get("exp", 0)) < int(time.time()):
        return None
    if not payload.get("guest_id") or not payload.get("device_id"):
        return None
    return payload


def invites_match(presented: str, expected: str) -> bool:
    if not presented or not expected:
        return False
    return hmac.compare_digest(presented.strip(), expected.strip())
