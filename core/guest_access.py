"""Guest / family-tester access: registry, tokens, and tool allowlist.

See docs/roadmap/guest-tester-access-2026-07.md.
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

from core.runtime_paths import guest_profiles_path

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


def guest_tools_for_age_band(age_band: str, config: Any | None = None) -> frozenset[str]:
    """Tool allowlist for a guest based on owner-assigned age band."""
    allowed = set(GUEST_ALLOWED_TOOLS)
    band = normalize_age_band(age_band)
    guest_cfg = getattr(getattr(config, "web_ui", None), "guest_access", None) if config else None
    if guest_cfg is not None and getattr(guest_cfg, "allow_document_search", False):
        allowed.add("document_search")
    if (
        band == "adult"
        and guest_cfg is not None
        and getattr(guest_cfg, "adult_allow_document_search", True)
    ):
        allowed.add("document_search")
    return frozenset(allowed)


def resolve_effective_role(user_role: str, guest_profile: dict[str, Any] | None = None) -> str:
    """Map session user_role + guest profile to RBAC role name."""
    from core.guest_policy_loader import default_policy_role

    if (user_role or "owner").lower() != "guest":
        return "owner"
    if guest_profile and guest_profile.get("rbac_role"):
        return str(guest_profile["rbac_role"]).strip().lower()
    band = normalize_age_band((guest_profile or {}).get("age_band"))
    if band == "adult":
        return "family_adult"
    if band == "child":
        return "family_kid"
    return default_policy_role()


def tools_for_role(role: str, config: Any | None = None) -> frozenset[str] | None:
    """Return allowed tools for RBAC role; None means all tools (owner)."""
    role_key = (role or "owner").strip().lower()
    if role_key in ("owner", ""):
        return None
    from core.guest_policy_loader import policy_roles

    roles = policy_roles()
    entry = roles.get(role_key) if isinstance(roles, dict) else None
    if isinstance(entry, dict):
        tools = entry.get("allowed_tools")
        if tools == ["*"] or tools == "*":
            return None
        if isinstance(tools, list):
            return frozenset(str(t) for t in tools)
    if role_key == "guest":
        return GUEST_ALLOWED_TOOLS
    return GUEST_ALLOWED_TOOLS


def is_tool_allowed(tool_name: str, role: str, config: Any | None = None) -> bool:
    """Check whether *tool_name* is permitted for *role*."""
    allowed = tools_for_role(role, config)
    if allowed is None:
        return True
    return tool_name in allowed


def normalize_age_band(value: str | None, *, default: str = "teen") -> str:
    from core.content_policy import normalize_age_band as _norm

    return _norm(value, default=default)


# API path prefixes guests may call (exact or prefix match).
GUEST_ALLOWED_API_PREFIXES = (
    "/api/guest/",
    "/api/chat",
    "/api/status",
    "/api/export",
    "/api/tools",
    "/api/search/providers",
    "/api/sessions",
    "/api/commands",
)

# Register / me are public (invite-gated); other /api/guest/* need a guest token.
GUEST_PUBLIC_PATHS = frozenset(
    {
        "/api/guest/register",
        "/api/guest/status",
    }
)

def default_profiles_path() -> Path:
    return guest_profiles_path()


DEFAULT_PROFILES_PATH = default_profiles_path  # lazy; call as default_profiles_path()


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
        self.path = Path(path) if path else default_profiles_path()
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
        """Single active guest by join name (most recently seen if duplicates exist)."""
        matches = self.find_all_by_display_name(display_name)
        if not matches:
            return None
        if len(matches) == 1:
            return matches[0]
        return max(matches, key=lambda g: float(g.get("last_seen") or 0))

    def find_all_by_display_name(self, display_name: str) -> list[dict[str, Any]]:
        target = display_name.strip().lower()
        if not target:
            return []
        out = []
        for guest in self._data["guests"].values():
            if guest.get("revoked"):
                continue
            if (guest.get("display_name") or "").strip().lower() == target:
                out.append(guest)
        return out

    def duplicate_display_names(self) -> dict[str, list[dict[str, Any]]]:
        """Active guests grouped by display_name where count > 1."""
        by_name: dict[str, list[dict[str, Any]]] = {}
        for guest in self.list_active_guests():
            key = (guest.get("display_name") or "").strip().lower()
            if not key:
                continue
            by_name.setdefault(key, []).append(guest)
        return {k: v for k, v in by_name.items() if len(v) > 1}

    def register_or_update(
        self,
        *,
        display_name: str,
        device_id: str,
        default_age_band: str = "teen",
    ) -> dict[str, Any]:
        now = time.time()
        default_age_band = normalize_age_band(default_age_band)
        existing = self.find_by_device(device_id)
        if existing:
            existing["display_name"] = display_name.strip()[:80] or existing["display_name"]
            existing["last_seen"] = now
            # age_band is owner-assigned only — never changed on /join re-register.
            self._save()
            out = dict(existing)
            out["_returning"] = True
            return out

        name = display_name.strip()[:80] or "Guest"
        by_name = self.find_by_display_name(name)
        if by_name:
            devices = by_name.setdefault("device_ids", [])
            if device_id not in devices:
                devices.append(device_id)
            by_name["display_name"] = name
            by_name["last_seen"] = now
            self._save()
            out = dict(by_name)
            out["_returning"] = True
            return out

        guest_id = str(uuid.uuid4())
        profile = {
            "guest_id": guest_id,
            "display_name": display_name.strip()[:80] or "Guest",
            "device_ids": [device_id],
            "age_band": default_age_band,
            "first_seen": now,
            "last_seen": now,
            "revoked": False,
        }
        self._data["guests"][guest_id] = profile
        self._save()
        out = dict(profile)
        out["_returning"] = False
        return out

    def set_age_band(self, guest_id: str, age_band: str) -> dict[str, Any] | None:
        """Owner-only: assign child / teen / adult tier for a guest."""
        normalized = normalize_age_band(age_band)
        guest = self._data["guests"].get(guest_id)
        if not guest or guest.get("revoked"):
            return None
        guest["age_band"] = normalized
        self._save()
        return dict(guest)

    def set_age_band_by_name(self, display_name: str, age_band: str) -> dict[str, Any] | None:
        guest = self.find_by_display_name(display_name)
        if not guest:
            return None
        return self.set_age_band(guest["guest_id"], age_band)

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
        guest["revoked_at"] = time.time()
        self._save()
        return True

    def merge_guests(self, *, target_guest_id: str, source_guest_id: str) -> dict[str, Any] | None:
        """Owner-only: fold source account into target (devices, timestamps); revoke source."""
        if target_guest_id == source_guest_id:
            return self.get(target_guest_id)
        target = self._data["guests"].get(target_guest_id)
        source = self._data["guests"].get(source_guest_id)
        if not target or not source or target.get("revoked") or source.get("revoked"):
            return None
        now = time.time()
        for device_id in source.get("device_ids") or []:
            if device_id not in target.setdefault("device_ids", []):
                target["device_ids"].append(device_id)
        target["first_seen"] = min(
            float(target.get("first_seen") or now),
            float(source.get("first_seen") or now),
        )
        target["last_seen"] = max(
            float(target.get("last_seen") or 0),
            float(source.get("last_seen") or 0),
        )
        # Keep target display_name and age_band; unify casing from target.
        source["revoked"] = True
        source["revoked_at"] = now
        source["merged_into"] = target_guest_id
        self._save()
        return dict(target)

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
    profile = reg.get(payload["guest_id"])
    if profile:
        merged["age_band"] = profile.get("age_band", "teen")
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
