"""Append-only audit log for guest / family-tester activity."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.guest_access import GuestRegistry

logger = logging.getLogger("WitsV3.GuestAudit")

DEFAULT_AUDIT_DIR = Path("data/guest_audit")


class GuestAuditLog:
    """Per-guest daily JSONL files for safety review and debugging."""

    def __init__(self, base_dir: Path | str | None = None, *, enabled: bool = True):
        self.base_dir = Path(base_dir) if base_dir else DEFAULT_AUDIT_DIR
        self.enabled = enabled

    def log(
        self,
        *,
        guest_id: str,
        event_type: str,
        display_name: str | None = None,
        device_id: str | None = None,
        session_id: str | None = None,
        content: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Path | None:
        if not self.enabled or not guest_id:
            return None

        record: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "type": event_type,
            "guest_id": guest_id,
        }
        if display_name:
            record["display_name"] = display_name
        if device_id:
            record["device_id"] = device_id
        if session_id:
            record["session_id"] = session_id
        if content is not None:
            record["content"] = content
        if meta:
            record["meta"] = meta

        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = self.base_dir / guest_id / f"{day}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(
            "Guest audit [%s] %s (%s)",
            event_type,
            display_name or guest_id[:8],
            guest_id[:8],
        )
        return path

    def read_day(self, guest_id: str, day: str | None = None) -> list[dict[str, Any]]:
        day = day or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = self.base_dir / guest_id / f"{day}.jsonl"
        if not path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows

    def list_guest_ids(self) -> list[str]:
        if not self.base_dir.is_dir():
            return []
        return sorted(
            p.name for p in self.base_dir.iterdir() if p.is_dir() and not p.name.startswith(".")
        )

    def read_recent(
        self,
        guest_id: str,
        *,
        days: int = 7,
        max_events: int = 500,
    ) -> list[dict[str, Any]]:
        days = max(1, min(days, 90))
        max_events = max(1, min(max_events, 2000))
        rows: list[dict[str, Any]] = []
        today = datetime.now(timezone.utc).date()
        for offset in range(days):
            day = (today - timedelta(days=offset)).isoformat()
            rows.extend(self.read_day(guest_id, day))
            if len(rows) >= max_events:
                break
        rows.sort(key=lambda r: r.get("ts", ""))
        return rows[-max_events:]


def resolve_guest_profiles(
    *,
    display_name: str | None = None,
    guest_id: str | None = None,
    registry_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Match registered guests by id and/or display name (case-insensitive)."""
    from core.guest_access import GuestRegistry

    registry = GuestRegistry(registry_path)
    if guest_id:
        profile = registry.get(guest_id)
        return [profile] if profile else []

    if not display_name:
        return [g for g in registry.list_guests() if not g.get("revoked")]

    target = display_name.strip().lower()
    return [
        g
        for g in registry.list_guests()
        if not g.get("revoked") and (g.get("display_name") or "").strip().lower() == target
    ]


def format_guest_audit_report(
    events: list[dict[str, Any]],
    *,
    profile: dict[str, Any] | None = None,
    days: int = 1,
) -> str:
    """Human-readable audit digest for the owner or LLM synthesis."""
    name = (profile or {}).get("display_name") or "Unknown guest"
    guest_id = (profile or {}).get("guest_id") or (events[0].get("guest_id") if events else "?")

    if not events:
        return (
            f"No guest audit events for {name} (id {guest_id[:8]}…) "
            f"in the last {days} day(s). They may not have chatted yet."
        )

    by_type: dict[str, int] = {}
    blocked: list[str] = []
    user_msgs: list[str] = []
    assistant_msgs: list[str] = []
    tools: list[str] = []

    for ev in events:
        et = ev.get("type", "unknown")
        by_type[et] = by_type.get(et, 0) + 1
        content = (ev.get("content") or "").strip()
        if et == "content_blocked" and content:
            blocked.append(content[:200])
        elif et == "chat_user" and content:
            user_msgs.append(content[:300])
        elif et == "chat_assistant" and content:
            assistant_msgs.append(content[:300])
        elif et == "tool_call" and content:
            tools.append(content[:200])

    lines = [
        f"Guest audit: {name} (id {guest_id})",
        f"Window: last {days} day(s), {len(events)} event(s)",
        f"Event counts: {', '.join(f'{k}={v}' for k, v in sorted(by_type.items()))}",
    ]
    if user_msgs:
        lines.append("User messages:")
        for msg in user_msgs[-10:]:
            lines.append(f"  - {msg}")
    if tools:
        lines.append("Tool calls:")
        for t in tools[-10:]:
            lines.append(f"  - {t}")
    if assistant_msgs:
        lines.append("Assistant replies:")
        for msg in assistant_msgs[-5:]:
            lines.append(f"  - {msg}")
    if blocked:
        lines.append("Blocked content (policy):")
        for b in blocked:
            lines.append(f"  - {b}")
    return "\n".join(lines)


def sync_audit_display_names(
    audit: GuestAuditLog | None = None,
    registry: GuestRegistry | None = None,
) -> int:
    """Rewrite guest audit JSONL display_name fields from the registry. Returns rows updated."""
    from core.guest_access import GuestRegistry

    audit = audit or GuestAuditLog()
    registry = registry or GuestRegistry()
    updated = 0
    if not audit.base_dir.is_dir():
        return 0
    for guest_dir in audit.base_dir.iterdir():
        if not guest_dir.is_dir():
            continue
        guest_id = guest_dir.name
        canonical = registry.display_name_for(guest_id, fallback=None)
        if not canonical or canonical == "Guest":
            continue
        for jsonl in guest_dir.glob("*.jsonl"):
            lines_out: list[str] = []
            changed = False
            for line in jsonl.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("display_name") != canonical:
                    row["display_name"] = canonical
                    changed = True
                    updated += 1
                lines_out.append(json.dumps(row, ensure_ascii=False))
            if changed:
                jsonl.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    return updated


def build_owner_audit_digest(
    *,
    display_name: str | None = None,
    guest_id: str | None = None,
    days: int = 1,
    include_all_guests: bool = False,
    audit: GuestAuditLog | None = None,
    registry_path: Path | str | None = None,
) -> str:
    """Aggregate audit logs for owner review inside chat."""
    audit = audit or GuestAuditLog()
    days = max(1, min(days, 30))

    if include_all_guests and not display_name and not guest_id:
        profiles = resolve_guest_profiles(registry_path=registry_path)
        if not profiles:
            ids = audit.list_guest_ids()
            if not ids:
                return "No guest audit logs found yet. Guests appear after someone registers at /join."
            sections = []
            for gid in ids:
                events = audit.read_recent(gid, days=days)
                sections.append(
                    format_guest_audit_report(events, profile={"guest_id": gid, "display_name": gid[:8]}, days=days)
                )
            return "\n\n---\n\n".join(sections)

    profiles = resolve_guest_profiles(
        display_name=display_name, guest_id=guest_id, registry_path=registry_path
    )
    if display_name and not profiles:
        return f"No registered guest named '{display_name}'. Check spelling or ask for all guest activity."

    if not profiles:
        return "No guest profiles found. Share an invite at /join first."

    sections: list[str] = []
    for profile in profiles:
        gid = profile["guest_id"]
        events = audit.read_recent(gid, days=days)
        sections.append(format_guest_audit_report(events, profile=profile, days=days))
    return "\n\n---\n\n".join(sections)
