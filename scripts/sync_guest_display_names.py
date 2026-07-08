#!/usr/bin/env python3
"""Normalize guest join names in registry + audit logs (Sean, TESTER, etc.)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.guest_access import GuestRegistry
from core.guest_audit import GuestAuditLog, sync_audit_display_names

# Canonical /join display names (case-insensitive match on existing value).
CANONICAL_DISPLAY_NAMES = {
    "sean": "Sean",
    "tester": "TESTER",
}


def normalize_registry(registry: GuestRegistry) -> list[str]:
    changed: list[str] = []
    for guest in registry.list_guests():
        if guest.get("revoked"):
            continue
        current = (guest.get("display_name") or "").strip()
        canonical = CANONICAL_DISPLAY_NAMES.get(current.lower())
        if canonical and current != canonical:
            guest["display_name"] = canonical
            changed.append(f"{guest['guest_id'][:8]}… -> {canonical}")
    if changed:
        registry._save()
    return changed


def main() -> int:
    registry = GuestRegistry()
    audit = GuestAuditLog()

    print("=== Guest display name sync ===")
    reg_changes = normalize_registry(registry)
    if reg_changes:
        print("Registry updated:")
        for line in reg_changes:
            print(f"  {line}")
    else:
        print("Registry: all known guests already use join names.")

    for guest in registry.list_guests():
        if not guest.get("revoked"):
            print(f"  - {guest.get('display_name')} ({guest['guest_id'][:8]}…)")

    audit_updates = sync_audit_display_names(audit=audit, registry=registry)
    print(f"Audit JSONL rows relabeled: {audit_updates}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
