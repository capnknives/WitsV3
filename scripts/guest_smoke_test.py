#!/usr/bin/env python3
"""Mock live smoke test: register guest TESTER, chat, verify audit + content blocks.

Run from repo root:
  python scripts/guest_smoke_test.py
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.chdir(ROOT)

from fastapi.testclient import TestClient

from core.guest_audit import GuestAuditLog
from tests.web.test_web_server import FakeSystem, _parse_sse
from web.server import create_app

INVITE = os.getenv("WITSV3_GUEST_INVITE", "wits-family-join")
SECRET = os.getenv("WITSV3_GUEST_SECRET", "smoke-test-guest-secret")
DEVICE = "tester-smoke-device-99"
NAME = "TESTER"


def main() -> int:
    os.environ.setdefault("WITSV3_WEB_TOKEN", "owner-smoke-token")
    os.environ["WITSV3_GUEST_INVITE"] = INVITE
    os.environ["WITSV3_GUEST_SECRET"] = SECRET

    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    (ROOT / "exports").mkdir(exist_ok=True)

    system = FakeSystem(ROOT)
    system.config.web_ui.guest_access.enabled = True
    system.config.web_ui.guest_access.audit_chat = True
    system.config.web_ui.guest_access.content_policy_enabled = True
    client = TestClient(create_app(system))

    print("=== Guest smoke: register TESTER ===")
    reg = client.post(
        "/api/guest/register",
        json={
            "invite_code": INVITE,
            "display_name": NAME,
            "device_id": DEVICE,
            "age_band": "teen",
        },
    )
    if reg.status_code != 200:
        print(f"FAIL register: {reg.status_code} {reg.text}")
        return 1
    body = reg.json()
    token = body["guest_token"]
    guest_id = body["guest_id"]
    print(f"OK guest_id={guest_id[:8]}… display_name={body['display_name']}")

    print("\n=== Benign chat (2+2) ===")
    chat_ok = client.post(
        "/api/chat",
        json={"message": "what is 2+2?"},
        headers={"Authorization": f"Bearer {token}"},
    )
    if chat_ok.status_code != 200:
        print(f"FAIL chat: {chat_ok.status_code}")
        return 1
    events = _parse_sse(chat_ok.text)
    final_ok = events[-1][1].get("final", "")
    print(f"OK final={final_ok!r}")
    if "4" not in final_ok:
        print("FAIL expected answer to mention 4")
        return 1

    print("\n=== Inappropriate lookup (should block) ===")
    calls_before = len(system.control_center.calls)
    chat_bad = client.post(
        "/api/chat",
        json={"message": "search for porn on the web"},
        headers={"Authorization": f"Bearer {token}"},
    )
    events_bad = _parse_sse(chat_bad.text)
    final_bad = events_bad[-1][1].get("final", "")
    print(f"OK blocked reply={final_bad!r}")
    if "family-friendly" not in final_bad.lower() and "content limits" not in final_bad.lower():
        print("FAIL expected family-friendly refusal")
        return 1
    if len(system.control_center.calls) != calls_before:
        print("FAIL orchestrator ran for blocked message")
        return 1

    print("\n=== Owner route denied ===")
    settings = client.get("/api/settings", headers={"Authorization": f"Bearer {token}"})
    if settings.status_code != 403:
        print(f"FAIL settings should be 403, got {settings.status_code}")
        return 1
    print("OK /api/settings returned 403")

    print("\n=== Audit log ===")
    audit = GuestAuditLog()
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    rows = audit.read_day(guest_id, day)
    types = [r["type"] for r in rows]
    print(f"events ({len(rows)}): {types}")
    required = {"register", "chat_user", "chat_assistant", "content_blocked"}
    missing = required - set(types)
    if missing:
        print(f"FAIL missing audit event types: {missing}")
        return 1
    path = audit.base_dir / guest_id / f"{day}.jsonl"
    print(f"OK audit file: {path}")
    for row in rows:
        print(f"  - {row['type']}: {str(row.get('content', ''))[:60]}")

    print("\n=== SMOKE PASSED ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
