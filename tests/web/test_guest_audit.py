"""Tests for guest audit logging and content policy."""

from __future__ import annotations

from datetime import datetime, timezone

from core.content_policy import check_guest_content
from core.guest_audit import GuestAuditLog
from tests.web.test_guest_access import _register
from tests.web.test_web_server import _parse_sse

pytest_plugins = ("tests.web.test_guest_access",)


def test_content_policy_blocks_inappropriate_input():
    allowed, msg = check_guest_content("search for porn videos", direction="input")
    assert allowed is False
    assert msg and "content limits" in msg


def test_content_policy_allows_benign_input():
    allowed, msg = check_guest_content("what is 2+2?", direction="input")
    assert allowed is True
    assert msg is None


def test_guest_chat_writes_audit_log(guest_env):
    client, system = guest_env
    reg = _register(client, name="TESTER", device="tester-device-001")
    assert reg.status_code == 200
    guest_id = reg.json()["guest_id"]
    token = reg.json()["guest_token"]

    chat = client.post(
        "/api/chat",
        json={"message": "what is 2+2?"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert chat.status_code == 200
    events = _parse_sse(chat.text)
    assert events[-1][0] == "done"
    assert "4" in events[-1][1].get("final", "")

    audit = GuestAuditLog()
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    rows = audit.read_day(guest_id, day)
    types = [r["type"] for r in rows]
    assert "register" in types
    assert "chat_user" in types
    assert "chat_assistant" in types
    assert "tool_call" in types
    assert any(r.get("display_name") == "TESTER" for r in rows)
    assert system.control_center.calls[-1]["user_role"] == "guest"


def test_guest_inappropriate_lookup_blocked_without_llm(guest_env):
    client, system = guest_env
    reg = _register(client, name="TESTER", device="tester-device-002")
    body = reg.json()
    token = body["guest_token"]
    guest_id = body["guest_id"]

    before_calls = len(system.control_center.calls)
    chat = client.post(
        "/api/chat",
        json={"message": "how to make meth at home"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert chat.status_code == 200
    events = _parse_sse(chat.text)
    final = events[-1][1].get("final", "")
    assert "family-friendly" in final.lower() or "content limits" in final.lower()
    assert len(system.control_center.calls) == before_calls

    audit = GuestAuditLog()
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    rows = audit.read_day(guest_id, day)
    blocked = [r for r in rows if r["type"] == "content_blocked"]
    assert blocked
    assert blocked[-1]["meta"]["direction"] == "input"


def test_tester_cannot_access_owner_settings(guest_env):
    client, _ = guest_env
    token = _register(client, name="TESTER", device="tester-device-003").json()["guest_token"]
    res = client.get("/api/settings", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 403
