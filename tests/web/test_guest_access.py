"""Tests for guest / family-tester access."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from core.guest_access import (
    GUEST_ALLOWED_TOOLS,
    GuestRegistry,
    issue_guest_token,
    validate_guest_token,
)
from tests.web.test_web_server import FakeSystem, _parse_sse
from web.server import create_app


@pytest.fixture
def guest_env(tmp_path, monkeypatch):
    monkeypatch.setenv("WITSV3_WEB_TOKEN", "owner-sekrit")
    monkeypatch.setenv("WITSV3_GUEST_INVITE", "family-code")
    monkeypatch.setenv("WITSV3_GUEST_SECRET", "guest-signing-secret-xyz")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "exports").mkdir()

    system = FakeSystem(tmp_path)
    system.config.web_ui.guest_access.enabled = True
    system.config.web_ui.guest_access.allow_lan_only = True
    client = TestClient(create_app(system))
    return client, system


def _register(client, name="Alex", device="device-abc-12345", invite="family-code"):
    return client.post(
        "/api/guest/register",
        json={
            "invite_code": invite,
            "display_name": name,
            "device_id": device,
            "age_band": "teen",
        },
    )


def test_guest_status_reports_enabled(guest_env):
    client, _ = guest_env
    res = client.get("/api/guest/status")
    assert res.status_code == 200
    assert res.json()["enabled"] is True


def test_register_rejects_bad_invite(guest_env):
    client, _ = guest_env
    res = _register(client, invite="wrong")
    assert res.status_code == 401


def test_register_and_chat(guest_env):
    client, system = guest_env
    reg = _register(client)
    assert reg.status_code == 200
    body = reg.json()
    assert body["display_name"] == "Alex"
    assert body["returning"] is False
    token = body["guest_token"]

    chat = client.post(
        "/api/chat",
        json={"message": "what is 2+2?"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert chat.status_code == 200
    events = _parse_sse(chat.text)
    assert events[0][0] == "session"
    assert events[0][1]["role"] == "guest"
    assert events[-1][0] == "done"

    assert system.control_center.calls[-1]["user_role"] == "guest"
    assert any(k.startswith("guest:") for k in system.session_histories)


def test_chat_logs_guest_label(guest_env, caplog):
    import logging

    caplog.set_level(logging.INFO, logger="WitsV3.WebUI")
    caplog.set_level(logging.INFO, logger="uvicorn.access")
    client, _ = guest_env
    reg = _register(client, name="TESTER", device="device-tester-9001")
    token = reg.json()["guest_token"]
    client.post(
        "/api/chat",
        json={"message": "I love Minecraft"},
        headers={"Authorization": f"Bearer {token}"},
    )
    combined = " ".join(r.message for r in caplog.records)
    assert "[TESTER]" in combined
    assert "chat:" in combined.lower() or "POST /api/chat" in combined


def test_owner_admin_accounts(guest_env):
    client, _ = guest_env
    _register(client, name="Sean", device="device-sean-9001")
    res = client.get(
        "/api/guest/admin/accounts",
        headers={"Authorization": "Bearer owner-sekrit"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["enabled"] is True
    names = [g["display_name"] for g in body["guests"]]
    assert "Sean" in names


def test_guest_blocked_from_settings(guest_env):
    client, _ = guest_env
    token = _register(client).json()["guest_token"]
    res = client.get("/api/settings", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 403


def test_guest_blocked_from_memory_prune(guest_env):
    client, _ = guest_env
    token = _register(client).json()["guest_token"]
    res = client.post(
        "/api/memory/prune",
        json={"filter_dict": {"type": "x"}, "confirm": "PRUNE"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 403


def test_returning_guest_same_device(guest_env):
    client, _ = guest_env
    first = _register(client).json()
    second = _register(client).json()
    assert second["returning"] is True
    assert second["guest_id"] == first["guest_id"]


def test_owner_token_still_works(guest_env):
    client, system = guest_env
    res = client.get("/api/settings", headers={"Authorization": "Bearer owner-sekrit"})
    assert res.status_code == 200
    chat = client.post(
        "/api/chat",
        json={"message": "hi"},
        headers={"Authorization": "Bearer owner-sekrit"},
    )
    assert chat.status_code == 200
    assert system.control_center.calls[-1]["user_role"] == "owner"


def test_join_page_served(guest_env):
    client, _ = guest_env
    res = client.get("/join")
    assert res.status_code == 200
    assert b"Join WITS" in res.content


def test_guest_tools_filtered(guest_env):
    client, _ = guest_env
    token = _register(client).json()["guest_token"]
    res = client.get("/api/tools", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 200
    names = {t["name"] for t in res.json()["tools"]}
    assert names <= GUEST_ALLOWED_TOOLS | {"calculator"}  # fake registry may only have calc
    assert "ingest_documents" not in names


def test_guest_sessions_isolated_from_owner(guest_env):
    client, system = guest_env
    owner_headers = {"Authorization": "Bearer owner-sekrit"}
    guest_token = _register(client, name="Mia", device="device-mia-42").json()["guest_token"]
    guest_headers = {"Authorization": f"Bearer {guest_token}"}

    owner_chat = _parse_sse(
        client.post(
            "/api/chat",
            json={"message": "owner planning notes"},
            headers=owner_headers,
        ).text
    )
    owner_session = owner_chat[0][1]["session_id"]

    guest_chat = _parse_sse(
        client.post(
            "/api/chat",
            json={"message": "guest minecraft tips"},
            headers=guest_headers,
        ).text
    )
    guest_session = guest_chat[0][1]["session_id"]

    owner_list = client.get("/api/sessions", headers=owner_headers).json()["sessions"]
    guest_list = client.get("/api/sessions", headers=guest_headers).json()["sessions"]

    assert len(owner_list) == 1
    assert owner_list[0]["session_id"] == owner_session
    assert len(guest_list) == 1
    assert guest_list[0]["session_id"] == guest_session

    assert (
        client.get(
            f"/api/sessions/{guest_session}/messages",
            headers=owner_headers,
        ).status_code
        == 404
    )

    guest_messages = client.get(
        f"/api/sessions/{guest_session}/messages",
        headers=guest_headers,
    ).json()
    assert guest_messages["messages"][0]["content"] == "guest minecraft tips"
    assert any(k.startswith("guest:") for k in system.session_histories)


def test_token_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("WITSV3_GUEST_SECRET", "roundtrip-secret")
    monkeypatch.setenv("WITSV3_GUEST_INVITE", "x")
    tok = issue_guest_token(guest_id="g1", device_id="d1", display_name="Alex", ttl_hours=1)
    payload = validate_guest_token(tok)
    assert payload is not None
    assert payload["guest_id"] == "g1"
    assert payload["display_name"] == "Alex"


def test_guest_registry_persist(tmp_path):
    path = tmp_path / "guests.json"
    reg = GuestRegistry(path)
    p = reg.register_or_update(display_name="Alex", device_id="dev-1")
    reg2 = GuestRegistry(path)
    found = reg2.find_by_device("dev-1")
    assert found is not None
    assert found["guest_id"] == p["guest_id"]
