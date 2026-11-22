"""Red-team RT-5: guest access on + require_auth but no owner web token."""

import pytest
from fastapi.testclient import TestClient

from tests.web.test_web_server import FakeSystem
from web.server import create_app


def _build(tmp_path, monkeypatch, token):
    if token:
        monkeypatch.setenv("WITSV3_WEB_TOKEN", token)
    else:
        monkeypatch.delenv("WITSV3_WEB_TOKEN", raising=False)
    monkeypatch.setenv("WITSV3_GUEST_INVITE", "family-code")
    monkeypatch.setenv("WITSV3_GUEST_SECRET", "guest-signing-secret-xyz")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "var" / "data").mkdir(parents=True)
    (tmp_path / "var" / "exports").mkdir(parents=True)
    system = FakeSystem(tmp_path)
    system.config.web_ui.require_auth = True
    system.config.web_ui.guest_access.enabled = True
    system.config.web_ui.guest_access.allow_lan_only = True
    return TestClient(create_app(system)), system


def _calls(client, headers):
    return {
        "settings": client.get("/api/settings", headers=headers),
        "chat": client.post("/api/chat", json={"message": "hi"}, headers=headers),
        "documents": client.get("/api/documents", headers=headers),
    }


@pytest.mark.parametrize("name", ["settings", "chat", "documents"])
def test_no_token_no_header_refused(tmp_path, monkeypatch, name):
    client, system = _build(tmp_path, monkeypatch, None)
    res = _calls(client, {})[name]
    assert res.status_code in (401, 403)
    assert not system.control_center.calls


@pytest.mark.parametrize("name", ["settings", "chat", "documents"])
def test_no_token_garbage_bearer_refused(tmp_path, monkeypatch, name):
    client, system = _build(tmp_path, monkeypatch, None)
    res = _calls(client, {"Authorization": "Bearer garbage-token"})[name]
    assert res.status_code in (401, 403)
    assert not system.control_center.calls


@pytest.mark.parametrize("name", ["settings", "chat", "documents"])
def test_control_token_set_unauth_refused(tmp_path, monkeypatch, name):
    client, system = _build(tmp_path, monkeypatch, "owner-sekrit")
    res = _calls(client, {})[name]
    assert res.status_code in (401, 403)
    assert not system.control_center.calls


def test_no_guests_no_token_still_owner(tmp_path, monkeypatch):
    client, system = _build(tmp_path, monkeypatch, None)
    system.config.web_ui.guest_access.enabled = False
    client = TestClient(create_app(system))
    assert client.get("/api/settings").status_code == 200


def test_no_token_valid_guest_still_works(tmp_path, monkeypatch):
    client, _ = _build(tmp_path, monkeypatch, None)
    reg = client.post(
        "/api/guest/register",
        json={
            "invite_code": "family-code",
            "display_name": "Mom",
            "device_id": "device-abc-12345",
            "age_band": "teen",
        },
    )
    assert reg.status_code == 200, reg.text
    headers = {"Authorization": f"Bearer {reg.json()['guest_token']}"}
    assert client.get("/api/status", headers=headers).status_code == 200
    assert client.get("/api/settings", headers=headers).status_code == 403


def test_no_token_guests_on_logs_startup_warning(tmp_path, monkeypatch, caplog):
    with caplog.at_level("WARNING", logger="WitsV3.WebUI"):
        _build(tmp_path, monkeypatch, None)
    assert any("owner UI is LOCKED" in r.getMessage() for r in caplog.records)


def test_guests_on_require_auth_off_still_needs_owner_token(tmp_path, monkeypatch):
    _, system = _build(tmp_path, monkeypatch, "owner-sekrit")
    system.config.web_ui.require_auth = False
    client = TestClient(create_app(system))
    assert client.get("/api/settings").status_code == 401
    owner = {"Authorization": "Bearer owner-sekrit"}
    assert client.get("/api/settings", headers=owner).status_code == 200
