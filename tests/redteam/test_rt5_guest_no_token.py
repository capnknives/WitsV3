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


@pytest.mark.xfail(strict=True, reason="RT-5: no web token => middleware treats anon as owner, 200")
@pytest.mark.parametrize("name", ["settings", "chat", "documents"])
def test_no_token_no_header_refused(tmp_path, monkeypatch, name):
    client, system = _build(tmp_path, monkeypatch, None)
    res = _calls(client, {})[name]
    assert res.status_code in (401, 403)
    assert not system.control_center.calls


@pytest.mark.xfail(
    strict=True, reason="RT-5: no web token => garbage bearer still treated as owner"
)
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
