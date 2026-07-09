"""Tests for labeled HTTP access logs."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from core.config import WitsV3Config
from web.access_log import owner_display_name, resolve_caller_label


@pytest.fixture
def config():
    return WitsV3Config()


def _request(path: str, token: str | None = None, *, auth_role=None, guest=None):
    req = MagicMock()
    req.url.path = path
    req.state = MagicMock()
    req.state.auth_role = auth_role
    req.state.guest = guest
    headers = {}
    if token:
        headers["authorization"] = f"Bearer {token}"
    req.headers = headers
    return req


def test_owner_display_name_from_env(config, monkeypatch):
    monkeypatch.setenv("WITSV3_OWNER_NAME", "Richard")
    assert owner_display_name(config) == "Richard"


def test_owner_display_name_config_default(config, monkeypatch):
    monkeypatch.delenv("WITSV3_OWNER_NAME", raising=False)
    config.web_ui.owner_display_name = "Richard"
    assert owner_display_name(config) == "Richard"


def test_resolve_caller_label_owner(config, monkeypatch):
    monkeypatch.setenv("WITSV3_WEB_TOKEN", "owner-tok")
    monkeypatch.setenv("WITSV3_OWNER_NAME", "Richard")
    req = _request("/api/status", "owner-tok", auth_role="owner")
    assert resolve_caller_label(req, config) == "Richard"


def test_resolve_caller_label_guest(config, monkeypatch):
    monkeypatch.setenv("WITSV3_GUEST_SECRET", "guest-secret")
    monkeypatch.setenv("WITSV3_GUEST_INVITE", "invite")
    from core.guest_access import GuestRegistry, issue_guest_token

    reg = GuestRegistry()
    reg._data = {
        "guests": {
            "g1": {
                "guest_id": "g1",
                "display_name": "Sean",
                "device_ids": ["device-12345678"],
                "revoked": False,
            }
        }
    }
    token = issue_guest_token(
        guest_id="g1", device_id="device-12345678", display_name="OLD_TOKEN_NAME"
    )
    req = _request(
        "/api/chat",
        token,
        auth_role="guest",
        guest={"guest_id": "g1", "display_name": "OLD_TOKEN_NAME", "device_id": "device-12345678"},
    )
    assert resolve_caller_label(req, config, reg) == "Sean"


def test_resolve_caller_label_register_preset(config):
    req = _request("/api/guest/register", auth_role=None, guest=None)
    req.state.caller_label = "Sean"
    assert resolve_caller_label(req, config) == "Sean"


def test_resolve_caller_label_static_page(config):
    req = _request("/", auth_role=None, guest=None)
    assert resolve_caller_label(req, config) == "-"


def test_resolve_caller_label_anon_api(config, monkeypatch):
    monkeypatch.setenv("WITSV3_WEB_TOKEN", "owner-tok")
    req = _request("/api/status", auth_role=None, guest=None)
    assert resolve_caller_label(req, config) == "anon"


def test_log_http_access_format(config, caplog):
    import logging

    from web.access_log import log_http_access

    caplog.set_level(logging.INFO, logger="uvicorn.access")
    req = MagicMock()
    req.url.path = "/api/chat"
    req.url.query = ""
    req.method = "POST"
    req.client = MagicMock(host="127.0.0.1", port=51234)
    req.state = MagicMock()
    req.state.caller_label = "TESTER"
    log_http_access(req, 200, config)
    assert any("[TESTER]" in r.message and "POST /api/chat" in r.message for r in caplog.records)
