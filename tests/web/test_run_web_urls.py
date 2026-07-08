"""Tests for run_web startup URL hints (owner vs guest LAN links)."""

from core.config import WitsV3Config
from run_web import startup_urls


def test_startup_urls_guest_mode_uses_join_not_owner_token(monkeypatch):
    monkeypatch.setenv("WITSV3_GUEST_INVITE", "family-code")
    monkeypatch.setenv("WITSV3_GUEST_SECRET", "signing-secret")
    config = WitsV3Config()
    config.web_ui.guest_access.enabled = True

    localhost, lan = startup_urls(config, 8000, "owner-secret-token")

    assert "owner-secret-token" in localhost
    assert "owner_token=" in localhost
    assert lan.endswith("/join")
    assert "owner-secret-token" not in lan
    assert "token=" not in lan


def test_startup_urls_without_guest_uses_owner_token_on_lan(monkeypatch):
    monkeypatch.delenv("WITSV3_GUEST_INVITE", raising=False)
    monkeypatch.delenv("WITSV3_GUEST_SECRET", raising=False)
    config = WitsV3Config()
    config.web_ui.guest_access.enabled = False

    localhost, lan = startup_urls(config, 8000, "owner-secret-token")

    assert "owner_token=" in localhost
    assert "owner_token=" in lan


def test_startup_urls_guest_mode_without_web_token(monkeypatch):
    monkeypatch.setenv("WITSV3_GUEST_INVITE", "family-code")
    monkeypatch.setenv("WITSV3_GUEST_SECRET", "signing-secret")
    config = WitsV3Config()
    config.web_ui.guest_access.enabled = True

    localhost, lan = startup_urls(config, 8000, "")

    assert localhost.endswith("/")
    assert lan.endswith("/join")
