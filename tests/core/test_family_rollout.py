"""Family tester rollout: RBAC roles, /join sessions, Downloads access."""

from __future__ import annotations

import sys

import pytest

from core.filesystem_policy import read_roots_for_role, resolve_allowed_read_path
from core.guest_access import (
    GuestRegistry,
    is_tool_allowed,
    resolve_effective_role,
    tools_for_role,
)
from tests.web.test_guest_access import _register

pytest_plugins = ("tests.web.test_guest_access",)


def test_family_adult_read_roots_include_downloads(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "core.filesystem_policy.PROJECT_ROOT",
        tmp_path,
        raising=False,
    )
    downloads = tmp_path.parent / "Downloads"
    downloads.mkdir(exist_ok=True)
    monkeypatch.setattr(
        "core.filesystem_policy.configured_read_roots",
        lambda config=None: [tmp_path.resolve(), downloads.resolve()],
    )
    roots = read_roots_for_role("family_adult", config=None)
    assert tmp_path.resolve() in roots
    assert downloads.resolve() in roots


def test_family_kid_read_roots_project_only(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "core.filesystem_policy.PROJECT_ROOT",
        tmp_path,
        raising=False,
    )
    downloads = tmp_path.parent / "Downloads"
    downloads.mkdir(exist_ok=True)
    monkeypatch.setattr(
        "core.filesystem_policy.configured_read_roots",
        lambda config=None: [tmp_path.resolve(), downloads.resolve()],
    )
    roots = read_roots_for_role("family_kid", config=None)
    assert roots == [tmp_path.resolve()]


def test_family_kid_denied_downloads_file(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "core.filesystem_policy.PROJECT_ROOT",
        tmp_path,
        raising=False,
    )
    downloads = tmp_path.parent / "Downloads"
    downloads.mkdir(exist_ok=True)
    secret = downloads / "secret.pdf"
    secret.write_text("x", encoding="utf-8")
    monkeypatch.setattr(
        "core.filesystem_policy.configured_read_roots",
        lambda config=None: [tmp_path.resolve(), downloads.resolve()],
    )
    with pytest.raises(PermissionError):
        resolve_allowed_read_path(str(secret), role="family_kid", config=None)


def test_family_adult_can_read_downloads_file(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "core.filesystem_policy.PROJECT_ROOT",
        tmp_path,
        raising=False,
    )
    downloads = tmp_path.parent / "Downloads"
    downloads.mkdir(exist_ok=True)
    doc = downloads / "notes.txt"
    doc.write_text("hello", encoding="utf-8")
    monkeypatch.setattr(
        "core.filesystem_policy.configured_read_roots",
        lambda config=None: [tmp_path.resolve(), downloads.resolve()],
    )
    resolved = resolve_allowed_read_path(str(doc), role="family_adult", config=None)
    assert resolved == doc.resolve()


def test_guest_denied_downloads_outside_project(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "core.filesystem_policy.PROJECT_ROOT",
        tmp_path,
        raising=False,
    )
    outside = (
        "C:/Windows/win.ini" if sys.platform == "win32" else "/etc/hosts"
    )
    with pytest.raises(PermissionError):
        resolve_allowed_read_path(outside, role="guest", config=None)


def test_family_kid_cannot_write_files():
    assert not is_tool_allowed("write_file", "family_kid")
    assert is_tool_allowed("read_file", "family_kid")


def test_family_adult_has_read_tools_not_write():
    allowed = tools_for_role("family_adult")
    assert allowed is not None
    assert "read_file" in allowed
    assert "list_directory" in allowed
    assert "write_file" not in allowed


def test_join_adult_age_band_maps_to_family_adult(guest_env, tmp_path, monkeypatch):
    client, _ = guest_env
    monkeypatch.chdir(tmp_path)
    (tmp_path / "var" / "data").mkdir(parents=True, exist_ok=True)

    reg = client.post(
        "/api/guest/register",
        json={
            "invite_code": "family-code",
            "display_name": "Mom",
            "device_id": "mom-device-001",
            "age_band": "teen",
        },
    )
    assert reg.status_code == 200
    guest_id = reg.json()["guest_id"]

    owner = "owner-sekrit"
    patch = client.patch(
        "/api/guest/admin/age-band",
        json={"display_name": "Mom", "age_band": "adult"},
        headers={"Authorization": f"Bearer {owner}"},
    )
    assert patch.status_code == 200

    registry = GuestRegistry()
    profile = registry.get(guest_id)
    assert profile is not None
    assert resolve_effective_role("guest", profile) == "family_adult"


def test_join_child_age_band_maps_to_family_kid(guest_env, tmp_path, monkeypatch):
    client, _ = guest_env
    monkeypatch.chdir(tmp_path)
    (tmp_path / "var" / "data").mkdir(parents=True, exist_ok=True)

    reg = client.post(
        "/api/guest/register",
        json={
            "invite_code": "family-code",
            "display_name": "Kid",
            "device_id": "kid-device-001",
            "age_band": "teen",
        },
    )
    assert reg.status_code == 200
    guest_id = reg.json()["guest_id"]

    owner = "owner-sekrit"
    patch = client.patch(
        "/api/guest/admin/age-band",
        json={"display_name": "Kid", "age_band": "child"},
        headers={"Authorization": f"Bearer {owner}"},
    )
    assert patch.status_code == 200

    registry = GuestRegistry()
    profile = registry.get(guest_id)
    assert profile is not None
    assert resolve_effective_role("guest", profile) == "family_kid"


def test_guest_chat_uses_isolated_session(guest_env):
    client, system = guest_env
    reg = _register(client, name="Sean", device="sean-rollout-001")
    token = reg.json()["guest_token"]
    guest_id = reg.json()["guest_id"]

    chat = client.post(
        "/api/chat",
        json={"message": "hello"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert chat.status_code == 200

    assert system.control_center.calls[-1]["user_role"] == "guest"
    assert any(k.startswith(f"guest:{guest_id}:") for k in system.session_histories)
