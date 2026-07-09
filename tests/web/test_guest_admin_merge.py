"""Tests for guest account dedup, merge, revoke, and profile-only owner queries."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from agents.wcca_routing_mixin import OrchestratorRoutingMixin
from core.guest_access import GuestRegistry
from core.guest_user_profile import GuestUserProfileStore
from tests.web.test_web_server import FakeSystem
from web.server import create_app


class _RoutingProbe(OrchestratorRoutingMixin):
    pass


@pytest.fixture
def guest_env(tmp_path, monkeypatch):
    monkeypatch.setenv("WITSV3_WEB_TOKEN", "owner-sekrit")
    monkeypatch.setenv("WITSV3_GUEST_INVITE", "family-code")
    monkeypatch.setenv("WITSV3_GUEST_SECRET", "guest-signing-secret-xyz")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    system = FakeSystem(tmp_path)
    system.config.web_ui.guest_access.enabled = True
    client = TestClient(create_app(system))
    return client, system


def test_register_same_name_reuses_account(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    reg = GuestRegistry()
    first = reg.register_or_update(display_name="TESTER", device_id="device-aaa-11111")
    second = reg.register_or_update(display_name="TESTER", device_id="device-bbb-22222")
    assert first["guest_id"] == second["guest_id"]
    assert len(reg.find_all_by_display_name("TESTER")) == 1
    assert len(second["device_ids"]) == 2


def test_merge_guests_combines_devices_and_profiles(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    reg = GuestRegistry()
    a = reg.register_or_update(display_name="TESTER", device_id="device-aaa-11111")
    # Force a duplicate (legacy data shape)
    reg._data["guests"]["dup-id"] = {
        "guest_id": "dup-id",
        "display_name": "TESTER",
        "device_ids": ["device-ccc-33333"],
        "age_band": "teen",
        "first_seen": 1,
        "last_seen": 2,
        "revoked": False,
    }
    reg._save()

    store = GuestUserProfileStore()
    store.update_from_turn(
        guest_id=a["guest_id"],
        display_name="TESTER",
        user_message="I love Minecraft",
    )
    store.update_from_turn(
        guest_id="dup-id",
        display_name="TESTER",
        user_message="I like coding",
    )

    merged = reg.merge_guests(target_guest_id=a["guest_id"], source_guest_id="dup-id")
    assert merged is not None
    assert "device-ccc-33333" in merged["device_ids"]
    profile = store.load_merged_for_display_name("TESTER", reg)
    assert profile["turn_count"] >= 2
    interests = profile.get("interests") or {}
    assert "Minecraft" in interests or "Coding" in interests


def test_owner_merge_api(guest_env, tmp_path):
    import json

    client, system = guest_env
    r1 = client.post(
        "/api/guest/register",
        json={
            "invite_code": "family-code",
            "display_name": "TESTER",
            "device_id": "device-aaa-11111",
        },
    )
    gid = r1.json()["guest_id"]
    path = tmp_path / "data" / "guest_profiles.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["guests"]["dup-id"] = {
        "guest_id": "dup-id",
        "display_name": "TESTER",
        "device_ids": ["device-ccc-33333"],
        "age_band": "teen",
        "first_seen": 1,
        "last_seen": 2,
        "revoked": False,
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    client2 = TestClient(create_app(system))
    res = client2.post(
        "/api/guest/admin/merge",
        json={"target_guest_id": gid, "source_guest_id": "dup-id"},
        headers={"Authorization": "Bearer owner-sekrit"},
    )
    assert res.status_code == 200
    assert res.json()["merged"] is True
    reg = GuestRegistry()
    assert len(reg.find_all_by_display_name("TESTER")) == 1


def test_owner_revoke_api(guest_env, tmp_path):
    client, system = guest_env
    r1 = client.post(
        "/api/guest/register",
        json={
            "invite_code": "family-code",
            "display_name": "TESTER",
            "device_id": "device-aaa-11111",
        },
    )
    gid = r1.json()["guest_id"]
    client2 = TestClient(create_app(system))
    res = client2.request(
        "DELETE",
        "/api/guest/admin/account",
        json={"guest_id": gid},
        headers={"Authorization": "Bearer owner-sekrit"},
    )
    assert res.status_code == 200
    reg = GuestRegistry()
    assert reg.get(gid) is None


def test_owner_edit_profile_facts_api(guest_env):
    client, system = guest_env
    r1 = client.post(
        "/api/guest/register",
        json={
            "invite_code": "family-code",
            "display_name": "Christina",
            "device_id": "device-aaa-11111",
        },
    )
    gid = r1.json()["guest_id"]
    store = GuestUserProfileStore()
    store.update_from_turn(
        guest_id=gid,
        display_name="Christina",
        user_message="I am Richard's wife.",
        assistant_message="Hi Christina!",
    )

    client2 = TestClient(create_app(system))
    res = client2.patch(
        "/api/guest/admin/profile/facts",
        json={"guest_id": gid, "facts": ["Is Richard's wife"]},
        headers={"Authorization": "Bearer owner-sekrit"},
    )
    assert res.status_code == 200
    body = res.json()
    facts = [f["text"] for f in body["profile"]["facts"]]
    assert facts == ["Is Richard's wife"]

    profile = store.load_merged_for_display_name("Christina")
    assert [f["text"] for f in profile["facts"]] == ["Is Richard's wife"]


def test_guest_cannot_edit_profile_facts_api(guest_env):
    client, system = guest_env
    r1 = client.post(
        "/api/guest/register",
        json={
            "invite_code": "family-code",
            "display_name": "Christina",
            "device_id": "device-aaa-11111",
        },
    )
    gid = r1.json()["guest_id"]
    guest_token = r1.json()["guest_token"]

    client2 = TestClient(create_app(system))
    res = client2.patch(
        "/api/guest/admin/profile/facts",
        json={"guest_id": gid, "facts": ["whatever I want"]},
        headers={"Authorization": f"Bearer {guest_token}"},
    )
    assert res.status_code == 403


def test_profile_query_signals(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    reg = GuestRegistry()
    reg.register_or_update(display_name="TESTER", device_id="device-aaa-11111")

    probe = _RoutingProbe()
    assert probe._needs_guest_profile_review("what does the system know about TESTER")
    assert probe._extract_guest_name_for_profile_query("what do you know about TESTER") == "TESTER"


@pytest.mark.asyncio
async def test_wcca_profile_query_skips_web_search_path():
    probe = _RoutingProbe()
    assert await probe._requires_orchestrator_for_input("what does the system know about TESTER")
