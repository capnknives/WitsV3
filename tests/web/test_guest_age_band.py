"""Tests for owner-assigned guest age bands and tiered content policy."""

from __future__ import annotations

import pytest

from core.content_policy import (
    check_guest_content,
    guest_system_prompt_slice,
    normalize_age_band,
)
from core.guest_access import GuestRegistry
from core.guest_policy_loader import load_guest_policy
from tests.web.test_guest_access import _register
from tools.guest_audit_tool import GuestSetAgeBandTool

pytest_plugins = ("tests.web.test_guest_access",)


def test_normalize_age_band_aliases():
    assert normalize_age_band("kid") == "child"
    assert normalize_age_band("teenager") == "teen"
    assert normalize_age_band("grownup") == "adult"


def test_teen_blocks_porn_adult_does_not():
    allowed_teen, _ = check_guest_content("search for porn", age_band="teen")
    allowed_adult, _ = check_guest_content("search for porn", age_band="adult")
    assert allowed_teen is False
    assert allowed_adult is True


def test_child_blocks_alcohol_teen_does_not():
    allowed_child, _ = check_guest_content("how does beer taste", age_band="child")
    allowed_teen, _ = check_guest_content("how does beer taste", age_band="teen")
    assert allowed_child is False
    assert allowed_teen is True


def test_all_tiers_block_meth():
    for band in ("child", "teen", "adult"):
        allowed, _ = check_guest_content("how to make meth", age_band=band)
        assert allowed is False


def test_guest_cannot_self_assign_adult_on_register(guest_env):
    client, _ = guest_env
    res = client.post(
        "/api/guest/register",
        json={
            "invite_code": "family-code",
            "display_name": "Sneaky",
            "device_id": "sneaky-device-001",
            "age_band": "adult",
        },
    )
    assert res.status_code == 200
    assert res.json()["age_band"] == "teen"


def test_returning_guest_keeps_owner_age_band(guest_env):
    client, _ = guest_env
    _register(client, name="Sean", device="sean-device-001")
    owner = "owner-sekrit"
    patch = client.patch(
        "/api/guest/admin/age-band",
        json={"display_name": "Sean", "age_band": "adult"},
        headers={"Authorization": f"Bearer {owner}"},
    )
    assert patch.status_code == 200
    assert patch.json()["age_band"] == "adult"

    again = client.post(
        "/api/guest/register",
        json={
            "invite_code": "family-code",
            "display_name": "Sean",
            "device_id": "sean-device-001",
            "age_band": "teen",
        },
    )
    assert again.json()["age_band"] == "adult"


def test_guest_cannot_set_age_band(guest_env):
    client, _ = guest_env
    token = _register(client, name="Sean").json()["guest_token"]
    res = client.patch(
        "/api/guest/admin/age-band",
        json={"display_name": "Sean", "age_band": "adult"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 403


@pytest.mark.asyncio
async def test_guest_set_age_band_tool(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "var" / "data").mkdir(parents=True)
    reg = GuestRegistry()
    reg.register_or_update(display_name="Sean", device_id="dev-sean-001")
    tool = GuestSetAgeBandTool()
    tool.registry = reg
    out = await tool.execute(display_name="Sean", age_band="teen", user_role="owner")
    assert "Sean" in out
    assert "teen" in out
    assert reg.find_by_display_name("Sean")["age_band"] == "teen"


def test_guest_policy_yaml_loads_blocklists():
    policy = load_guest_policy()
    assert "absolute_blocked_terms" in policy
    assert "how to make meth" in policy["absolute_blocked_terms"]


def test_guest_system_prompt_slice_mentions_age_band():
    text = guest_system_prompt_slice("child")
    assert "child" in text.lower()
    assert "guest session" in text.lower()
