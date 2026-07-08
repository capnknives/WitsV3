"""Tests for per-guest user profile store and owner summary tool."""

from __future__ import annotations

import json

import pytest

from agents.wcca_routing_mixin import OrchestratorRoutingMixin
from core.guest_access import GuestRegistry
from core.guest_user_profile import GuestUserProfileStore
from tools.guest_profile_tool import GuestUserProfileSummaryTool


class _RoutingProbe(OrchestratorRoutingMixin):
    pass


def test_profile_update_extracts_interests_and_facts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = GuestUserProfileStore()
    profile = store.update_from_turn(
        guest_id="gid-sean",
        display_name="Sean",
        user_message="I love Minecraft and building redstone farms. I'm interested in coding too.",
        assistant_message="That sounds fun!",
    )
    assert profile["turn_count"] == 1
    assert profile["interests"]["Minecraft"] >= 1
    assert profile["interests"]["Minecraft redstone"] >= 1
    assert profile["interests"]["Coding"] >= 1
    assert any("love Minecraft" in f["text"] for f in profile["facts"])

    path = tmp_path / "data" / "guest_user_profiles" / "gid-sean.json"
    assert path.is_file()
    reloaded = json.loads(path.read_text(encoding="utf-8"))
    assert reloaded["display_name"] == "Sean"


def test_personalization_block_after_turns(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = GuestUserProfileStore()
    store.update_from_turn(
        guest_id="gid-t",
        display_name="TESTER",
        user_message="I play Fortnite every day",
        assistant_message="Cool!",
    )
    block = store.personalization_block("gid-t", "TESTER")
    assert "TESTER" in block
    assert "Fortnite" in block


@pytest.mark.asyncio
async def test_guest_user_profile_summary_tool(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()

    reg = GuestRegistry()
    acct = reg.register_or_update(display_name="Sean", device_id="dev-sean-001")
    guest_id = acct["guest_id"]

    store = GuestUserProfileStore()
    store.update_from_turn(
        guest_id=guest_id,
        display_name="Sean",
        user_message="My favorite game is Minecraft survival mode",
        assistant_message="Nice!",
    )

    tool = GuestUserProfileSummaryTool()
    tool.store = store
    tool.registry = reg
    report = await tool.execute(display_name="Sean", user_role="owner")

    assert "Sean" in report
    assert "Minecraft" in report
    assert "Conversation turns" in report


@pytest.mark.asyncio
async def test_guest_user_profile_summary_denied_for_guest():
    tool = GuestUserProfileSummaryTool()
    report = await tool.execute(display_name="Sean", user_role="guest")
    assert "only available to the owner" in report


@pytest.mark.asyncio
async def test_wcca_routes_guest_profile_questions_to_orchestrator():
    probe = _RoutingProbe()
    assert await probe._requires_orchestrator_for_input(
        "What is Sean interested in from our guest conversations?"
    )
    assert await probe._requires_orchestrator_for_input(
        "What do we know about TESTER's hobbies?"
    )
