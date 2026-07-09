"""Routing tests for accumulated-knowledge queries (agents/wcca_routing_mixin.py)."""

from __future__ import annotations

from agents.wcca_routing_mixin import OrchestratorRoutingMixin


class _RoutingProbe(OrchestratorRoutingMixin):
    pass


def test_needs_knowledge_log_review_matches_recurring_bug_phrasing():
    probe = _RoutingProbe()
    assert probe._needs_knowledge_log_review("what bugs keep happening")
    assert probe._needs_knowledge_log_review("are there any recurring errors")
    assert probe._needs_knowledge_log_review("what do you know about this project")


def test_needs_knowledge_log_review_ignores_unrelated_messages():
    probe = _RoutingProbe()
    assert not probe._needs_knowledge_log_review("what's the weather like today")
    assert not probe._needs_knowledge_log_review("hello there")


def test_knowledge_log_signals_do_not_collide_with_guest_profile_signals(tmp_path, monkeypatch):
    """Recurring-bug phrasing must not get misrouted into guest-profile lookup,
    and vice versa — even with a registered guest whose name could coincide."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    from core.guest_access import GuestRegistry

    reg = GuestRegistry()
    reg.register_or_update(display_name="Christina", device_id="device-ddd-44444")

    probe = _RoutingProbe()
    assert probe._needs_knowledge_log_review("what bugs keep happening")
    assert not probe._needs_guest_profile_review("what bugs keep happening")

    assert probe._needs_guest_profile_review("tell me about the user christina")
    assert not probe._needs_knowledge_log_review("tell me about the user christina")
