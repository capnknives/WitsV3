"""Tests for guest audit summary tool and owner routing."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from agents.wcca_routing_mixin import OrchestratorRoutingMixin
from core.guest_access import GuestRegistry
from core.guest_audit import GuestAuditLog, build_owner_audit_digest
from tools.guest_audit_tool import GuestAuditSummaryTool


class _RoutingProbe(OrchestratorRoutingMixin):
    pass


@pytest.mark.asyncio
async def test_guest_audit_summary_tool_owner_reads_logs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()

    reg = GuestRegistry()
    profile = reg.register_or_update(display_name="TESTER", device_id="dev-tester-001")
    guest_id = profile["guest_id"]

    audit = GuestAuditLog()
    audit.log(
        guest_id=guest_id,
        event_type="chat_user",
        display_name="TESTER",
        content="what is 2+2?",
    )
    audit.log(
        guest_id=guest_id,
        event_type="content_blocked",
        display_name="TESTER",
        content="search for porn",
        meta={"direction": "input"},
    )

    tool = GuestAuditSummaryTool()
    tool.audit = audit
    report = await tool.execute(display_name="TESTER", days=1, user_role="owner")

    assert "TESTER" in report
    assert "what is 2+2?" in report
    assert "content_blocked" in report or "Blocked content" in report
    assert "search for porn" in report


@pytest.mark.asyncio
async def test_guest_audit_summary_denied_for_guest_role():
    tool = GuestAuditSummaryTool()
    report = await tool.execute(display_name="TESTER", user_role="guest")
    assert "only available to the owner" in report


@pytest.mark.asyncio
async def test_wcca_routes_guest_log_questions_to_orchestrator():
    probe = _RoutingProbe()
    assert await probe._requires_orchestrator_for_input("summarize TESTER guest logs from today")
    assert await probe._requires_orchestrator_for_input("what did my nephew ask?")


def test_build_owner_audit_digest_all_guests(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    reg = GuestRegistry()
    p = reg.register_or_update(display_name="Alex", device_id="dev-alex-001")
    audit = GuestAuditLog()
    audit.log(guest_id=p["guest_id"], event_type="register", display_name="Alex")

    digest = build_owner_audit_digest(include_all_guests=True, days=1, audit=audit)
    assert "Alex" in digest
    assert "register" in digest
