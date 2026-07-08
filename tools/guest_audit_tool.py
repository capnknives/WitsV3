"""Owner-only tool to read and summarize guest / family-tester audit logs."""

from __future__ import annotations

from typing import Any

from core.base_tool import BaseTool
from core.guest_audit import GuestAuditLog, build_owner_audit_digest


class GuestAuditSummaryTool(BaseTool):
    """Summarize guest activity from data/guest_audit for the owner."""

    def __init__(self):
        super().__init__(
            name="guest_audit_summary",
            description=(
                "Owner-only: summarize guest/family-tester chat audit logs. "
                "Use when the owner asks what a guest did, wants a log summary, "
                "or asks about TESTER or other join users. Returns recent messages, "
                "tool calls, and any blocked inappropriate content."
            ),
        )
        self.audit = GuestAuditLog()

    async def execute(
        self,
        display_name: str | None = None,
        guest_id: str | None = None,
        days: int = 1,
        include_all_guests: bool = False,
        **kwargs: Any,
    ) -> str:
        user_role = kwargs.get("user_role", "owner")
        if user_role == "guest":
            return (
                "guest_audit_summary is only available to the owner. "
                "Guests cannot view other users' activity logs."
            )

        try:
            report = build_owner_audit_digest(
                display_name=display_name,
                guest_id=guest_id,
                days=days,
                include_all_guests=include_all_guests,
                audit=self.audit,
            )
            self.logger.info(
                "Guest audit summary for owner (name=%s, all=%s, days=%s)",
                display_name,
                include_all_guests,
                days,
            )
            return report
        except Exception as e:
            self.logger.error("Guest audit summary failed: %s", e, exc_info=True)
            return f"Error reading guest audit logs: {e}"

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "display_name": {
                    "type": "string",
                    "description": "Guest display name to summarize (e.g. TESTER, Alex). Omit with include_all_guests for everyone.",
                },
                "guest_id": {
                    "type": "string",
                    "description": "Optional guest UUID if known.",
                },
                "days": {
                    "type": "integer",
                    "description": "How many days of audit history to include (default 1, max 30).",
                    "default": 1,
                },
                "include_all_guests": {
                    "type": "boolean",
                    "description": "If true, summarize all guests (ignores display_name unless set).",
                    "default": False,
                },
            },
            "required": [],
        }
