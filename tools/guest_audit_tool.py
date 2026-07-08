"""Owner-only tool to read and summarize guest / family-tester audit logs."""

from __future__ import annotations

from typing import Any

from core.base_tool import BaseTool
from core.guest_access import GuestRegistry, format_active_guest_accounts
from core.guest_audit import GuestAuditLog, build_owner_audit_digest


class GuestAccountsListTool(BaseTool):
    """List active /join guest accounts for the owner."""

    def __init__(self):
        super().__init__(
            name="guest_accounts_list",
            description=(
                "Owner-only: list all active guest/family-tester accounts registered via /join. "
                "Use when the owner asks who has joined, to list guest accounts, active testers, "
                "or family members with access. For chat activity summaries use guest_audit_summary."
            ),
        )
        self.registry = GuestRegistry()

    async def execute(self, include_revoked: bool = False, **kwargs: Any) -> str:
        user_role = kwargs.get("user_role", "owner")
        if user_role == "guest":
            return (
                "guest_accounts_list is only available to the owner. "
                "Guests cannot list other accounts."
            )
        try:
            report = format_active_guest_accounts(
                self.registry, include_revoked=include_revoked
            )
            self.logger.info("Guest accounts list for owner (revoked=%s)", include_revoked)
            return report
        except Exception as e:
            self.logger.error("Guest accounts list failed: %s", e, exc_info=True)
            return f"Error listing guest accounts: {e}"

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "include_revoked": {
                    "type": "boolean",
                    "description": "If true, also list revoked guest accounts.",
                    "default": False,
                },
            },
            "required": [],
        }


class GuestAuditSummaryTool(BaseTool):
    """Summarize guest activity from data/guest_audit for the owner."""

    def __init__(self):
        super().__init__(
            name="guest_audit_summary",
            description=(
                "Owner-only: summarize guest/family-tester chat audit logs. "
                "Use when the owner asks what a guest did or wants activity/details for a "
                "specific tester. To list who has joined (account roster), use "
                "guest_accounts_list instead."
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


class GuestSetAgeBandTool(BaseTool):
    """Owner assigns child / teen / adult protection tier for a guest."""

    def __init__(self):
        super().__init__(
            name="guest_set_age_band",
            description=(
                "Owner-only: set a guest's age/protection tier (child, teen, or adult). "
                "Use when the owner says someone is a teen/kid/adult, e.g. set Sean to teen "
                "or set my wife to adult. Guests cannot change their own tier."
            ),
        )
        self.registry = GuestRegistry()

    async def execute(
        self,
        age_band: str,
        display_name: str | None = None,
        guest_id: str | None = None,
        **kwargs: Any,
    ) -> str:
        user_role = kwargs.get("user_role", "owner")
        if user_role == "guest":
            return "guest_set_age_band is only available to the owner."

        from core.content_policy import age_band_description, normalize_age_band

        band = normalize_age_band(age_band)
        if guest_id:
            profile = self.registry.set_age_band(guest_id.strip(), band)
        elif display_name:
            profile = self.registry.set_age_band_by_name(display_name.strip(), band)
        else:
            return "Provide display_name or guest_id to set a guest's age band."

        if not profile:
            return f"No guest found for {display_name or guest_id}."

        name = profile.get("display_name", "?")
        return (
            f"Set {name} to age band '{band}' ({age_band_description(band)}). "
            "Content filters and tool access now follow this owner-assigned tier."
        )

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "age_band": {
                    "type": "string",
                    "description": "child, teen, or adult (owner-assigned protection tier).",
                },
                "display_name": {
                    "type": "string",
                    "description": "Guest /join display name (e.g. Sean).",
                },
                "guest_id": {
                    "type": "string",
                    "description": "Optional guest UUID if known.",
                },
            },
            "required": ["age_band"],
        }
