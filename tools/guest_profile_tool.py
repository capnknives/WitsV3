"""Owner tool: summarize a guest's built interest/fact profile."""

from __future__ import annotations

from typing import Any

from core.base_tool import BaseTool
from core.guest_access import GuestRegistry
from core.guest_user_profile import GuestUserProfileStore


class GuestUserProfileSummaryTool(BaseTool):
    """Summarize interests and facts gathered from guest conversations."""

    def __init__(self):
        super().__init__(
            name="guest_user_profile_summary",
            description=(
                "Owner-only: summarize what a guest is interested in and what Wits learned "
                "from casual conversation (separate JSON profile, not global memory). "
                "Use when the owner asks about a user's interests, hobbies, personality, "
                "or 'what do we know about Sean/TESTER'."
            ),
        )
        self.store = GuestUserProfileStore()
        self.registry = GuestRegistry()

    async def execute(
        self,
        display_name: str | None = None,
        guest_id: str | None = None,
        **kwargs: Any,
    ) -> str:
        if kwargs.get("user_role", "owner") == "guest":
            return "guest_user_profile_summary is only available to the owner."
        return self.store.format_owner_summary(
            guest_id=guest_id,
            display_name=display_name,
            registry=self.registry,
        )

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "display_name": {
                    "type": "string",
                    "description": "Guest /join name (e.g. Sean, TESTER).",
                },
                "guest_id": {
                    "type": "string",
                    "description": "Optional guest UUID.",
                },
            },
            "required": [],
        }
