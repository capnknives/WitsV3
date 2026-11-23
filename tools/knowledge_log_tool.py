"""Owner tools: read and add to the cross-session knowledge log.

See core/knowledge_log.py for the store (recurring errors + durable facts).
"""

from __future__ import annotations

from typing import Any

from core.base_tool import BaseTool
from core.guest_user_profile import GuestUserProfileStore
from core.knowledge_log import KnowledgeLogStore
from core.roles import normalize_role
from core.tool_context import tool_user_role


class KnowledgeLogSummaryTool(BaseTool):
    """Summarize recurring errors, durable facts, and shared guest interests."""

    def __init__(self):
        super().__init__(
            name="knowledge_log_summary",
            description=(
                "Owner-only: summarize accumulated project knowledge — recurring "
                "errors/bugs seen across self-repair scans, durable saved facts about "
                "the project, and interests shared by multiple guests. Use when the "
                "owner asks what bugs keep happening, what Wits knows about the "
                "project, or for accumulated knowledge."
            ),
        )
        self.store = KnowledgeLogStore()
        self.guest_store = GuestUserProfileStore()

    async def execute(self, **kwargs: Any) -> str:
        if normalize_role(tool_user_role(kwargs.get("user_role", "owner"))) == "guest":
            return "knowledge_log_summary is only available to the owner."
        try:
            guest_summaries = self.guest_store.list_profile_summaries()
        except Exception:
            guest_summaries = []
        return self.store.format_owner_summary(guest_profile_summaries=guest_summaries)

    def get_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}


class KnowledgeLogAddFactTool(BaseTool):
    """Save a durable, owner-confirmed fact about the project."""

    def __init__(self):
        super().__init__(
            name="knowledge_log_add_fact",
            description=(
                "Owner-only: save a lasting fact about the project or owner when explicitly asked "
                "to remember it. Not for guest details or one-off chat."
            ),
        )
        self.store = KnowledgeLogStore()

    async def execute(self, text: str = "", category: str = "project", **kwargs: Any) -> str:
        if normalize_role(tool_user_role(kwargs.get("user_role", "owner"))) == "guest":
            return "knowledge_log_add_fact is only available to the owner."
        text = (text or "").strip()
        if not text:
            return "No fact text provided."
        added = self.store.add_fact(text, source="owner", category=category or "project")
        return f"Saved fact: {text}" if added else "That fact is already recorded."

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "one short sentence",
                },
                "category": {
                    "type": "string",
                    "description": "'project' (default), 'preference' or 'other'",
                    "default": "project",
                },
            },
            "required": ["text"],
        }
