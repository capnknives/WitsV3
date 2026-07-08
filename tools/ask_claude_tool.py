"""
Ask-Claude escalation tool for WitsV3.

Lets the local model queue a question for Claude (Anthropic API) when it is
stuck or unsure. The request is NOT sent immediately - it waits in the web UI
for the user to approve or deny it, so no tokens are ever spent without
explicit per-request consent.
"""

from typing import Any

from core.base_tool import BaseTool
from core.config import load_config
from core.escalation import get_escalation_manager


class AskClaudeTool(BaseTool):
    """Queue a question for Claude, pending user approval in the web UI."""

    def __init__(self):
        super().__init__(
            name="ask_claude",
            description=(
                "Escalate a question to Claude (a much more capable cloud model) when you "
                "are stuck, unsure, or the task exceeds your abilities. The request is NOT "
                "sent immediately: the user must approve it in the web UI first (it costs "
                "them real money). Use sparingly - only when you genuinely cannot answer "
                "well yourself. Include enough context for Claude to answer standalone."
            ),
        )
        self.config = load_config()

    async def execute(self, question: str = "", context: str = "", **kwargs) -> dict[str, Any]:
        if not question.strip():
            return {"success": False, "error": "question is required"}

        if not self.config.escalation.enabled:
            return {
                "success": False,
                "error": "Escalation to Claude is disabled in settings.",
            }

        manager = get_escalation_manager()
        request = manager.create(
            question=question,
            context=context,
            model=self.config.escalation.model,
            max_tokens=self.config.escalation.max_tokens,
        )
        estimate = request.estimate()
        return {
            "success": True,
            "escalation_id": request.id,
            "status": "pending_user_approval",
            "message": (
                "Your question has been queued for Claude, but it will only be sent "
                "after the user approves it in the web UI (estimated worst-case cost "
                f"${estimate['max_cost_usd']}). Tell the user you've asked to escalate "
                "this question to Claude and they need to approve or deny it in the "
                "chat interface. Claude's answer will appear in the conversation once "
                "approved."
            ),
        }

    def get_schema(self) -> dict[str, Any]:
        return {
            "name": "ask_claude",
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question for Claude. Must be self-contained.",
                    },
                    "context": {
                        "type": "string",
                        "description": "Relevant context (code, errors, prior attempts) so Claude can answer without seeing the conversation.",
                    },
                },
                "required": ["question"],
            },
        }
