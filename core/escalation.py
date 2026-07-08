# core/escalation.py
"""
Ask-Claude escalation for WitsV3.

When the local model is stuck, the ask_claude tool queues an escalation
request here. NOTHING is sent to the Anthropic API until the user approves
the request in the web UI (each request individually — there is no
"always allow"). Deny is always available and costs nothing.

The API key is read from the ANTHROPIC_API_KEY environment variable
(loaded from the gitignored .env by core.config).
"""

import asyncio
import builtins
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("WitsV3.Escalation")

# $ per million tokens (input, output) — used for the pre-approval estimate
# shown in the UI. Prices as of mid-2026.
MODEL_PRICES = {
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-fable-5": (10.00, 50.00),
}

SYSTEM_PROMPT = (
    "You are Claude, being consulted by WITS, a local Ollama-based assistant "
    "that got stuck and escalated a question to you (with its user's approval). "
    "Answer the question directly and concisely so the local model can continue. "
    "Do not ask follow-up questions - give your best answer with stated assumptions."
)


@dataclass
class EscalationRequest:
    id: str
    question: str
    context: str = ""
    status: str = "pending"  # pending | approved | denied | answered | failed
    created_at: float = field(default_factory=time.time)
    model: str = "claude-opus-4-8"
    max_tokens: int = 2048
    answer: str | None = None
    error: str | None = None
    usage: dict[str, int] | None = None
    cost_usd: float | None = None

    def estimate(self) -> dict[str, Any]:
        """Rough pre-approval cost estimate (chars/4 heuristic for input,
        worst case = the full max_tokens for output)."""
        in_price, out_price = MODEL_PRICES.get(self.model, (5.00, 25.00))
        est_input_tokens = (len(self.question) + len(self.context) + len(SYSTEM_PROMPT)) // 4 + 50
        est_input_cost = est_input_tokens * in_price / 1_000_000
        max_output_cost = self.max_tokens * out_price / 1_000_000
        return {
            "input_tokens": est_input_tokens,
            "max_cost_usd": round(est_input_cost + max_output_cost, 4),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "context": self.context,
            "status": self.status,
            "created_at": self.created_at,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "answer": self.answer,
            "error": self.error,
            "usage": self.usage,
            "cost_usd": self.cost_usd,
            "estimate": self.estimate(),
        }


class EscalationManager:
    """In-memory queue of escalation requests (lives for the server process)."""

    MAX_KEPT = 50

    def __init__(self):
        self.requests: dict[str, EscalationRequest] = {}
        self._lock = asyncio.Lock()

    @staticmethod
    def api_key_configured() -> bool:
        return bool(os.getenv("ANTHROPIC_API_KEY"))

    def create(
        self,
        question: str,
        context: str = "",
        model: str = "claude-opus-4-8",
        max_tokens: int = 2048,
    ) -> EscalationRequest:
        request = EscalationRequest(
            id=uuid.uuid4().hex[:12],
            question=question.strip(),
            context=context.strip(),
            model=model,
            max_tokens=max_tokens,
        )
        self.requests[request.id] = request
        self._prune()
        logger.info(f"Escalation queued: {request.id} ({len(request.question)} chars)")
        return request

    def get(self, request_id: str) -> EscalationRequest | None:
        return self.requests.get(request_id)

    def list(self) -> list[dict[str, Any]]:
        ordered = sorted(self.requests.values(), key=lambda r: r.created_at, reverse=True)
        return [r.to_dict() for r in ordered]

    def pending(self) -> builtins.list[dict[str, Any]]:
        return [r.to_dict() for r in self.requests.values() if r.status == "pending"]

    def deny(self, request_id: str) -> bool:
        request = self.requests.get(request_id)
        if not request or request.status != "pending":
            return False
        request.status = "denied"
        logger.info(f"Escalation denied by user: {request_id}")
        return True

    async def approve(self, request_id: str) -> EscalationRequest:
        """Call the Anthropic API for an approved request. Raises on bad state."""
        async with self._lock:
            request = self.requests.get(request_id)
            if not request:
                raise KeyError(f"unknown escalation: {request_id}")
            if request.status != "pending":
                raise ValueError(f"escalation {request_id} is {request.status}, not pending")
            request.status = "approved"

        try:
            answer, usage = await self._call_claude(request)
            request.answer = answer
            request.usage = usage
            in_price, out_price = MODEL_PRICES.get(request.model, (5.00, 25.00))
            request.cost_usd = round(
                usage.get("input_tokens", 0) * in_price / 1_000_000
                + usage.get("output_tokens", 0) * out_price / 1_000_000,
                4,
            )
            request.status = "answered"
            logger.info(f"Escalation answered: {request_id} (${request.cost_usd})")
        except Exception as e:
            request.status = "failed"
            request.error = str(e)
            logger.error(f"Escalation failed: {request_id}: {e}")
        return request

    async def _call_claude(self, request: EscalationRequest):
        from anthropic import AsyncAnthropic  # imported lazily; needs ANTHROPIC_API_KEY

        if not self.api_key_configured():
            raise RuntimeError("ANTHROPIC_API_KEY is not set - add it to .env")

        user_content = request.question
        if request.context:
            user_content = f"Context from the local session:\n{request.context}\n\nQuestion:\n{request.question}"

        kwargs: dict[str, Any] = dict(
            model=request.model,
            max_tokens=request.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
        # Fable 5 has thinking always on and rejects the parameter; the other
        # models get adaptive thinking (recommended for anything non-trivial).
        if not request.model.startswith("claude-fable"):
            kwargs["thinking"] = {"type": "adaptive"}

        async with AsyncAnthropic() as client:
            response = await client.messages.create(**kwargs)

        if response.stop_reason == "refusal":
            raise RuntimeError("Claude declined this request (safety refusal)")

        text = "\n".join(b.text for b in response.content if b.type == "text").strip()
        if not text:
            raise RuntimeError(f"Claude returned no text (stop_reason={response.stop_reason})")
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
        return text, usage

    def _prune(self) -> None:
        if len(self.requests) <= self.MAX_KEPT:
            return
        finished = sorted(
            (r for r in self.requests.values() if r.status != "pending"),
            key=lambda r: r.created_at,
        )
        for stale in finished[: len(self.requests) - self.MAX_KEPT]:
            del self.requests[stale.id]


# Global manager — the ask_claude tool and the web server share this queue.
_manager: EscalationManager | None = None


def get_escalation_manager() -> EscalationManager:
    global _manager
    if _manager is None:
        _manager = EscalationManager()
    return _manager
