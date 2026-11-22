"""Per-turn caller context for the shared WitsControlCenterAgent (RT-4).

One control-center agent serves every chat turn, so the caller's role must not
live on ``self``. ``run_in_turn`` wraps a turn's async generator and binds its
``TurnContext`` around every step only; between steps the variable is restored,
so two turns stepped by hand from the same task never see each other's role.
"""

from __future__ import annotations

import contextvars
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass
class TurnContext:
    user_role: str = "owner"
    guest_profile: dict[str, Any] | None = None
    skip_global_memory_store: bool = False
    guest_personalization_context: str = ""

    @classmethod
    def for_request(cls, **kwargs: Any) -> TurnContext:
        role = kwargs.get("user_role", "owner")
        return cls(
            user_role=role,
            guest_profile=kwargs.get("guest_profile"),
            skip_global_memory_store=role == "guest",
            guest_personalization_context=kwargs.get("guest_personalization_context", "") or "",
        )


_CURRENT_TURN: contextvars.ContextVar[TurnContext | None] = contextvars.ContextVar(
    "wits_current_turn", default=None
)


def current_turn() -> TurnContext | None:
    """The turn being stepped right now, or None outside any turn."""
    return _CURRENT_TURN.get()


def turn_for(agent: Any) -> TurnContext:
    """Current turn; outside a turn, today's defaults (agent attributes, else owner)."""
    turn = _CURRENT_TURN.get()
    if turn is not None:
        return turn
    return TurnContext(
        user_role=getattr(agent, "_request_user_role", "owner"),
        guest_profile=getattr(agent, "_request_guest_profile", None),
        skip_global_memory_store=getattr(agent, "_skip_global_memory_store", False),
        guest_personalization_context=getattr(agent, "_guest_personalization_context", ""),
    )


async def run_in_turn(turn: TurnContext, agen: AsyncGenerator[T, None]) -> AsyncIterator[T]:
    """Yield from ``agen`` with ``turn`` bound only while ``agen`` is executing."""
    try:
        while True:
            token = _CURRENT_TURN.set(turn)
            try:
                item = await agen.__anext__()
            except StopAsyncIteration:
                return
            finally:
                _CURRENT_TURN.reset(token)
            yield item
    finally:
        token = _CURRENT_TURN.set(turn)
        try:
            await agen.aclose()
        finally:
            _CURRENT_TURN.reset(token)
