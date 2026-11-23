"""Per-turn caller context for the shared WitsControlCenterAgent (RT-4).

One control-center agent serves every chat turn, so the caller's role must not
live on ``self``. ``run_in_turn`` wraps a turn's async generator and binds its
``TurnContext`` around every step only; between steps the variable is restored,
so two turns stepped by hand from the same task never see each other's role.

Outside a bound turn, ``turn_for`` returns an explicit OWNER default (logged at debug).
That is deliberate: every guest-reachable entry (web/server.py /api/chat) goes through
``WitsControlCenterAgent.run()``, which binds a turn, and the orchestrators' ``run()``
bind one from their own ``user_role`` kwarg when called directly (``bind_turn_if_unbound``).
The only unbound callers are owner-local: run.py self-repair, BackgroundAgent jobs,
scripts and tests. Shared agent attributes are never consulted.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import logging
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Iterator
from dataclasses import dataclass
from typing import Any, TypeVar

from core.roles import normalize_role

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class TurnContext:
    user_role: str = "owner"
    guest_profile: dict[str, Any] | None = None
    skip_global_memory_store: bool = False
    guest_personalization_context: str = ""

    @classmethod
    def for_request(cls, **kwargs: Any) -> TurnContext:
        """Build the turn for one request from the caller's kwargs.

        Args:
            **kwargs: Request kwargs; ``user_role`` is the caller's role.

        Returns:
            A turn whose ``user_role`` is exactly ``"owner"`` or ``"guest"``.
        """
        # Reason: RT-10, fail closed. A missing, empty or unknown role is a guest; trusted
        # internal callers (CLI, scheduled jobs, scripts) pass user_role="owner" explicitly.
        role = normalize_role(kwargs.get("user_role"))
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
    """Current turn; outside a turn, a fresh owner default (see module docstring)."""
    turn = _CURRENT_TURN.get()
    if turn is not None:
        return turn
    logger.debug("No bound turn for %s; using owner default", type(agent).__name__)
    return TurnContext()


@contextlib.contextmanager
def bind_turn(turn: TurnContext) -> Iterator[TurnContext]:
    """Bind ``turn`` for a plain (non-generator) await, e.g. scripts calling a helper."""
    token = _CURRENT_TURN.set(turn)
    try:
        yield turn
    finally:
        _CURRENT_TURN.reset(token)


def bind_turn_if_unbound(method: Callable[..., AsyncGenerator[T, None]]):
    """Decorate an agent's async-generator ``run``: inside a turn, pass through; called
    directly, bind a turn built from its own kwargs (so ``user_role="guest"`` holds)."""

    @functools.wraps(method)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> AsyncIterator[T]:
        agen = method(self, *args, **kwargs)
        if _CURRENT_TURN.get() is None:
            agen = run_in_turn(TurnContext.for_request(**kwargs), agen)
        async with contextlib.aclosing(agen) as events:
            async for item in events:
                yield item

    return wrapper


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
