"""Per-turn caller context for the shared WitsControlCenterAgent (RT-4).

One control-center agent serves every chat turn, so the caller's role must not
live on ``self``. ``run_in_turn`` wraps a turn's async generator and binds its
``TurnContext`` around every step only; between steps the variable is restored,
so two turns stepped by hand from the same task never see each other's role.

Outside a bound turn, ``turn_for`` returns an explicit OWNER default (logged at debug).
That is deliberate: every guest-reachable entry (web/server.py /api/chat) goes through
``WitsControlCenterAgent.run()``, which binds a turn, and the orchestrators' ``run()``
bind one from their own ``user_role`` kwarg (``bind_turn_if_unbound``; no role is a
guest). The only unbound callers are owner-local: run.py self-repair, BackgroundAgent
jobs, scripts and tests. Shared agent attributes are never consulted.

RT-10: a guest from any source wins. ``call_turn`` gives an explicit ``user_role`` kwarg
and the bound turn a veto each, so a nested call cannot lift a guest turn to owner and an
explicit ``user_role="guest"`` inside an owner turn stays a guest.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import logging
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Iterator
from dataclasses import dataclass, replace
from typing import Any, TypeVar

from core.roles import GUEST, is_owner, normalize_role
from core.tool_context import ToolContext

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


def call_turn(kwargs: dict[str, Any]) -> TurnContext:
    """The turn a call should run in, given its kwargs and the bound turn (RT-10).

    Args:
        kwargs: The call's kwargs; ``user_role``, when present, is the caller's role.

    Returns:
        The bound turn when every source is owner (or it is already a guest turn); a turn
        built from ``kwargs`` when none is bound (no role there is a guest); otherwise a
        guest copy of the bound turn.
    """
    bound = _CURRENT_TURN.get()
    if bound is None:
        return TurnContext.for_request(**kwargs)
    explicit_guest = "user_role" in kwargs and not is_owner(kwargs["user_role"])
    if is_owner(bound.user_role) and not explicit_guest:
        return bound
    if bound.user_role == GUEST and bound.skip_global_memory_store:
        return bound
    # Reason: a guest from either source wins; the copy keeps the turn's other fields. Only
    # an owner turn may pick the guest's profile; a non-owner turn keeps its own, so a
    # nested call can never name a wider policy than its caller's.
    profile = bound.guest_profile
    if is_owner(bound.user_role):
        profile = kwargs.get("guest_profile") or profile
    return replace(bound, user_role=GUEST, skip_global_memory_store=True, guest_profile=profile)


def guest_tool_context(
    turn: TurnContext,
    *,
    session_id: str | None = None,
    conversation_history: Any | None = None,
    turn_id: str | None = None,
) -> ToolContext | None:
    """Guest ToolContext for a tool call in ``turn``, or None for an exact owner turn.

    Args:
        turn: The turn the tool call runs in.
        session_id: Chat session id, or None.
        conversation_history: The turn's ConversationHistory, or None.
        turn_id: Turn metrics id, or None.

    Returns:
        None for an owner turn; otherwise a guest context whose ``read_role`` is the
        turn's RBAC role (its policy read roots), never owner.
    """
    if is_owner(turn.user_role):
        return None
    from core.guest_access import resolve_effective_role

    return ToolContext(
        session_id=session_id,
        user_role=GUEST,
        conversation_history=conversation_history,
        turn_id=turn_id,
        read_role=resolve_effective_role(GUEST, turn.guest_profile),
    )


def request_turn(kwargs: dict[str, Any]) -> TurnContext:
    """A fresh turn from ``kwargs`` that can never be more than the bound turn (RT-10).

    Args:
        kwargs: The request kwargs; ``user_role`` is the caller's role.

    Returns:
        ``TurnContext.for_request(**kwargs)``, as a guest when a non-owner turn is bound.
    """
    turn = TurnContext.for_request(**kwargs)
    bound = _CURRENT_TURN.get()
    if bound is not None and not is_owner(bound.user_role) and is_owner(turn.user_role):
        return TurnContext.for_request(**{**kwargs, "user_role": GUEST})
    return turn


@contextlib.contextmanager
def bind_turn(turn: TurnContext) -> Iterator[TurnContext]:
    """Bind ``turn`` for a plain (non-generator) await, e.g. scripts calling a helper."""
    token = _CURRENT_TURN.set(turn)
    try:
        yield turn
    finally:
        _CURRENT_TURN.reset(token)


def bind_turn_if_unbound(method: Callable[..., AsyncGenerator[T, None]]):
    """Decorate an agent's async-generator ``run`` so it always runs in ``call_turn``:
    the bound turn when it and the kwargs agree on owner, else a guest turn."""

    @functools.wraps(method)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> AsyncIterator[T]:
        agen = method(self, *args, **kwargs)
        turn = call_turn(kwargs)
        if turn is not _CURRENT_TURN.get():
            agen = run_in_turn(turn, agen)
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
