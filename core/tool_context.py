"""Per-call context for tools: who the turn is for and what conversation it is in.

The native loop sets a ``ToolContext`` around every tool call of a turn, so tools get
the session, role and conversation without the model passing them. Tools read the
context first and fall back to their argument only when none is set, which keeps the
old path (arguments injected by ``_prepare_tool_args``) working unchanged.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

# Reason: the model must never choose these; on the native path they come from the context
# and are removed from every schema the model sees.
CONTEXT_PARAMS = frozenset({"user_role", "conversation_history", "session_id"})


@dataclass(frozen=True)
class ToolContext:
    """What a tool may know about the turn it runs in.

    Attributes:
        session_id: Chat session id, or None.
        user_role: Role of the person the turn is for; only "owner" counts as owner.
        conversation_history: The turn's ConversationHistory, or None.
        turn_id: Turn metrics id, or None.
        read_role: For a non-owner turn, the RBAC role whose read roots apply (e.g.
            "family_adult"); None means the plain guest roots.
    """

    session_id: str | None
    user_role: str
    conversation_history: Any | None
    turn_id: str | None
    read_role: str | None = None


# Reason: a ContextVar follows the task, so concurrent turns keep their own context and
# tasks started by asyncio.gather inside one turn inherit it.
_CURRENT: ContextVar[ToolContext | None] = ContextVar("wits_tool_context", default=None)


def current_tool_context() -> ToolContext | None:
    """Return the context set for the running tool call, or None outside one."""
    return _CURRENT.get()


@contextmanager
def use_tool_context(ctx: ToolContext) -> Iterator[None]:
    """Set ``ctx`` for the enclosed block and restore the previous context on exit.

    Args:
        ctx: The context tools should see.

    Yields:
        None.
    """
    token = _CURRENT.set(ctx)
    try:
        yield
    finally:
        _CURRENT.reset(token)


def tool_user_role(argument: Any = "owner") -> Any:
    """Role a tool should act for: the context's if set, else its own argument.

    Args:
        argument: The ``user_role`` the tool was called with (old path).

    Returns:
        With a context, "owner" or "guest"; without one, ``argument`` unchanged.
    """
    ctx = current_tool_context()
    if ctx is None:
        return argument
    # Reason: fail closed; any value other than "owner" (including "" or an unknown role,
    # which some policies would read as owner) gets the guest treatment.
    return "owner" if ctx.user_role == "owner" else "guest"


def tool_read_role(argument: Any = "owner") -> Any:
    """Role whose read roots a file tool uses: the context's if set, else its argument.

    Args:
        argument: The ``user_role`` the tool was called with (no context set).

    Returns:
        With a context: "owner" only for an exact owner turn, else the context's non-owner
        ``read_role`` or "guest". Without one, ``argument`` unchanged.
    """
    ctx = current_tool_context()
    if ctx is None:
        return argument
    if ctx.user_role == "owner":
        return "owner"
    # Reason: RT-10, a family guest keeps its policy read roots but can never read as owner.
    read_role = ctx.read_role
    if isinstance(read_role, str) and read_role.strip() and read_role.strip().lower() != "owner":
        return read_role
    return "guest"


def tool_conversation(argument: Any = None) -> Any:
    """Conversation a tool should read: the context's if set, else its own argument."""
    ctx = current_tool_context()
    return argument if ctx is None else ctx.conversation_history


def strip_context_params(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a normalised schema without the ``CONTEXT_PARAMS`` names.

    Args:
        schema: Output of ``normalise_schema`` (closed, so removed names become unknown).

    Returns:
        The schema with those names gone from ``properties`` and ``required``.
    """
    stripped = dict(schema)
    stripped["properties"] = {
        k: v for k, v in schema.get("properties", {}).items() if k not in CONTEXT_PARAMS
    }
    stripped["required"] = [k for k in schema.get("required", []) if k not in CONTEXT_PARAMS]
    return stripped
