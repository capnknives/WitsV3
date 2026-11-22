"""Helpers for the native tool loop (agents/native_tool_orchestrator.py).

Kept apart so the loop file stays short. Nothing here does I/O.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from core.llm_interface import LLMMessage
from core.prompt_prefix import order_tools
from core.schemas import StreamData
from core.tool_call_guard import SchemaError, normalise_schema

logger = logging.getLogger(__name__)

# Reason: the system message must be byte-identical every turn so Ollama can reuse its
# KV cache; goal, documents, memory and time go into the turn context instead.
NATIVE_SYSTEM_PROMPT = (
    "You are WitsV3, a local assistant. Use the tools when they help, then answer the "
    "user plainly. Call tools only through the structured tool-call format. Anything "
    "under 'Context for this turn:' in the user's message is background for this turn."
)

STEP_LIMIT_ERROR = "Could not complete within the step limit."


def build_tool_specs(
    registry: Any, exclude: frozenset[str]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Build the Ollama tool list and the raw schemas the guard checks against.

    Args:
        registry: Tool registry with a ``tools`` mapping, or None.
        exclude: Tool names the loop never offers.

    Returns:
        (tools payload sorted by name, tool name -> raw schema). A tool whose schema
        cannot be normalised is left out of both, so calling it counts as unknown.
    """
    if not registry:
        return [], {}
    payload: list[dict[str, Any]] = []
    schemas: dict[str, dict[str, Any]] = {}
    for name, tool in registry.tools.items():
        if name in exclude:
            continue
        raw = tool.get_schema() if hasattr(tool, "get_schema") else {}
        try:
            parameters = normalise_schema(raw)
        except SchemaError as exc:
            logger.warning("Native loop skips tool %s: unusable schema (%s)", name, exc)
            continue
        desc = tool.get_llm_description()
        description = desc.get("description", "") if isinstance(desc, dict) else desc
        payload.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": str(description),
                    "parameters": parameters,
                },
            }
        )
        schemas[name] = raw
    return order_tools(payload), schemas


def decode_arguments(arguments: Any) -> dict[str, Any]:
    """Decode arguments the guard has already accepted (so the JSON is valid)."""
    if isinstance(arguments, str):
        return json.loads(arguments) if arguments.strip() else {}
    return dict(arguments or {})


def tool_call_event(source: str, tool_name: str, ok: bool) -> StreamData:
    """One event per executed tool; names and outcome only, never arguments."""
    return StreamData(
        type="tool_call",
        content=f"Called {tool_name}",
        source=source,
        metadata={"tool_name": tool_name, "ok": ok},
    )


def build_turn_context(goal: str, user_message: str, documents: str, memory: str) -> dict[str, str]:
    """Return the per-turn context in a fixed key order (goal, documents, memory).

    Reason: a stable key order keeps the appended block predictable between turns. The
    goal is left out when it only repeats the user message.
    """
    context: dict[str, str] = {}
    if goal.strip() != user_message.strip():
        context["goal"] = goal
    context["documents"] = documents
    if memory.strip():
        context["memory"] = memory
    return context


def assistant_tool_message(content: str, calls: list[dict[str, Any]]) -> LLMMessage:
    """The assistant turn that made ``calls``, in Ollama's shape, to precede the replies.

    Arguments that are not a JSON object (the guard rejects those) are sent as ``{}``,
    because Ollama only accepts an object there; the tool reply explains the refusal.
    """
    shaped = []
    for call in calls:
        try:
            arguments = decode_arguments(call.get("arguments"))
        except (ValueError, TypeError, RecursionError):
            arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        shaped.append({"function": {"name": str(call.get("name", "")), "arguments": arguments}})
    return LLMMessage(role="assistant", content=content or "", tool_calls=shaped)


def _role(item: Any) -> str:
    return item.get("role", "") if isinstance(item, dict) else getattr(item, "role", "")


def history_window(history: list[Any], history_turns: int) -> list[Any]:
    """Trim history in blocks so the kept start stays put between cuts.

    All history is kept until it exceeds ``2 * history_turns`` messages; then the oldest
    are dropped so ``history_turns`` remain, and that cut point holds until the kept part
    exceeds the limit again. The start is a pure function of the length, so consecutive
    turns between cuts send the same first message and Ollama keeps its cached prefix.

    Args:
        history: Earlier messages, oldest first (dicts or LLMMessage).
        history_turns: Messages kept right after a cut.

    Returns:
        The slice of ``history`` to send.
    """
    h = max(1, history_turns)
    excess = len(history) - 2 * h
    if excess <= 0:
        return list(history)
    # Reason: each cut advances the start by h + 1 (from 2h + 1 kept down to h).
    start = -(-excess // (h + 1)) * (h + 1)
    # Reason: never keep tool replies without the assistant message that made the calls.
    while start > 0 and _role(history[start]) == "tool":
        start -= 1
    return list(history[start:])
