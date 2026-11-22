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
MODEL_CALL_FAILED = "The model call failed"


def resident_model(config: Any) -> str:
    """The one model the native loop uses for every round of every turn.

    Args:
        config: WitsV3Config.

    Returns:
        ``conversation.model`` if set, else ``ollama_settings.orchestrator_model`` if set,
        else ``ollama_settings.default_model``. The model router is never consulted.
    """
    ollama = config.ollama_settings
    return (
        getattr(config.conversation, "model", None)
        or getattr(ollama, "orchestrator_model", None)
        or ollama.default_model
    )


def model_call_error(exc: Exception, model: str | None) -> str:
    """Event text for a failed model call; a missing model is named in plain words."""
    text = str(exc)
    if "404" in text or "not found" in text.lower():
        return (
            f"The model '{model}' was requested but is not installed on the Ollama server. "
            f"Install it (ollama pull {model}) or set conversation.model to an installed one."
        )
    return f"{MODEL_CALL_FAILED}: {exc}"


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


def decode_arguments(arguments: Any) -> Any:
    """Decode a JSON argument string; any other value is returned unchanged.

    Only an empty string becomes ``{}``. Non-object values (null, lists, numbers) come
    back as they are so the guard, not this helper, decides about them.

    Raises:
        ValueError: The string is not valid JSON.
    """
    if isinstance(arguments, str):
        return json.loads(arguments) if arguments.strip() else {}
    return arguments


def tool_reply(call: dict[str, Any], text: str) -> LLMMessage:
    """The role "tool" reply to one call of the preceding assistant message."""
    name = call.get("name")
    return LLMMessage(role="tool", content=text, name=name if isinstance(name, str) else None)


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

    Arguments that are not a JSON object are sent as ``{}`` here only, because Ollama
    accepts nothing else in this field; the guard saw the raw value and the tool reply
    carries its verdict.
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

    All history is kept until it exceeds ``2 * h`` messages, where ``h`` is
    ``history_turns`` but at least 2 (one full user/assistant exchange). Then the
    oldest are dropped so about ``h`` remain, and that cut point holds until the kept
    part exceeds the limit again. The start depends only on the (append-only) history,
    so consecutive turns between cuts send the same first message and Ollama keeps its
    cached prefix.

    Args:
        history: Earlier messages, oldest first (dicts or LLMMessage).
        history_turns: Messages to keep right after a cut.

    Returns:
        The slice of ``history`` to send; after a cut it starts on a user message.
    """
    h = max(2, history_turns)
    excess = len(history) - 2 * h
    if excess <= 0:
        return list(history)
    # Reason: each cut advances the start by h + 1 (from 2h + 1 kept down to h).
    cut = -(-excess // (h + 1)) * (h + 1)
    # Reason: start on a user message; moving forward to one never lands inside a tool
    # group (assistant tool_calls plus its replies), since groups hold no user message.
    users = [i for i, item in enumerate(history) if _role(item) == "user"]
    later = [i for i in users if i >= cut]
    earlier = [i for i in users if i < cut]
    if later and len(history) - later[0] >= 2:
        return list(history[later[0] :])
    if earlier:
        # Reason: keep at least one full exchange rather than a lone trailing message.
        return list(history[earlier[-1] :])
    return list(history[cut:])
