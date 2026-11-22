"""Stable prompt prefix helpers.

Ollama reuses its KV cache only while the start of the prompt is byte-identical, so
everything that changes per turn must sit in the last message. These functions are
pure (no I/O) and define the fixed message order:

1. One system message: static system prompt, then the skill catalog.
2. Earlier turns, oldest first, unchanged.
3. The new user message, with changing text appended under a heading.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from .llm_interface import LLMMessage

CONTEXT_HEADING = "Context for this turn:"


def _system_text(system_prompt: str, skill_catalog: str) -> str:
    """Join the static system prompt and the skill catalog."""
    catalog = skill_catalog.strip()
    if not catalog:
        return system_prompt
    return f"{system_prompt}\n\n{catalog}"


def _as_message(item: LLMMessage | Mapping[str, str]) -> LLMMessage:
    """Return a history entry as an LLMMessage without altering its text."""
    if isinstance(item, LLMMessage):
        return item
    return LLMMessage(
        role=item["role"],
        content=item["content"],
        name=item.get("name"),
        tool_calls=item.get("tool_calls"),
    )


def _user_text(user_message: str, turn_context: Mapping[str, str] | None) -> str:
    """Append per-turn context to the user text, or return the text unchanged."""
    if not turn_context:
        return user_message
    # Reason: changing text goes last so it never disturbs the cached prefix.
    lines = [f"{key}: {value}" for key, value in turn_context.items()]
    return f"{user_message}\n\n{CONTEXT_HEADING}\n" + "\n".join(lines)


def build_messages(
    system_prompt: str,
    skill_catalog: str,
    history: Sequence[LLMMessage | Mapping[str, str]],
    user_message: str,
    turn_context: dict[str, str] | None,
) -> list[LLMMessage]:
    """Build the message list for one turn in the fixed order.

    Args:
        system_prompt: Static system prompt; must not hold dates, goals or memory.
        skill_catalog: Static skill catalog text, placed after the system prompt.
        history: Earlier turns, oldest first, passed through unchanged.
        user_message: The new user text.
        turn_context: Changing text (time, memory, documents) or None.

    Returns:
        System message, history, then the new user message.
    """
    messages = [LLMMessage(role="system", content=_system_text(system_prompt, skill_catalog))]
    messages.extend(_as_message(item) for item in history)
    messages.append(LLMMessage(role="user", content=_user_text(user_message, turn_context)))
    return messages


def _tool_name(tool: Mapping[str, Any]) -> str:
    """Return a tool's function name from either schema shape."""
    function = tool.get("function")
    if isinstance(function, Mapping):
        return str(function.get("name", ""))
    return str(tool.get("name", ""))


def order_tools(tools: list[dict]) -> list[dict]:
    """Sort tool definitions by function name so the tool list is order-stable."""
    return sorted(tools, key=_tool_name)


def prefix_fingerprint(messages: Sequence[LLMMessage], tools: list[dict]) -> str:
    """Hash everything before the last message, plus the ordered tools.

    Args:
        messages: Full message list for a turn.
        tools: Tool definitions (ordered here, so input order does not matter).

    Returns:
        Hex sha256 of the prefix.
    """
    payload = {
        # Reason: tool_calls are hashed only when set, so older fingerprints are unchanged.
        "messages": [
            [m.role, m.content, m.name] + ([m.tool_calls] if m.tool_calls else [])
            for m in messages[:-1]
        ],
        "tools": order_tools(tools),
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
