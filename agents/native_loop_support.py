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
from core.tool_context import strip_context_params

logger = logging.getLogger(__name__)

# Reason: the system message must be byte-identical every turn so Ollama can reuse its
# KV cache; goal, documents, memory and time go into the turn context instead.
NATIVE_SYSTEM_PROMPT = (
    "You are WitsV3, a local assistant. Use the tools when they help, then answer the "
    "user plainly. Call tools only through the structured tool-call format. Anything "
    "under 'Context for this turn:' in the user's message is background that may be "
    "irrelevant; answer the newest user message, not the background. "
    "If none of the listed tools fits the request, call find_tools with a few words "
    "describing what is needed before answering that it cannot be done."
)

MEMORY_HEADING = "Things the owner has told you before (background, not a request):"
MEMORY_FACT_CHARS = 300
MEMORY_FETCH_LIMIT = 25
MEMORY_KEEP = 5
# Reason: these are the old path's transcript residue; replaying one makes the model
# answer an old request, so no setting may let them into the turn context.
NEVER_RECALLED_TYPES = frozenset(
    {
        "REASONING",
        "OBSERVATION",
        "GOAL",
        "FINAL_ANSWER",
        "INTENT_ANALYSIS",
        "USER_INPUT",
        "AGENT_THOUGHT",
        "TOOL_CALL",
        "TOOL_RESPONSE",
        "GENERATED_CODE",
    }
)
_TRANSCRIPT_PREFIXES = ("user input:", "goal:", "final answer:")

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
        (tools payload sorted by name, tool name -> schema the guard checks). Both use
        the normalised schema without ``CONTEXT_PARAMS``, so a call that supplies one fails
        as an unknown parameter. A tool whose schema cannot be normalised is left out of
        both, so calling it counts as unknown.
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
            # Reason: role, session and conversation come from the tool context; the
            # model must not be able to set them.
            parameters = strip_context_params(normalise_schema(raw))
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
        schemas[name] = parameters
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


def recalled_facts(
    segments: list[Any], allowed: list[str], min_score: float, keep: int = MEMORY_KEEP
) -> str:
    """The background memory block for the turn context, or "" when nothing qualifies.

    Args:
        segments: Search results, best first.
        allowed: Configured ``conversation.memory_types``; transcript kinds are removed.
        min_score: Drop results whose ``relevance_score`` is lower; 0 turns it off.
        keep: Most facts to include.
    """
    types = {t.upper() for t in allowed} - NEVER_RECALLED_TYPES
    lines: list[str] = []
    for seg in segments:
        if str(getattr(seg, "type", "")).upper() not in types:
            continue
        score = getattr(seg, "relevance_score", None)
        if min_score > 0 and score is not None and score < min_score:
            continue
        text = " ".join((getattr(seg.content, "text", None) or "").split())
        # Reason: a stored fact may begin with the old path's label, which would read
        # as a fresh request.
        if text.lower().startswith(_TRANSCRIPT_PREFIXES):
            text = text.split(":", 1)[1].strip()
        if text:
            lines.append(f"- {text[:MEMORY_FACT_CHARS]}")
        if len(lines) >= keep:
            break
    return f"{MEMORY_HEADING}\n" + "\n".join(lines) if lines else ""


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


def _chars(item: Any) -> int:
    """Characters of one history entry: content plus any tool calls."""
    if isinstance(item, dict):
        content, calls = item.get("content") or "", item.get("tool_calls")
    else:
        content, calls = getattr(item, "content", "") or "", getattr(item, "tool_calls", None)
    return len(content) + (len(json.dumps(calls, default=str)) if calls else 0)


def estimate_tokens(text_or_payload: Any) -> float:
    """Rough token count: characters divided by 4 (JSON for non-strings)."""
    text = text_or_payload if isinstance(text_or_payload, str) else json.dumps(text_or_payload)
    return len(text) / 4


def as_sent_history(conversation: Any, user_input: str) -> tuple[list[dict[str, str]], Any | None]:
    """Earlier messages rebuilt as sent, and the current user message if found.

    The caller adds the current user message before the run, so it is the last message;
    it is found only when that message has role "user" and its content equals
    ``user_input`` (after strip), and is then left out of the history because the loop
    sends it itself.
    Earlier user messages get their stored ``sent_context`` back, so they are
    byte-identical to what the model saw; messages without it pass through unchanged.

    Args:
        conversation: ConversationHistory or None.
        user_input: This turn's user text.

    Returns:
        (history as role/content dicts, the current ConversationMessage or None).
    """
    if conversation is None:
        return [], None
    messages = list(conversation.messages)
    current = None
    # Reason: a matching user message with a reply after it is an older turn that used
    # the same words; writing onto it would change what was already sent.
    if (
        messages
        and messages[-1].role == "user"
        and messages[-1].content.strip() == user_input.strip()
    ):
        current = messages.pop()
    history = []
    for msg in messages:
        block = (msg.metadata or {}).get("sent_context") if msg.role == "user" else None
        content = msg.content + block if isinstance(block, str) else msg.content
        history.append({"role": msg.role, "content": content})
    return history, current


def store_sent_context(message: Any | None, block: str) -> None:
    """Record the context block sent with ``message``; no message or block, no write."""
    if message is None or not block:
        return
    if message.metadata is None:
        message.metadata = {}
    message.metadata["sent_context"] = block


def history_window(
    history: list[Any],
    history_turns: int,
    token_budget: float | None = None,
    base_tokens: float = 0.0,
) -> list[Any]:
    """Trim history in blocks so the kept start stays put between cuts.

    All history is kept until it exceeds ``2 * h`` messages, where ``h`` is
    ``history_turns`` but at least 2 (one full user/assistant exchange). Then the
    oldest are dropped so about ``h`` remain, and that cut point holds until the kept
    part exceeds the limit again. The start depends only on the (append-only) history,
    so consecutive turns between cuts send the same first message and Ollama keeps its
    cached prefix.

    A second trigger cuts back to ``h`` in the same way when the estimated tokens of the
    kept part plus ``base_tokens`` exceed ``token_budget``.

    Args:
        history: Earlier messages, oldest first (dicts or LLMMessage).
        history_turns: Messages to keep right after a cut.
        token_budget: Token limit for kept history plus ``base_tokens``; None disables it.
        base_tokens: Tokens always sent (system prompt, tool payload).

    Returns:
        The slice of ``history`` to send; after a cut it starts on a user message.
    """
    h = max(2, history_turns)
    # Reason: replay the history one message at a time so a cut made on an earlier turn
    # is made again at the same place; the start moves only when a trigger fires anew.
    # With no token budget this gives cuts at multiples of h + 1, as before.
    cut, kept_chars = 0, 0
    sizes = [_chars(item) for item in history]
    for n, size in enumerate(sizes, start=1):
        kept_chars += size
        over_budget = token_budget is not None and base_tokens + kept_chars / 4 > token_budget
        if n - cut > 2 * h or (over_budget and n - h > cut):
            kept_chars -= sum(sizes[cut : n - h])
            cut = n - h
    if cut == 0:
        return list(history)
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
