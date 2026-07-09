"""Persist conversation facts before the history window drops older turns."""

from __future__ import annotations

import logging
from typing import Any

from core.llm_interface import BaseLLMInterface
from core.memory_manager import MemoryManager
from core.schemas import ConversationHistory

logger = logging.getLogger(__name__)

FLUSH_SEGMENT_TYPE = "CONVERSATION_FLUSH"
_FLUSH_BUFFER = 2  # wait until window + buffer messages before flushing


async def _last_flushed_through(memory_manager: MemoryManager, session_id: str) -> int:
    segments = await memory_manager.get_recent_memory(
        limit=20,
        filter_dict={"type": FLUSH_SEGMENT_TYPE, "session_id": session_id},
    )
    if not segments:
        return 0
    return max(int(seg.metadata.get("flushed_through", 0) or 0) for seg in segments)


def _format_batch(messages: list[Any]) -> str:
    lines: list[str] = []
    for msg in messages:
        role = getattr(msg, "role", "user")
        content = (getattr(msg, "content", None) or "").strip()
        if not content:
            continue
        lines.append(f"{role.upper()}: {content}")
    return "\n".join(lines)


async def maybe_flush_conversation_memory(
    conversation: ConversationHistory | None,
    *,
    history_window: int,
    memory_manager: MemoryManager | None,
    llm_interface: BaseLLMInterface | None,
    session_id: str,
    skip_global_store: bool = False,
) -> bool:
    """Summarize and store conversation turns that are about to leave the window.

    Returns True when a new flush segment was written.
    """
    if (
        skip_global_store
        or conversation is None
        or not conversation.messages
        or memory_manager is None
        or llm_interface is None
    ):
        return False

    total = len(conversation.messages)
    keep = max(2, int(history_window))
    if total <= keep + _FLUSH_BUFFER:
        return False

    flush_through = total - keep
    last_flushed = await _last_flushed_through(memory_manager, session_id)
    if flush_through <= last_flushed:
        return False

    batch = conversation.messages[last_flushed:flush_through]
    transcript = _format_batch(batch)
    if not transcript.strip():
        return False

    prompt = (
        "You are preserving durable facts from a chat session before older turns "
        "leave the active context window.\n"
        "Write a short summary (max 8 bullet points) capturing:\n"
        "- explicit user preferences or decisions\n"
        "- names, dates, filenames, or tasks still in progress\n"
        "- anything the assistant promised to do\n"
        "Skip greetings and filler. Use plain text bullets.\n\n"
        f"SESSION EXCERPT:\n{transcript}"
    )

    try:
        summary = await llm_interface.generate_text(prompt=prompt, max_tokens=512, temperature=0.2)
    except Exception as exc:
        logger.warning("Conversation flush LLM call failed for %s: %s", session_id, exc)
        return False

    summary = (summary or "").strip()
    if not summary:
        return False

    await memory_manager.add_memory(
        type=FLUSH_SEGMENT_TYPE,
        source="conversation_compaction",
        content_text=summary,
        importance=0.85,
        metadata={
            "session_id": session_id,
            "flushed_through": flush_through,
            "message_count": total,
            "history_window": keep,
        },
    )
    logger.info(
        "Flushed conversation memory for session %s through message %s",
        session_id,
        flush_through,
    )
    return True


async def get_session_flush_context(
    memory_manager: MemoryManager | None,
    session_id: str,
    *,
    limit: int = 2,
) -> str:
    """Return persisted flush summaries for a session (for prompt injection)."""
    if memory_manager is None or not session_id:
        return ""
    segments = await memory_manager.get_recent_memory(
        limit=limit,
        filter_dict={"type": FLUSH_SEGMENT_TYPE, "session_id": session_id},
    )
    if not segments:
        return ""
    segments = sorted(
        segments,
        key=lambda s: int(s.metadata.get("flushed_through", 0) or 0),
    )
    bullets = [seg.content.text.strip() for seg in segments if seg.content.text]
    if not bullets:
        return ""
    return "PERSISTED SESSION MEMORY (from earlier turns):\n" + "\n---\n".join(bullets)
