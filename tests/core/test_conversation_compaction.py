"""Tests for pre-compaction conversation memory flush."""

from unittest.mock import AsyncMock

import pytest

from core.conversation_compaction import (
    FLUSH_SEGMENT_TYPE,
    get_session_flush_context,
    maybe_flush_conversation_memory,
)
from core.schemas import ConversationHistory


class DummyLLM:
    async def generate_text(self, prompt: str, **kwargs) -> str:
        return "- User wants audit summary\n- File: report.md"


@pytest.mark.asyncio
async def test_flush_skipped_when_under_window():
    history = ConversationHistory(session_id="sess-1")
    for i in range(5):
        history.add_message("user", f"message {i}")
        history.add_message("assistant", f"reply {i}")

    memory = AsyncMock()
    flushed = await maybe_flush_conversation_memory(
        history,
        history_window=20,
        memory_manager=memory,
        llm_interface=DummyLLM(),
        session_id="sess-1",
    )
    assert flushed is False
    memory.add_memory.assert_not_called()


@pytest.mark.asyncio
async def test_flush_persists_dropped_turns():
    history = ConversationHistory(session_id="sess-2")
    for i in range(30):
        history.add_message("user", f"message {i}")
        history.add_message("assistant", f"reply {i}")

    memory = AsyncMock()
    memory.get_recent_memory = AsyncMock(return_value=[])
    memory.add_memory = AsyncMock(return_value="flush-1")

    flushed = await maybe_flush_conversation_memory(
        history,
        history_window=10,
        memory_manager=memory,
        llm_interface=DummyLLM(),
        session_id="sess-2",
    )
    assert flushed is True
    memory.add_memory.assert_called_once()
    kwargs = memory.add_memory.call_args.kwargs
    assert kwargs["type"] == FLUSH_SEGMENT_TYPE
    assert kwargs["metadata"]["session_id"] == "sess-2"
    assert kwargs["metadata"]["flushed_through"] == 50


@pytest.mark.asyncio
async def test_flush_skipped_for_guest_global_store():
    history = ConversationHistory(session_id="sess-guest")
    for i in range(30):
        history.add_message("user", f"message {i}")

    memory = AsyncMock()
    flushed = await maybe_flush_conversation_memory(
        history,
        history_window=10,
        memory_manager=memory,
        llm_interface=DummyLLM(),
        session_id="sess-guest",
        skip_global_store=True,
    )
    assert flushed is False


@pytest.mark.asyncio
async def test_get_session_flush_context_formats_segments():
    class Seg:
        def __init__(self, text, flushed_through):
            self.content = type("C", (), {"text": text})()
            self.metadata = {"flushed_through": flushed_through}

    memory = AsyncMock()
    memory.get_recent_memory = AsyncMock(
        return_value=[Seg("Earlier user chose audit.md", 12)]
    )
    ctx = await get_session_flush_context(memory, "sess-3")
    assert "PERSISTED SESSION MEMORY" in ctx
    assert "audit.md" in ctx
