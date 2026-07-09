"""Tests for embedding input truncation."""

import pytest

from core.config import WitsV3Config
from core.memory_manager import (
    DEFAULT_MAX_EMBEDDING_CHARS,
    SKIP_EMBEDDING_SEGMENT_TYPES,
    MemorySegment,
    MemorySegmentContent,
    prepare_text_for_embedding,
    resolve_max_embedding_chars,
    truncate_for_embedding,
)
from unittest.mock import AsyncMock, MagicMock


def test_truncate_for_embedding_within_limit_unchanged():
    text = "short text"
    assert truncate_for_embedding(text, 100) == text


def test_truncate_for_embedding_respects_max_including_suffix():
    text = "x" * 100
    suffix = "…"
    result = truncate_for_embedding(text, 50, suffix=suffix)
    assert len(result) == 50
    assert result.endswith(suffix)


def test_truncate_for_embedding_long_suffix_in_base_agent():
    text = "y" * 7000
    suffix = "\n… [truncated for memory]"
    max_chars = 6000
    result = truncate_for_embedding(text, max_chars, suffix=suffix)
    assert len(result) == max_chars
    assert result.endswith(suffix)


def test_resolve_max_embedding_chars_none_config():
    assert resolve_max_embedding_chars(None) == DEFAULT_MAX_EMBEDDING_CHARS


def test_resolve_max_embedding_chars_missing_memory_manager():
    class FakeConfig:
        memory_manager = None

    assert resolve_max_embedding_chars(FakeConfig()) == DEFAULT_MAX_EMBEDDING_CHARS


def test_resolve_max_embedding_chars_from_config():
    config = WitsV3Config()
    config.memory_manager.max_embedding_chars = 4096
    assert resolve_max_embedding_chars(config) == 4096


def test_prepare_text_for_embedding_caps_length():
    text = "z" * 10000
    prepared = prepare_text_for_embedding(text, WitsV3Config())
    assert len(prepared) <= DEFAULT_MAX_EMBEDDING_CHARS


def test_skip_embedding_segment_types_include_intent_analysis():
    assert "INTENT_ANALYSIS" in SKIP_EMBEDDING_SEGMENT_TYPES


@pytest.mark.asyncio
async def test_oversized_tool_response_skips_embedding(tmp_path, monkeypatch):
    """TOOL_RESPONSE segments over 2x cap should not call embed API."""
    config = WitsV3Config()
    config.memory_manager.max_embedding_chars = 100
    config.memory_manager.memory_file_path = str(tmp_path / "mem.json")
    llm = MagicMock()
    llm.get_embedding = AsyncMock(return_value=[0.1])

    from core.memory_manager import BasicMemoryBackend

    backend = BasicMemoryBackend(config, llm)
    segment = MemorySegment(
        type="TOOL_RESPONSE",
        source="test",
        content=MemorySegmentContent(tool_output="x" * 500),
        importance=0.5,
    )
    await backend._generate_embedding_if_needed(segment)
    llm.get_embedding.assert_not_called()
    assert segment.embedding is None
