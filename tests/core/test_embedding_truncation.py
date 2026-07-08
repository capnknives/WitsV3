"""Tests for embedding input truncation."""

from core.config import WitsV3Config
from core.memory_manager import (
    DEFAULT_MAX_EMBEDDING_CHARS,
    resolve_max_embedding_chars,
    truncate_for_embedding,
)


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
