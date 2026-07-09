"""Embedding text preparation helpers (extracted from memory_manager)."""

from __future__ import annotations

from typing import Any

DEFAULT_MAX_EMBEDDING_CHARS = 6000

SKIP_EMBEDDING_SEGMENT_TYPES = frozenset(
    {
        "INTENT_ANALYSIS",
        "REASONING",
        "AGENT_THOUGHT",
        "CREATOR_RECOGNITION",
    }
)


def truncate_for_embedding(text: str, max_chars: int, suffix: str = "…") -> str:
    """Truncate *text* so the returned string length is at most *max_chars*."""
    if len(text) <= max_chars:
        return text
    if len(suffix) >= max_chars:
        return suffix[:max_chars]
    return text[: max_chars - len(suffix)] + suffix


def resolve_max_embedding_chars(config_or_settings: Any) -> int:
    """Return max embedding input length from config or memory settings."""
    if config_or_settings is None:
        return DEFAULT_MAX_EMBEDDING_CHARS
    mm = getattr(config_or_settings, "memory_manager", config_or_settings)
    if mm is None:
        return DEFAULT_MAX_EMBEDDING_CHARS
    value = getattr(mm, "max_embedding_chars", DEFAULT_MAX_EMBEDDING_CHARS)
    if not isinstance(value, int) or value <= 0:
        return DEFAULT_MAX_EMBEDDING_CHARS
    return value


def prepare_text_for_embedding(text: str, config_or_settings: Any) -> str:
    """Truncate text to a safe length before calling the embedding API."""
    max_chars = resolve_max_embedding_chars(config_or_settings)
    return truncate_for_embedding(text, max_chars)
