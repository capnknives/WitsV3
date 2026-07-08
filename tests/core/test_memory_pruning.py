"""Regression tests for automatic memory pruning (BasicMemoryBackend).

`add_segment` used to have its `_prune_if_needed()` call commented out
("Temporarily disabled ... to fix syntax errors"), which silently made
`enable_auto_pruning` a no-op for the default `basic` backend regardless of
config. These tests exercise the count- and size-based pruning paths through
the real `add_segment` call to guard against a regression.
"""

import pytest

from core.config import WitsV3Config
from core.memory_manager import BasicMemoryBackend, MemorySegment, MemorySegmentContent


class FakeLLM:
    async def get_embedding(self, text, model=None):
        return [0.1, 0.2, 0.3]


def _make_backend(tmp_path, **memory_overrides):
    config = WitsV3Config()
    config.memory_manager.memory_file_path = str(tmp_path / "memory.json")
    config.memory_manager.enable_auto_pruning = True
    for key, value in memory_overrides.items():
        setattr(config.memory_manager, key, value)
    backend = BasicMemoryBackend(config, FakeLLM())
    return backend


def _segment(text="hello world", importance=0.5):
    return MemorySegment(
        type="USER_INPUT",
        source="user",
        content=MemorySegmentContent(text=text),
        importance=importance,
    )


@pytest.mark.asyncio
async def test_add_segment_prunes_when_count_exceeds_limit(tmp_path):
    backend = _make_backend(tmp_path, max_memory_segments=3, max_memory_size_mb=10_000)
    await backend.initialize()

    for i in range(5):
        await backend.add_segment(_segment(text=f"segment {i}"))

    assert len(backend.segments) == 3


@pytest.mark.asyncio
async def test_add_segment_prunes_when_size_threshold_exceeded(tmp_path):
    # BasicMemoryBackend._get_memory_size_mb() reads the real on-disk JSON
    # file (~0.0003 MB/segment for these tiny fixtures), not the base
    # class's rough estimate — so the threshold has to be sized to that,
    # not to segment count. 0.8 * 0.006 MB is crossed well before 30 adds,
    # with the count limit set high enough to isolate the size-based path.
    backend = _make_backend(
        tmp_path, max_memory_size_mb=0.006, pruning_threshold=0.8, max_memory_segments=10_000
    )
    await backend.initialize()

    for i in range(30):
        await backend.add_segment(_segment(text=f"segment {i}"))

    size_mb = await backend._get_memory_size_mb()
    assert size_mb <= backend.settings.max_memory_size_mb
    assert len(backend.segments) < 30
    assert len(backend.segments) >= 10  # prune_memory keeps a floor of 10


@pytest.mark.asyncio
async def test_add_segment_does_not_prune_when_disabled(tmp_path):
    backend = _make_backend(
        tmp_path, enable_auto_pruning=False, max_memory_segments=3, max_memory_size_mb=10_000
    )
    await backend.initialize()

    for i in range(5):
        await backend.add_segment(_segment(text=f"segment {i}"))

    assert len(backend.segments) == 5


@pytest.mark.asyncio
async def test_add_segment_keeps_highest_importance_under_hybrid_strategy(tmp_path):
    backend = _make_backend(
        tmp_path,
        max_memory_segments=2,
        max_memory_size_mb=10_000,
        pruning_strategy="least_relevant",
    )
    await backend.initialize()

    await backend.add_segment(_segment(text="low", importance=0.1))
    await backend.add_segment(_segment(text="high", importance=0.9))
    await backend.add_segment(_segment(text="medium", importance=0.5))

    assert len(backend.segments) == 2
    kept_texts = {s.content.text for s in backend.segments}
    assert kept_texts == {"high", "medium"}
