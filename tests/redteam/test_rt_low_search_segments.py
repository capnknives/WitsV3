"""Red team, lower priority: FAISS search_segments and the ``type`` filter.

The native loop asks for owner facts with ``filter_dict={"type": seg_type}``
(agents/native_tool_orchestrator.py, _memory_context). ``type`` is a segment
attribute, not metadata. Safe behaviour: search_segments returns the segment of
that type, as get_recent_segments does. Stub embeddings, files under tmp_path.
"""

from __future__ import annotations

import hashlib
from unittest.mock import MagicMock

import numpy as np
import pytest

pytest.importorskip("faiss")

from core.config import WitsV3Config  # noqa: E402
from core.faiss_memory_backend import FaissCPUMemoryBackend  # noqa: E402
from core.memory_manager import MemorySegment, MemorySegmentContent  # noqa: E402

pytestmark = pytest.mark.asyncio


class _BagOfWordsLLM:
    async def get_embedding(self, text, model=None):
        vec = np.zeros(384)
        for word in text.lower().split():
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % 384] += 1.0
        norm = np.linalg.norm(vec)
        return (vec / norm if norm else vec).tolist()


def _config(tmp_path):
    config = MagicMock(spec=WitsV3Config)
    config.memory_manager = MagicMock()
    config.memory_manager.memory_file_path = str(tmp_path / "memory.json")
    config.memory_manager.faiss_index_path = str(tmp_path / "faiss_index.bin")
    config.memory_manager.vector_dim = 384
    config.memory_manager.max_memory_segments = 1000
    config.memory_manager.pruning_interval_seconds = 3600
    config.ollama_settings = MagicMock()
    config.ollama_settings.embedding_model = "stub"
    return config


async def _backend(tmp_path):
    backend = FaissCPUMemoryBackend(_config(tmp_path), _BagOfWordsLLM())
    await backend.initialize()
    for seg_type, text in (("USER_FACT", "my truck is blue"), ("TRANSCRIPT", "my truck is red")):
        await backend.add_segment(
            MemorySegment(type=seg_type, source="t", content=MemorySegmentContent(text=text))
        )
    return backend


async def test_get_recent_segments_type_filter_control(tmp_path):
    backend = await _backend(tmp_path)
    found = await backend.get_recent_segments(limit=5, filter_dict={"type": "USER_FACT"})
    assert [s.type for s in found] == ["USER_FACT"]


async def test_search_segments_type_filter_returns_matching_segment(tmp_path):
    backend = await _backend(tmp_path)
    found = await backend.search_segments("truck", limit=5, filter_dict={"type": "USER_FACT"})
    assert [s.type for s in found] == ["USER_FACT"]


async def test_search_segments_metadata_filter_still_works(tmp_path):
    backend = await _backend(tmp_path)
    await backend.add_segment(
        MemorySegment(
            type="USER_FACT",
            source="t",
            content=MemorySegmentContent(text="my truck is green"),
            metadata={"session_id": "s1"},
        )
    )
    found = await backend.search_segments("truck", limit=5, filter_dict={"session_id": "s1"})
    assert [s.content.text for s in found] == ["my truck is green"]
