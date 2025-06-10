"""Tests for the FAISS memory backends."""

import os
import pytest
import asyncio
import numpy as np
from typing import AsyncGenerator
from unittest.mock import MagicMock, patch, AsyncMock
from pathlib import Path
import tempfile
import shutil
import faiss

from core.faiss_memory_backend import FaissCPUMemoryBackend, FaissGPUMemoryBackend
from core.memory_manager import MemorySegment, MemorySegmentContent
from core.config import WitsV3Config
from core.llm_interface import BaseLLMInterface

class DummyLLM(BaseLLMInterface):
    def __init__(self):
        # Don't call super() since we're just using this for testing
        pass

    async def generate_text(self, prompt: str, **kwargs) -> str:
        return "dummy response"

    async def stream_text(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        yield "dummy"
        yield " stream"

    async def get_embedding(self, text, model=None):
        # Create consistent but unique embeddings based on text
        # This helps test search functionality
        text_hash = hash(text) % 10000
        embedding = np.zeros(384)
        embedding[0] = text_hash / 10000
        embedding[1] = 0.5  # Add some consistency

        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.tolist()

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Clean up
    shutil.rmtree(temp_dir)

@pytest.fixture
def mock_config(temp_dir):
    """Create a mock configuration with temporary paths."""
    config = MagicMock(spec=WitsV3Config)

    # Create nested mock objects manually
    config.memory_manager = MagicMock()
    config.memory_manager.memory_file_path = os.path.join(temp_dir, "memory.json")
    config.memory_manager.faiss_index_path = os.path.join(temp_dir, "faiss_index.bin")
    config.memory_manager.vector_dim = 384
    config.memory_manager.max_memory_segments = 1000
    config.memory_manager.pruning_interval_seconds = 3600

    config.ollama_settings = MagicMock()
    config.ollama_settings.embedding_model = "test-embedding-model"

    return config

@pytest.mark.asyncio
async def test_faiss_cpu_backend_initialize(mock_config):
    """Test initializing the FAISS CPU backend."""
    llm = DummyLLM()
    backend = FaissCPUMemoryBackend(mock_config, llm)

    await backend.initialize()

    assert backend.is_initialized
    assert backend.index is not None
    assert backend.index.d == mock_config.memory_manager.vector_dim
    assert isinstance(backend.index, faiss.IndexFlatL2)

@pytest.mark.asyncio
async def test_faiss_add_and_get_segment(mock_config):
    """Test adding and retrieving memory segments."""
    llm = DummyLLM()
    backend = FaissCPUMemoryBackend(mock_config, llm)
    await backend.initialize()

    # Create a test memory segment
    segment = MemorySegment(
        id="test-id",
        type="test",
        source="unit-test",
        content=MemorySegmentContent(text="Test content for search"),
        metadata={"test": True}
    )

    # Test adding the segment
    await backend.add_segment(segment)

    # Test retrieving the segment
    retrieved = await backend.get_segment("test-id")
    assert retrieved is not None
    assert retrieved.id == "test-id"
    assert retrieved.type == "test"
    assert retrieved.content.text == "Test content for search"
    assert retrieved.metadata["test"] is True
    assert retrieved.embedding is not None

@pytest.mark.asyncio
async def test_faiss_search_segments(mock_config):
    """Test searching for memory segments by similarity."""
    llm = DummyLLM()
    backend = FaissCPUMemoryBackend(mock_config, llm)
    await backend.initialize()

    # Add several test segments with different content
    segments = [
        MemorySegment(
            id=f"test-id-{i}",
            type="test",
            source="unit-test",
            content=MemorySegmentContent(text=f"Memory content about {topic}"),
            metadata={"topic": topic}
        )
        for i, topic in enumerate(["python", "javascript", "rust", "python advanced", "typescript"])
    ]

    for segment in segments:
        await backend.add_segment(segment)

    # Search for python-related segments
    results = await backend.search_segments("python programming language", limit=3)

    assert len(results) > 0
    # The top results should be related to python
    python_results = [r for r in results if "python" in r.content.text.lower()]
    assert len(python_results) > 0

    # Test with filters
    filtered_results = await backend.search_segments(
        "programming language",
        limit=3,
        filter_dict={"topic": "javascript"}
    )

    assert len(filtered_results) > 0
    assert all(r.metadata.get("topic") == "javascript" for r in filtered_results)

@pytest.mark.asyncio
async def test_faiss_get_recent_segments(mock_config):
    """Test retrieving recent memory segments."""
    llm = DummyLLM()
    backend = FaissCPUMemoryBackend(mock_config, llm)
    await backend.initialize()

    # Add several test segments
    for i in range(10):
        segment = MemorySegment(
            id=f"test-id-{i}",
            type="test",
            source="unit-test",
            content=MemorySegmentContent(text=f"Memory content {i}"),
            metadata={"index": i}
        )
        await backend.add_segment(segment)

    # Get recent segments
    recent = await backend.get_recent_segments(limit=5)

    assert len(recent) == 5
    # Should be sorted by timestamp (most recent first)
    # Since we added them in sequence, the highest indices should be most recent
    indices = [s.metadata["index"] for s in recent]
    assert indices == sorted(indices, reverse=True)

    # Test with filters
    filtered_recent = await backend.get_recent_segments(
        limit=3,
        filter_dict={"index": 5}
    )

    assert len(filtered_recent) == 1
    assert filtered_recent[0].metadata["index"] == 5

@pytest.mark.asyncio
async def test_faiss_prune_memory(mock_config):
    """Test pruning memory segments."""
    llm = DummyLLM()

    # Set a very low max_memory_segments to trigger pruning
    mock_config.memory_manager.max_memory_segments = 5

    backend = FaissCPUMemoryBackend(mock_config, llm)
    await backend.initialize()

    # Add more segments than the limit
    for i in range(10):
        segment = MemorySegment(
            id=f"test-id-{i}",
            type="test",
            source="unit-test",
            content=MemorySegmentContent(text=f"Memory content {i}"),
            # Set varying importance to test pruning logic
            importance=i/10.0
        )
        await backend.add_segment(segment)

    # Manually trigger pruning
    await backend.prune_memory()

    # Should have been pruned to max_memory_segments
    assert len(backend.segments) == mock_config.memory_manager.max_memory_segments

    # The highest importance segments should be kept
    importances = [s.importance for s in backend.segments]
    assert all(imp >= 0.5 for imp in importances)

@pytest.mark.asyncio
async def test_faiss_persistence(mock_config):
    """Test that memory segments and FAISS index are persisted."""
    llm = DummyLLM()

    # First backend instance
    backend1 = FaissCPUMemoryBackend(mock_config, llm)
    await backend1.initialize()

    # Add segments
    for i in range(5):
        segment = MemorySegment(
            id=f"test-id-{i}",
            type="test",
            source="unit-test",
            content=MemorySegmentContent(text=f"Memory content {i}"),
        )
        await backend1.add_segment(segment)

    # Create a second backend instance that should load from the same files
    backend2 = FaissCPUMemoryBackend(mock_config, llm)
    await backend2.initialize()

    # Check that segments were loaded
    assert len(backend2.segments) == 5

    # Try search to verify index is working
    results = await backend2.search_segments("Memory content", limit=3)
    assert len(results) > 0

@pytest.mark.asyncio
async def test_faiss_gpu_backend_fallback(mock_config):
    """Test that GPU backend falls back to CPU if GPU is not available."""
    llm = DummyLLM()

    # Mock that GPU resources are not available
    with patch('faiss.StandardGpuResources', side_effect=Exception("GPU not available")):
        backend = FaissGPUMemoryBackend(mock_config, llm)
        await backend.initialize()

        # Should still initialize with CPU index
        assert backend.is_initialized
        assert backend.index is not None
        assert isinstance(backend.index, faiss.IndexFlatL2)

        # Add and search should still work
        segment = MemorySegment(
            id="test-id",
            type="test",
            source="unit-test",
            content=MemorySegmentContent(text="Test content"),
        )
        await backend.add_segment(segment)

        results = await backend.search_segments("Test", limit=1)
        assert len(results) == 1
