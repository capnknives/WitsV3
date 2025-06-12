"""
Tests for the memory handler module in the WITS Synthetic Brain.

These tests verify the functionality of the memory handler's integration with
various memory systems and its unified interface for memory operations.
"""

import asyncio
import pytest
import os
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

# Ensure we can import from the root
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the updated module
from core.memory_handler_updated import MemorySegment, MemoryContext, MemoryHandler


@pytest.fixture
def memory_handler():
    """Fixture for a memory handler instance with mocked dependencies."""
    # First patch the dependencies
    with patch('core.memory_handler_updated.MemoryManager', return_value=AsyncMock()) as mock_memory_manager, \
         patch('core.memory_handler_updated.WorkingMemory', return_value=MagicMock()) as mock_working_memory, \
         patch('core.memory_handler_updated.KnowledgeGraph', return_value=MagicMock()) as mock_knowledge_graph:

        # Create temp directory for episodes
        os.makedirs('./temp_episodes', exist_ok=True)

        # Override _load_config method
        MemoryHandler._load_config = MagicMock(return_value={})

        # Configure memory handler with test settings
        handler = MemoryHandler()
        handler.episodic_path = Path('./temp_episodes')

        # Configure mocks
        handler.memory_manager.store = AsyncMock()
        handler.memory_manager.search = AsyncMock(return_value=[
            {"key": "episodic:123", "content": "Test memory", "relevance": 0.9}
        ])

        handler.working_memory.get_snapshot = MagicMock(return_value={"key": "value"})
        handler.knowledge_graph.get_active_concepts = MagicMock(return_value=["test_concept"])

        yield handler

        # Cleanup
        if os.path.exists('./temp_episodes'):
            import shutil
            shutil.rmtree('./temp_episodes')


@pytest.mark.asyncio
async def test_remember_episodic(memory_handler):
    """Test storing an episodic memory."""
    # Store a test memory
    memory_id = await memory_handler.remember(
        content="Test episodic memory",
        memory_type="episodic",
        metadata={"source": "test"}
    )

    # Verify memory manager was called
    memory_handler.memory_manager.store.assert_called_once()

    # Verify the call arguments
    call_args = memory_handler.memory_manager.store.call_args[0]
    assert call_args[0].startswith("episodic:")
    assert call_args[1] == "Test episodic memory"
    assert call_args[2]["source"] == "test"

    # Verify memory ID format
    assert isinstance(memory_id, str)
    assert len(memory_id) > 10  # Should be a UUID


@pytest.mark.asyncio
async def test_recall(memory_handler):
    """Test recalling memories."""
    # Set up mock search results
    memory_handler.memory_manager.search.return_value = [
        {
            "key": "episodic:123",
            "content": "Test memory content",
            "metadata": {"source": "test"},
            "relevance": 0.9
        }
    ]

    # Recall memories
    results = await memory_handler.recall("test query")

    # Verify memory manager search was called
    memory_handler.memory_manager.search.assert_called_once_with("test query", limit=5)

    # Verify results
    assert len(results) == 1
    assert results[0]["key"] == "episodic:123"
    assert results[0]["content"] == "Test memory content"
    assert results[0]["relevance"] == 0.9


@pytest.mark.asyncio
async def test_get_current_context(memory_handler):
    """Test getting the current memory context."""
    # Set up mocks
    memory_handler.working_memory.get_snapshot.return_value = {"focus": "test topic"}
    memory_handler.knowledge_graph.get_active_concepts.return_value = ["concept1", "concept2"]

    # Get context
    context = await memory_handler.get_current_context()

    # Verify context structure
    assert "context_id" in context
    assert "working_memory" in context
    assert context["working_memory"]["focus"] == "test topic"
    assert "active_concepts" in context
    assert "concept1" in context["active_concepts"]
    assert "recent_memories" in context


@pytest.mark.asyncio
async def test_memory_segment_model():
    """Test the MemorySegment model."""
    # Create a memory segment
    segment = MemorySegment(
        id="test-id",
        content="Test content",
        metadata={"source": "test"},
        timestamp=time.time(),
        importance=0.7,
        memory_type="episodic"
    )

    # Verify attributes
    assert segment.id == "test-id"
    assert segment.content == "Test content"
    assert segment.metadata["source"] == "test"
    assert segment.importance == 0.7
    assert segment.memory_type == "episodic"
    assert isinstance(segment.timestamp, float)


@pytest.mark.asyncio
async def test_memory_context_model():
    """Test the MemoryContext model."""
    # Create a memory context
    context = MemoryContext(
        working_memory={"focus": "test"},
        active_concepts={"concept1", "concept2"},
        recent_memories=["id1", "id2"],
        context_id="test-context",
        creation_time=time.time()
    )

    # Verify attributes
    assert context.working_memory["focus"] == "test"
    assert "concept1" in context.active_concepts
    assert "id1" in context.recent_memories
    assert context.context_id == "test-context"
    assert isinstance(context.creation_time, float)
