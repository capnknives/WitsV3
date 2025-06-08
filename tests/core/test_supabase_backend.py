import pytest
import asyncio
from typing import AsyncGenerator
from unittest.mock import MagicMock, patch
from core.supabase_backend import SupabaseMemoryBackend
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
        return [0.1] * 384

@pytest.mark.asyncio
async def test_supabase_add_and_get_segment():
    """Test adding and retrieving memory segments with mocked Supabase."""

    # Create a mocked Supabase client
    with patch('core.supabase_backend.create_client') as mock_create_client:
        mock_supabase = MagicMock()
        mock_create_client.return_value = mock_supabase

        # Mock table operations
        mock_table = MagicMock()
        mock_supabase.table.return_value = mock_table

        # Mock insert operation
        mock_insert = MagicMock()
        mock_table.insert.return_value = mock_insert
        mock_insert.execute.return_value = MagicMock(data=[{"id": "test-id"}])

                # Mock select operation for get_recent_segments
        mock_select = MagicMock()
        mock_table.select.return_value = mock_select
        mock_order = MagicMock()
        mock_select.order.return_value = mock_order
        mock_limit = MagicMock()
        mock_order.limit.return_value = mock_limit
        mock_limit.execute.return_value = MagicMock(data=[{
            "id": "test-id",
            "type": "test",
            "source": "unit-test",
            "content": {"text": "Test content"},
            "metadata": {"test": True},
            "importance": 0.5,
            "embedding": [0.1] * 384
        }])

        # Create config and LLM
        config = WitsV3Config()
        config.supabase.url = "https://test.supabase.co"
        config.supabase.key = "test-key"

        llm = DummyLLM()

        # Initialize the backend
        backend = SupabaseMemoryBackend(config, llm)

        # Create a test memory segment
        segment = MemorySegment(
            id="test-id",
            type="test",
            source="unit-test",
            content=MemorySegmentContent(text="Test content"),
            metadata={"test": True}
        )

        # Test adding the segment
        await backend.add_segment(segment)

        # Verify insert was called
        mock_table.insert.assert_called_once()

        # Test retrieving the segment
        retrieved_segments = await backend.get_recent_segments()

        # Verify select was called
        mock_table.select.assert_called()

        # Verify we got the expected result
        assert len(retrieved_segments) == 1
        assert retrieved_segments[0].id == "test-id"
        assert retrieved_segments[0].content.text == "Test content"
