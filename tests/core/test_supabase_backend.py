import pytest
import asyncio
from core.supabase_backend import SupabaseMemoryBackend
from core.memory_manager import MemorySegment
from core.config import WitsV3Config
from core.llm_interface import BaseLLMInterface

class DummyLLM(BaseLLMInterface):
    async def get_embedding(self, text, model=None):
        return [0.1] * 384

@pytest.mark.asyncio
async def test_supabase_add_and_get_segment():
    config = WitsV3Config()
    config.supabase.url = "https://scdzgxvrppxpicinggy.supabase.co"
    config.supabase.key = "sbp_ee5abfbf912375dea50375d81d3a7e1bee1892d7"
    config.memory_manager.vector_dim = 384
    backend = SupabaseMemoryBackend(config, DummyLLM())
    await backend.initialize()
    segment = MemorySegment(
        id="test-id-andy",
        type="test",
        source="unit-test",
        content={"text": "Andy"},
        importance=1.0,
        embedding=[0.1] * 384,
        metadata={"test": True}
    )
    await backend.add_segment(segment)
    retrieved = await backend.get_segment("test-id-andy")
    assert retrieved is not None
    assert retrieved.content["text"] == "Andy"
