"""A normal turn must not store INTENT_ANALYSIS segments (guard in the control center run)."""

import json

import pytest

from agents.wits_control_center_agent import WitsControlCenterAgent
from core.config import WitsV3Config
from tests.agents.test_wcca_routing import FakeMemoryManager, ScriptedLLM

INTENT_JSON = json.dumps(
    {
        "type": "question",
        "complexity": "simple",
        "suggested_response": "direct_answer",
        "requires_tools": False,
    }
)


class RecordingMemoryManager(FakeMemoryManager):
    """Records the segment type of every write, whichever entry point is used."""

    def __init__(self):
        super().__init__({})
        self.stored_types: list[str] = []

    async def add_memory(self, *args, **kwargs):
        self.stored_types.append(kwargs.get("segment_type") or kwargs.get("type") or "?")

    async def add_segment(self, segment, *args, **kwargs):
        self.stored_types.append(segment.type)
        return segment.id

    async def add_memory_batch(self, segments):
        self.stored_types.extend(s.type for s in segments)


@pytest.mark.asyncio
async def test_new_turn_stores_no_intent_analysis_segment(monkeypatch):
    memory = RecordingMemoryManager()
    agent = WitsControlCenterAgent(
        agent_name="TestWCCA",
        config=WitsV3Config(),
        llm_interface=ScriptedLLM(INTENT_JSON),
        memory_manager=memory,
    )
    agent.has_enhanced_capabilities = False
    agent.meta_reasoning = None

    # Reason: also record at the agent seam in case store_memory bypasses the manager fake.
    seen: list[str] = []
    original = agent.store_memory

    async def spy(*args, **kwargs):
        seen.append(kwargs.get("segment_type", ""))
        return await original(*args, **kwargs)

    monkeypatch.setattr(agent, "store_memory", spy)

    async for _ in agent.run("what is the capital of France?", None, session_id="t"):
        pass

    assert "USER_INPUT" in seen  # the turn really went through the storing path
    assert "INTENT_ANALYSIS" not in seen
    assert "INTENT_ANALYSIS" not in memory.stored_types
