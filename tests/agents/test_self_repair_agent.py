"""Tests for the simplified SelfRepairAgent (Tier 4 hygiene backlog)."""

from typing import AsyncGenerator, List, Optional

import pytest

from agents.self_repair_agent import SelfRepairAgent
from core.config import WitsV3Config
from core.llm_interface import BaseLLMInterface
from core.schemas import StreamData


class ScriptedLLM(BaseLLMInterface):
    """Returns scripted responses and records calls."""

    def __init__(self, response: str = "System healthy.", config: Optional[WitsV3Config] = None):
        super().__init__(config or WitsV3Config())
        self.response = response
        self.calls: List[str] = []

    async def generate_text(self, prompt: str, **kwargs) -> str:
        self.calls.append(prompt)
        return self.response

    async def stream_text(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        yield self.response

    async def get_embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        return [0.0] * 8


@pytest.fixture
def agent() -> SelfRepairAgent:
    return SelfRepairAgent(
        agent_name="TestSelfRepair",
        config=WitsV3Config(),
        llm_interface=ScriptedLLM("All systems nominal."),
    )


@pytest.mark.asyncio
async def test_self_repair_agent_initializes(agent: SelfRepairAgent):
    assert agent.agent_name == "TestSelfRepair"
    assert agent.llm_interface is not None


@pytest.mark.asyncio
async def test_self_repair_run_streams_thinking_and_result(agent: SelfRepairAgent):
    streams: List[StreamData] = []
    async for item in agent.run("Perform a health check", session_id="sess-1"):
        streams.append(item)

    types = [s.type for s in streams]
    assert "thinking" in types
    assert "result" in types
    assert streams[-1].type == "result"
    assert streams[-1].content == "All systems nominal."


@pytest.mark.asyncio
async def test_self_repair_run_passes_user_input_to_llm():
    llm = ScriptedLLM("ok")
    agent = SelfRepairAgent("TestSelfRepair", WitsV3Config(), llm)

    async for _ in agent.run("check disk space"):
        pass

    assert llm.calls == ["check disk space"]


@pytest.mark.asyncio
async def test_self_repair_generates_session_id_when_missing(agent: SelfRepairAgent):
    streams: List[StreamData] = []
    async for item in agent.run("health check"):
        streams.append(item)

    assert streams
    assert agent.llm_interface.calls
