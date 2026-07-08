"""Smoke tests for agents that previously had no dedicated pytest coverage."""

from typing import AsyncGenerator, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.book_writing_agent import BookWritingAgent
from agents.llm_driven_orchestrator import LLMDrivenOrchestrator
from agents.neural_orchestrator_agent import NeuralOrchestratorAgent
from core.config import WitsV3Config
from core.llm_interface import BaseLLMInterface
from core.schemas import StreamData


class MinimalLLM(BaseLLMInterface):
    def __init__(self, config: Optional[WitsV3Config] = None):
        super().__init__(config or WitsV3Config())

    async def generate_text(self, prompt: str, **kwargs) -> str:
        return '{"task_type": "general_writing", "topic": "testing"}'

    async def stream_text(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        yield '{"task_type": "general_writing"}'

    async def get_embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        return [0.0] * 8


@pytest.fixture
def config() -> WitsV3Config:
    return WitsV3Config()


@pytest.mark.asyncio
async def test_book_writing_agent_general_path_yields_result(config: WitsV3Config):
    agent = BookWritingAgent(
        agent_name="TestBookWriter",
        config=config,
        llm_interface=MinimalLLM(),
    )
    analysis = {
        "task_type": "general_writing",
        "genre": "non-fiction",
        "style": "expository",
        "topic": "software testing",
        "length": 100,
    }

    with patch.object(
        agent, "_analyze_writing_task", new=AsyncMock(return_value=analysis)
    ), patch.object(
        agent, "generate_response", new=AsyncMock(return_value="Sample essay content.")
    ), patch.object(agent, "store_memory", new=AsyncMock()):
        streams: List[StreamData] = []
        async for item in agent.run("Write a short essay about testing"):
            streams.append(item)

    assert any(s.type == "thinking" for s in streams)
    assert any(s.type == "result" for s in streams)
    assert any("Sample essay content" in s.content for s in streams)


@pytest.mark.asyncio
async def test_neural_orchestrator_injects_neural_context(config: WitsV3Config):
    agent = NeuralOrchestratorAgent(
        agent_name="TestNeuralOrch",
        config=config,
        llm_interface=MinimalLLM(),
        tool_registry=MagicMock(),
    )
    agent.cross_domain_learning = None
    state = {
        "goal": "Explain photosynthesis",
        "observations": [],
        "completed": False,
    }
    mock_insights = {"active_concepts": ["biology", "energy"]}

    async def noop_parent_loop(self, state, session_id):
        """Stub parent ReAct loop — neural layer yields before this runs."""
        return
        yield  # pragma: no cover — makes this an async generator

    with patch.object(
        agent, "_get_neural_insights", new=AsyncMock(return_value=mock_insights)
    ), patch.object(
        LLMDrivenOrchestrator, "_execute_react_loop", new=noop_parent_loop
    ):
        streams: List[StreamData] = []
        async for item in agent._execute_react_loop(state, "sess-1"):
            streams.append(item)

    assert state.get("neural_context") == mock_insights
    thinking_streams = [s for s in streams if s.type == "thinking"]
    assert thinking_streams, "neural layer should yield before parent loop"
    assert any(
        "biology" in s.content or "concepts" in s.content.lower()
        for s in thinking_streams
    )
