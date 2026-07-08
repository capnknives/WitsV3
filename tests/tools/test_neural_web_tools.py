"""Tests for the neural web tools' dependency-injection wiring.

These tools (enhanced_reasoning, neural_web_nlp_extract, neural_web_visualize)
were previously unreachable: the tool_registry only auto-discovers tools with
zero required constructor args, and these took (config, llm_interface). They
also had a constructor bug (`super().__init__(config)` against a BaseTool
that expects `(name, description)`), so they would raise if ever
instantiated. See planning/roadmap/composer-orchestrator-search-quality-2026-07.md
Tier 3 #9.
"""

from typing import AsyncGenerator, List

import pytest

from core.config import WitsV3Config
from core.llm_interface import BaseLLMInterface, LLMResponse
from core.memory_manager import MemoryManager
from core.neural_memory_backend import NeuralMemoryBackend
from core.tool_registry import ToolRegistry
from tools.enhanced_reasoning import EnhancedReasoningTool
from tools.neural_web_nlp import NeuralWebNLPTool
from tools.neural_web_visualization import NeuralWebVisualizationTool


class DummyLLM(BaseLLMInterface):
    """Stub LLM: never returns usable JSON, so callers fall back to pattern-based logic."""

    def __init__(self):
        pass

    async def generate_text(self, prompt: str, **kwargs) -> str:
        return ""

    async def stream_text(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        yield ""

    async def get_embedding(self, text, model=None) -> List[float]:
        return [0.0] * 8

    async def generate_response(self, messages) -> LLMResponse:
        return LLMResponse(content="")


@pytest.fixture
def config(tmp_path):
    cfg = WitsV3Config()
    cfg.memory_manager.backend = "neural"
    cfg.memory_manager.neural_web_path = str(tmp_path / "neural_web.json")
    return cfg


def test_tools_construct_with_zero_args_and_are_auto_discovered():
    """Registry auto-discovery only instantiates tools with no required args."""
    registry = ToolRegistry()
    assert registry.get_tool("enhanced_reasoning") is not None
    assert registry.get_tool("neural_web_nlp_extract") is not None
    assert registry.get_tool("neural_web_visualize") is not None


@pytest.mark.asyncio
async def test_tools_error_cleanly_before_set_dependencies():
    """Before set_dependencies() is called, execute() should fail with a clear
    error instead of raising (e.g. AttributeError on a None llm_interface)."""
    for tool in (EnhancedReasoningTool(), NeuralWebNLPTool(), NeuralWebVisualizationTool()):
        if isinstance(tool, EnhancedReasoningTool):
            result = await tool.execute(goal="test", domain="general")
        elif isinstance(tool, NeuralWebNLPTool):
            result = await tool.execute(text="test")
        else:
            result = await tool.execute(format="png")

        assert result.success is False
        assert "set_dependencies" in result.error


@pytest.mark.asyncio
async def test_set_dependencies_resolves_live_neural_web(config):
    llm = DummyLLM()
    memory = MemoryManager(config=config, llm_interface=llm)
    await memory.initialize()
    assert isinstance(memory.backend, NeuralMemoryBackend)

    viz_tool = NeuralWebVisualizationTool()
    viz_tool.set_dependencies(config, llm, memory)
    assert viz_tool._neural_web is memory.backend.neural_web

    nlp_tool = NeuralWebNLPTool()
    nlp_tool.set_dependencies(config, llm, memory)
    assert nlp_tool._neural_web is memory.backend.neural_web

    reasoning_tool = EnhancedReasoningTool()
    reasoning_tool.set_dependencies(config, llm, memory)
    assert reasoning_tool._neural_web is memory.backend.neural_web


@pytest.mark.asyncio
async def test_set_dependencies_without_neural_backend_leaves_neural_web_none(config):
    config.memory_manager.backend = "basic"
    config.memory_manager.memory_file_path = str(config.memory_manager.neural_web_path) + ".basic.json"
    llm = DummyLLM()
    memory = MemoryManager(config=config, llm_interface=llm)
    await memory.initialize()
    assert not isinstance(memory.backend, NeuralMemoryBackend)

    viz_tool = NeuralWebVisualizationTool()
    viz_tool.set_dependencies(config, llm, memory)
    assert viz_tool._neural_web is None

    # Falls back to demo data rather than erroring.
    neural_web = await viz_tool._get_neural_web()
    assert neural_web is not None
    assert len(neural_web.concepts) > 0


@pytest.mark.asyncio
async def test_nlp_tool_adds_concepts_to_live_neural_web(config):
    llm = DummyLLM()
    memory = MemoryManager(config=config, llm_interface=llm)
    await memory.initialize()

    tool = NeuralWebNLPTool()
    tool.set_dependencies(config, llm, memory)

    result = await tool.execute(
        text="Python enables machine learning and artificial intelligence research.",
        add_to_neural_web=True,
    )

    assert result.success is True
    assert len(memory.backend.neural_web.concepts) > 0
