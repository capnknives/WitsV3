"""CI-safe smoke scenario tests (routing tier, no Ollama)."""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest

from agents.wits_control_center_agent import WitsControlCenterAgent
from core.config import WitsV3Config
from core.llm_interface import BaseLLMInterface
from scripts.smoke_harness import filter_scenarios, load_scenarios, run_routing_scenario, SmokeRunState


class DummyLLM(BaseLLMInterface):
    def __init__(self, config: WitsV3Config | None = None):
        if config is not None:
            super().__init__(config)

    async def generate_text(self, prompt: str, **kwargs) -> str:
        return "dummy"

    async def stream_text(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        yield "dummy"

    async def get_embedding(self, text, model=None):
        return [0.0] * 8


class FakeMemoryManager:
    async def get_recent_memory(self, limit, filter_dict=None):
        return []

    async def initialize(self):
        pass


@pytest.fixture
def wcca():
    config = WitsV3Config()
    agent = WitsControlCenterAgent(
        agent_name="TestWCCA",
        config=config,
        llm_interface=DummyLLM(config),
        memory_manager=FakeMemoryManager(),
    )

    async def _empty_inventory():
        return {}

    agent._get_document_inventory = _empty_inventory  # type: ignore[method-assign]
    return agent


@pytest.mark.integration
@pytest.mark.parametrize(
    "scenario_id",
    [
        s["id"]
        for s in filter_scenarios(load_scenarios(), tiers={"routing", "guest"}, live=False)
    ],
)
async def test_routing_smoke_scenario(wcca, scenario_id: str):
    scenarios = {s["id"]: s for s in load_scenarios()}
    scenario = scenarios[scenario_id]
    state = SmokeRunState()
    await run_routing_scenario(scenario, wcca, state)
    result = state.results[-1]
    assert result.passed, f"{scenario_id} failed: {result.detail}"
