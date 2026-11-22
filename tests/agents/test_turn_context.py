"""Per-turn caller context (RT-4 follow-up): default outside a turn, aclose, tasks,
sub-agent memory writes and direct calls. No model, no network."""

import asyncio
from collections.abc import AsyncGenerator

import pytest

from agents.base_agent import BaseAgent
from agents.turn_context import TurnContext, bind_turn_if_unbound, current_turn, turn_for
from agents.wits_control_center_agent import WitsControlCenterAgent
from core.config import WitsV3Config
from core.llm_interface import BaseLLMInterface


class DummyLLM(BaseLLMInterface):
    def __init__(self):
        pass

    async def generate_text(self, prompt: str, **kwargs) -> str:
        return "dummy"

    async def stream_text(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        yield "dummy"

    async def get_embedding(self, text, model=None):
        return [0.0] * 8


class RecordingMemory:
    def __init__(self):
        self.added: list[str] = []

    async def add_segment(self, segment):
        self.added.append(segment.content.text)
        return f"seg-{len(self.added)}"

    async def get_recent_memory(self, limit, filter_dict=None):
        return []


class SubAgent(BaseAgent):
    """Stands in for a specialised/orchestrator agent called directly with a role."""

    @bind_turn_if_unbound
    async def run(self, user_input, conversation_history=None, session_id=None, **kwargs):
        await self.store_memory(content=user_input)
        yield self.stream_result(turn_for(self).user_role)


@pytest.fixture
def memory():
    return RecordingMemory()


@pytest.fixture
def agent(monkeypatch, tmp_path, memory):
    monkeypatch.chdir(tmp_path)
    config = WitsV3Config()
    config.memory_manager.tiered_enabled = False
    wcca = WitsControlCenterAgent(
        agent_name="TC", config=config, llm_interface=DummyLLM(), memory_manager=memory
    )
    wcca.has_enhanced_capabilities = False
    wcca.meta_reasoning = None
    return wcca


def test_read_outside_turn_gets_owner_default(agent):
    agent._request_user_role = "guest"  # stale shared attributes are never consulted
    agent._skip_global_memory_store = True
    assert current_turn() is None
    assert turn_for(agent) == TurnContext()


async def test_aclose_mid_turn_runs_cleanup_promptly(agent):
    cleaned: list[str] = []

    async def fake_run_turn(turn, *args, **kwargs):
        try:
            yield agent.stream_result("first")
            yield agent.stream_result("second")
        finally:
            cleaned.append(turn.user_role)

    agent._run_turn = fake_run_turn
    gen = agent.run("hi", session_id="s1", user_role="guest")
    await gen.__anext__()
    await gen.aclose()
    assert cleaned == ["guest"]


async def test_task_spawned_in_guest_step_sees_guest(agent):
    async def probe():
        return current_turn().user_role

    async def fake_run_turn(turn, *args, **kwargs):
        role = await asyncio.create_task(probe())
        yield agent.stream_result(role)

    agent._run_turn = fake_run_turn
    events = [e async for e in agent.run("hi", session_id="s2", user_role="guest")]
    assert [e.content for e in events] == ["guest"]
    assert current_turn() is None


async def test_sub_agent_store_memory_skipped_in_guest_turn(agent, memory):
    sub = SubAgent("Sub", agent.config, DummyLLM(), memory_manager=memory)

    async def fake_run_turn(turn, *args, **kwargs):
        await sub.store_memory(content="guest secret")
        async for event in sub.run("guest nested"):
            yield event

    agent._run_turn = fake_run_turn
    events = [e async for e in agent.run("hi", session_id="s3", user_role="guest")]
    assert [e.content for e in events] == ["guest"]
    assert memory.added == []


async def test_sub_agent_called_directly_binds_its_own_role(memory):
    sub = SubAgent("Sub", WitsV3Config(), DummyLLM(), memory_manager=memory)
    events = [e async for e in sub.run("guest direct", user_role="guest")]
    assert [e.content for e in events] == ["guest"]
    assert memory.added == []
    assert current_turn() is None


@pytest.mark.parametrize("direct", [True, False])
async def test_direct_guest_call_behaves_as_guest(agent, memory, direct):
    msg = "remember my locker code is 1234"
    if direct:
        gen = agent._run_turn(TurnContext.for_request(user_role="guest"), msg, session_id="s4")
    else:
        gen = agent.run(msg, session_id="s4", user_role="guest")
    events = [e async for e in gen]
    assert events
    assert memory.added == []
