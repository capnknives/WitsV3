"""RT-4: caller role lives on the one shared WitsControlCenterAgent (run.py builds one
instance; web/server.py streams every /api/chat turn through it with no lock).

run() writes self._request_user_role / _skip_global_memory_store /
_guest_personalization_context, then yields and awaits before reading them back.
Each test interleaves two turns on ONE agent; ordering is forced by stepping the
generators by hand (the SSE consumer awaits between events) or by an asyncio.Event
gate in the fake memory manager. No sleeps, no model, no network.
"""

import asyncio
from collections.abc import AsyncGenerator
from types import SimpleNamespace

import pytest

from agents import routing_classifier as rc
from agents.wits_control_center_agent import WitsControlCenterAgent
from core.config import WitsV3Config
from core.llm_interface import BaseLLMInterface

OWNER_ONLY = ("guest_chat_history", "guest_profile", "knowledge_log")
GUEST_MSG = "list the recurring bugs zebra-guest"
OWNER_MSG = "hello there owner-turn"


class DummyLLM(BaseLLMInterface):
    def __init__(self):
        pass

    async def generate_text(self, prompt: str, **kwargs) -> str:
        return "dummy"

    async def stream_text(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        yield "dummy"

    async def get_embedding(self, text, model=None):
        return [0.0] * 8


class GatedMemory:
    """Records add_segment; the first get_recent_memory call can block on a gate."""

    def __init__(self, gate_first_inventory: bool = False):
        self.added: list[str] = []
        self.gate = asyncio.Event()
        self.parked = asyncio.Event()
        self._gate_pending = gate_first_inventory

    async def add_segment(self, segment):
        self.added.append(segment.content.text)
        return f"seg-{len(self.added)}"

    async def get_recent_memory(self, limit, filter_dict=None):
        if self._gate_pending:
            self._gate_pending = False
            self.parked.set()
            await self.gate.wait()
        return []


@pytest.fixture
def make_agent(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    def _make(memory, tiered: bool = False):
        config = WitsV3Config()
        config.memory_manager.tiered_enabled = tiered
        agent = WitsControlCenterAgent(
            agent_name="RT4", config=config, llm_interface=DummyLLM(), memory_manager=memory
        )
        agent.has_enhanced_capabilities = False
        agent.meta_reasoning = None
        return agent

    return _make


def _turn(agent, msg, role, sid):
    return agent.run(msg, session_id=sid, user_role=role).__aiter__()


async def _step_to_first_yield(gen):
    """Advance to 'Analyzing your request...' (wits_control_center_agent.py:216)."""
    event = await gen.__anext__()
    assert "Analyzing" in str(event.content)


async def _drain(gen):
    return [e async for e in gen]


async def test_guest_input_not_written_to_global_memory(make_agent):
    mem = GatedMemory()
    agent = make_agent(mem)
    guest = _turn(agent, GUEST_MSG, "guest", "g")
    await _step_to_first_yield(guest)
    await _step_to_first_yield(_turn(agent, OWNER_MSG, "owner", "o"))
    await _drain(guest)
    assert not any("zebra-guest" in text for text in mem.added)


async def test_guest_classifier_sees_guest_role(make_agent, monkeypatch):
    mem = GatedMemory(gate_first_inventory=True)
    agent = make_agent(mem)
    seen: list[tuple[str, str]] = []
    real = rc.classify_message

    def spy(ctx):
        seen.append((ctx.message, ctx.user_role))
        return real(ctx)

    monkeypatch.setattr(rc, "classify_message", spy)
    guest_task = asyncio.ensure_future(_drain(_turn(agent, GUEST_MSG, "guest", "g")))
    await mem.parked.wait()
    await _step_to_first_yield(_turn(agent, OWNER_MSG, "owner", "o"))
    mem.gate.set()
    await guest_task
    assert seen == [(GUEST_MSG, "guest")]


async def test_guest_does_not_reach_owner_only_route(make_agent, monkeypatch):
    agent = make_agent(GatedMemory())
    routes: list[str | None] = []
    ran: list[str] = []
    real = rc.classify_message
    monkeypatch.setattr(rc, "classify_message", lambda ctx: routes.append(r := real(ctx)) or r)
    import tools.knowledge_log_tool as klt

    async def fake_execute(self, **kwargs):
        ran.append(kwargs.get("user_role"))
        return "OWNER KNOWLEDGE"

    monkeypatch.setattr(klt.KnowledgeLogSummaryTool, "execute", fake_execute)
    guest = _turn(agent, GUEST_MSG, "guest", "g")
    await _step_to_first_yield(guest)
    await _step_to_first_yield(_turn(agent, OWNER_MSG, "owner", "o"))
    out = await _drain(guest)
    assert not [r for r in routes if r.destination in OWNER_ONLY]
    assert ran == [] and not any("OWNER KNOWLEDGE" in str(e.content) for e in out)


async def test_guest_does_not_get_owner_core_memory(make_agent, monkeypatch):
    import core.core_memory as cm

    fake_store = SimpleNamespace(as_prompt_block=lambda: "OWNER_CORE_SECRET")
    monkeypatch.setattr(cm, "get_core_memory_store", lambda config=None: fake_store)
    agent = make_agent(GatedMemory(), tiered=True)
    received: list[dict] = []

    async def fake_orch_run(**kwargs):
        received.append(kwargs)
        yield agent.stream_result("orchestrated")

    agent.orchestrator_agent = SimpleNamespace(run=fake_orch_run, tool_registry=None)
    guest = _turn(agent, "save our conversation as notes.txt", "guest", "g")
    await _step_to_first_yield(guest)
    await _drain(_turn(agent, OWNER_MSG, "owner", "o"))
    await _drain(guest)
    assert received, "guest turn should reach the orchestrator"
    assert all("OWNER_CORE_SECRET" not in k["guest_personalization_context"] for k in received)


async def test_owner_memory_write_not_skipped_by_guest_turn(make_agent):
    mem = GatedMemory()
    agent = make_agent(mem)
    owner = _turn(agent, OWNER_MSG, "owner", "o")
    await _step_to_first_yield(owner)
    await _step_to_first_yield(_turn(agent, GUEST_MSG, "guest", "g"))
    await _drain(owner)
    assert any("owner-turn" in text for text in mem.added)
