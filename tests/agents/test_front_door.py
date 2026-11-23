"""conversation.front_door (C3c): owner turns skip the intent router on native."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from agents.native_tool_orchestrator import NativeToolOrchestrator
from agents.wits_control_center_agent import WitsControlCenterAgent
from core.config import WitsV3Config
from core.golden import evaluate, load_cases
from core.llm_interface import LLMResponse
from core.turn_metrics import start_turn
from tests.agents.test_native_tool_orchestrator import _FakeRegistry

pytestmark = pytest.mark.asyncio

TURNS = Path(__file__).resolve().parents[1] / "golden" / "turns.yaml"
SOFTWARE = "Its richard i created the software you are wearing"


class _NoIntentLLM:
    """Answers chat calls; fails the test if any text prompt (the intent prompt) is asked."""

    def __init__(self, reply: str = "Hi Richard, good to see you."):
        self.reply = reply
        self.chat_calls = 0

    async def generate_chat(self, messages, tools=None, model=None, temperature=None):
        self.chat_calls += 1
        return LLMResponse(content=self.reply)

    async def generate_text(self, prompt, **kwargs):
        pytest.fail(f"text prompt requested on the native path: {prompt[:80]!r}")

    async def stream_text(self, prompt, **kwargs):
        pytest.fail("text stream requested on the native path")
        yield ""  # pragma: no cover


def _make(front_door: str, native: bool = True):
    config = WitsV3Config()
    config.conversation.front_door = front_door
    llm = _NoIntentLLM()
    registry = _FakeRegistry()
    orch = NativeToolOrchestrator("native", config, llm, None, registry) if native else None
    agent = WitsControlCenterAgent("wcca", config, llm, None, native_orchestrator=orch)
    agent.has_enhanced_capabilities = False
    agent.meta_reasoning = None
    return agent, llm, registry


def _forbid_legacy(agent, monkeypatch) -> None:
    """Fail on any legacy routing step the native path must skip."""

    def boom(name):
        def _fail(*args, **kwargs):
            pytest.fail(f"legacy step {name} ran on the native path")

        return _fail

    for name in ("_analyze_user_intent", "_build_intent_analysis_prompt", "_classify_routing"):
        monkeypatch.setattr(agent, name, boom(name))

    async def no_store(*args, **kwargs):
        pytest.fail(f"memory store on the native path: {kwargs.get('segment_type')}")

    monkeypatch.setattr(agent, "store_memory", no_store)


def _spy_legacy(agent, monkeypatch) -> list[str]:
    """Record that the intent router ran and answer with a fixed legacy reply."""
    seen: list[str] = []

    async def intent(user_input, *args, **kwargs):
        seen.append(user_input)
        return {"type": "conversation"}

    async def handle(*args, **kwargs):
        yield agent.stream_result("legacy reply")

    monkeypatch.setattr(agent, "_analyze_user_intent", intent)
    monkeypatch.setattr(agent, "_handle_intent_response", handle)
    return seen


async def _run(agent, message: str, **kwargs) -> tuple[list, object]:
    turn = start_turn("test", "s1", len(message))
    events = [e async for e in agent.run(message, None, session_id="s1", **kwargs)]
    return events, turn


async def test_native_owner_software_sentence_one_call_no_tool(monkeypatch):
    agent, llm, registry = _make("native")
    _forbid_legacy(agent, monkeypatch)
    events, turn = await _run(agent, SOFTWARE, user_role="owner")
    assert llm.chat_calls == 1
    assert registry.executed == []
    assert events[-1].type == "result" and events[-1].content
    assert turn.path == "native"
    case = next(c for c in load_cases(TURNS) if c.id == "owner-software-statement")
    result = evaluate(case, events, tool_names=[name for name, _ in registry.executed])
    assert result.failures == []


async def test_native_cli_owner_role_takes_native_path(monkeypatch):
    agent, llm, _ = _make("native")
    _forbid_legacy(agent, monkeypatch)
    # Reason: RT-10, the CLI (run.py) now passes user_role="owner"; no role is a guest.
    events, turn = await _run(agent, "good morning", user_role="owner")
    assert llm.chat_calls == 1 and turn.path == "native"


async def test_native_guest_takes_legacy_path(monkeypatch):
    agent, llm, _ = _make("native")
    seen = _spy_legacy(agent, monkeypatch)
    events, turn = await _run(agent, "good morning", user_role="guest")
    assert seen == ["good morning"]
    assert llm.chat_calls == 0
    assert events[-1].content == "legacy reply"
    assert turn.path == "legacy"


@pytest.mark.parametrize("role", [None, "", "admin"])
async def test_native_unknown_role_fails_closed(monkeypatch, role):
    agent, llm, _ = _make("native")
    seen = _spy_legacy(agent, monkeypatch)
    await _run(agent, "good morning", user_role=role)
    assert seen and llm.chat_calls == 0


async def test_native_missing_loop_falls_back_to_legacy_and_warns_once(monkeypatch, caplog):
    agent, _, _ = _make("native", native=False)
    seen = _spy_legacy(agent, monkeypatch)
    with caplog.at_level(logging.WARNING):
        for _ in range(2):
            events, turn = await _run(agent, "good morning", user_role="owner")
            assert events[-1].content == "legacy reply" and turn.path == "legacy"
    assert len(seen) == 2
    warnings = [r for r in caplog.records if "no native loop" in r.getMessage()]
    assert len(warnings) == 1


async def test_legacy_setting_never_uses_native_loop(monkeypatch):
    agent, llm, _ = _make("legacy")
    seen = _spy_legacy(agent, monkeypatch)
    events, turn = await _run(agent, SOFTWARE, user_role="owner")
    assert seen == [SOFTWARE] and llm.chat_calls == 0
    assert events[-1].content == "legacy reply"
    assert turn.path == "legacy"


@pytest.mark.parametrize("front_door", ["legacy", "native"])
async def test_remember_returns_early_on_both_settings(monkeypatch, front_door):
    agent, llm, _ = _make(front_door)
    stored: list[str] = []

    async def store(*args, **kwargs):
        stored.append(kwargs.get("segment_type"))

    monkeypatch.setattr(agent, "store_memory", store)
    monkeypatch.setattr(
        agent, "_analyze_user_intent", lambda *a, **k: pytest.fail("intent after remember")
    )
    events, _ = await _run(agent, "remember that my truck is blue", user_role="owner")
    assert stored == ["USER_FACT"]
    assert llm.chat_calls == 0
    assert [e.type for e in events] == ["result"]
