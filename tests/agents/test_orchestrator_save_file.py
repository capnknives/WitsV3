"""Tests for save-to-file orchestrator tool arg injection."""

import pytest

from agents.base_orchestrator_agent import BaseOrchestratorAgent
from agents.llm_driven_orchestrator import LLMDrivenOrchestrator
from core.config import WitsV3Config
from core.llm_interface import BaseLLMInterface
from core.schemas import ConversationHistory


class _Harness(BaseOrchestratorAgent):
    def _build_reasoning_prompt(self, state):
        return ""

    def _parse_reasoning_response(self, response):
        return {}


class DummyLLM(BaseLLMInterface):
    def __init__(self):
        pass

    async def generate_text(self, prompt: str, **kwargs) -> str:
        return "{}"

    async def stream_text(self, prompt: str, **kwargs):
        yield ""

    async def get_embedding(self, text, model=None):
        return [0.0] * 8


def _harness() -> _Harness:
    return _Harness.__new__(_Harness)


def _conversation() -> ConversationHistory:
    conv = ConversationHistory(session_id="s1")
    conv.add_message("user", "Tell me a story")
    conv.add_message("assistant", "Once upon a time...")
    return conv


@pytest.mark.asyncio
async def test_prepare_write_file_fills_content_from_session():
    h = _harness()
    state = {
        "goal": "Save our conversation to exports/chat.txt",
        "conversation_history": _conversation(),
        "observations": [],
    }
    args = await h._prepare_tool_args("write_file", {"file_path": "exports/chat.txt"}, state)
    assert args["file_path"] == "var/exports/chat.txt"
    assert "USER: Tell me a story" in args["content"]
    assert "ASSISTANT: Once upon a time" in args["content"]


@pytest.mark.asyncio
async def test_prepare_write_file_prefers_read_history_observation():
    h = _harness()
    state = {
        "goal": "Save this conversation to a file",
        "conversation_history": _conversation(),
        "observations": ["Tool read_conversation_history result: USER: hi\n\nASSISTANT: hello"],
    }
    args = await h._prepare_tool_args("write_file", {"file_path": "out.txt"}, state)
    assert args["content"] == "USER: hi\n\nASSISTANT: hello"


@pytest.mark.asyncio
async def test_prepare_read_conversation_history_injects_session():
    h = _harness()
    conv = _conversation()
    state = {
        "goal": "Save our conversation to disk",
        "conversation_history": conv,
        "observations": [],
    }
    args = await h._prepare_tool_args("read_conversation_history", {}, state)
    assert args["conversation_history"] is conv
    assert args["max_messages"] == 0


def test_validate_reasoning_strips_huge_write_file_content():
    orch = LLMDrivenOrchestrator(
        agent_name="Test",
        config=WitsV3Config(),
        llm_interface=DummyLLM(),
    )
    parsed = {
        "action_type": "tool_call",
        "tool_name": "write_file",
        "tool_args": {
            "file_path": "big.txt",
            "content": "x" * 500,
        },
    }
    result = orch._validate_reasoning(parsed)
    assert "content" not in result["tool_args"]
    assert result["tool_args"]["file_path"] == "big.txt"
