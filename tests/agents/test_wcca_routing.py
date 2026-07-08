"""Tests for WitsControlCenterAgent intent routing.

Covers the July 7 2026 failures: document questions misrouted to casual
chat ("hi" substring-matched inside "things") or to clarification loops
because the intent analyzer had no knowledge of ingested documents.
"""

import inspect
from types import SimpleNamespace
from typing import AsyncGenerator

import pytest

from agents.advanced_coding_agent import AdvancedCodingAgent
from agents.base_orchestrator_agent import BaseOrchestratorAgent
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


class FakeMemoryManager:
    """Only what _get_document_inventory needs."""

    def __init__(self, doc_files):
        self._segments = [
            SimpleNamespace(metadata={"file_path": fp, "chunk_index": i})
            for fp, chunks in doc_files.items()
            for i in range(chunks)
        ]

    async def get_recent_memory(self, limit, filter_dict=None):
        return self._segments


@pytest.fixture
def wcca():
    agent = WitsControlCenterAgent(
        agent_name="TestWCCA",
        config=WitsV3Config(),
        llm_interface=DummyLLM(),
        memory_manager=FakeMemoryManager(
            {"Pleistocene_Megafauna_Audit_Report.md": 16, "proof_of_enrollment.v3 (2).pdf": 1}
        ),
    )
    # Force the plain routing path (no meta-reasoning shortcut)
    agent.has_enhanced_capabilities = False
    agent.meta_reasoning = None
    return agent


# ------------------------------------------------------- casual heuristic

def test_casual_word_requires_word_boundary(wcca):
    # "things" must not match casual word "hi" (regression: substring match)
    assert wcca._is_casual_conversation(
        "i've updated things, please check the results once more"
    ) is False


def test_greetings_still_casual(wcca):
    assert wcca._is_casual_conversation("hi there, how are you doing today my friend") is True
    assert wcca._is_casual_conversation("thanks!") is True


# ------------------------------------------------------- document routing

@pytest.mark.asyncio
@pytest.mark.parametrize("message", [
    # The three phrasings that failed on July 7 2026
    "i've updated things, please check the audit again",
    "summarize the audit report you have access to.",
    "I shared Pleistocene_Megafauna_Audit_Report.md with you, summarize it.",
])
async def test_document_mentions_route_to_orchestrator(wcca, message):
    intent = await wcca._analyze_user_intent(message, None)
    assert intent["suggested_response"] == "orchestrator"
    assert intent["requires_tools"] is True


# ------------------------------------------------- web-search routing

@pytest.mark.parametrize("message", [
    "What famous musician died of june 14th 2026?",   # the reported failure
    "look it up",                                     # explicit follow-up command
    "who won the world cup this year?",
    "what's the latest news on Ollama?",
    "search the web for python 3.14 release date",
    "what's the weather in Seattle right now?",
])
def test_current_info_questions_need_web_search(wcca, message):
    assert wcca._needs_web_search(message) is True


@pytest.mark.parametrize("message", [
    "hi there, how are you doing today my friend",  # 'today' must NOT trigger
    "thanks, that was helpful",
    "what can you do?",
    "write me a python function to reverse a list",
])
def test_ordinary_chat_does_not_need_web_search(wcca, message):
    assert wcca._needs_web_search(message) is False


@pytest.mark.asyncio
@pytest.mark.parametrize("message", [
    "What famous musician died of june 14th 2026?",
    "look it up",
])
async def test_current_info_routes_to_orchestrator(wcca, message):
    intent = await wcca._analyze_user_intent(message, None)
    assert intent["suggested_response"] == "orchestrator"
    assert intent["requires_tools"] is True


# ------------------------------------------------------- save-to-file routing

@pytest.mark.parametrize("message", [
    "Please save this conversation to a file",
    "Save a log of our conversations as a file",
    "write the story to disk as goku.txt",
])
def test_save_to_file_needs_orchestrator(wcca, message):
    assert wcca._needs_file_write(message) is True


@pytest.mark.asyncio
@pytest.mark.parametrize("message", [
    "Please save this conversation to exports/chat.txt",
    "Save a log of our conversations as a file",
])
async def test_save_to_file_routes_to_orchestrator(wcca, message):
    intent = await wcca._analyze_user_intent(message, None)
    assert intent["suggested_response"] == "orchestrator"
    assert intent["requires_tools"] is True
    assert "write_file" in intent["notes"] or "save" in intent["notes"].lower()


# ------------------------------------------------------- intent parsing

def test_goal_defined_routes_to_orchestrator(wcca):
    parsed = wcca._parse_intent_response(
        '{"type": "goal_defined", "confidence": 0.9, "goal_statement": "do the thing"}'
    )
    assert parsed["suggested_response"] == "orchestrator"
    assert parsed["requires_tools"] is True
    assert parsed["complexity"] == "moderate"


def test_direct_response_intent_metadata(wcca):
    parsed = wcca._parse_intent_response(
        '{"type": "direct_response", "direct_response": "Hey there!"}'
    )
    assert parsed["suggested_response"] == "direct"
    assert parsed["requires_tools"] is False
    assert parsed["direct_response"] == "Hey there!"


def test_clarification_intent_metadata(wcca):
    parsed = wcca._parse_intent_response(
        '{"type": "clarification_question", "clarification_question": "Which file?"}'
    )
    assert parsed["suggested_response"] == "clarification"
    assert parsed["requires_tools"] is False
    assert parsed["clarification_question"] == "Which file?"


# ---------------------------------------- intent handler (no double LLM)

class TrackingLLM(DummyLLM):
    """Records generate_text calls so handler tests can assert zero extra LLM work."""

    def __init__(self):
        self.calls = []

    async def generate_text(self, prompt: str, **kwargs) -> str:
        self.calls.append(prompt)
        return "unexpected llm output"


class MockOrchestrator:
    async def run(self, user_input, conversation_history=None, session_id=None, **kwargs):
        yield SimpleNamespace(type="result", content="orchestrator handled it", source="mock")


async def _collect_handler_results(wcca, intent, user_input):
    results = []
    async for stream_data in wcca._handle_intent_response(
        intent, user_input, None, "test-session"
    ):
        results.append(stream_data)
    return results


@pytest.mark.asyncio
async def test_handle_direct_response_uses_intent_text_without_llm(wcca):
    tracking = TrackingLLM()
    wcca.llm_interface = tracking
    intent = wcca._parse_intent_response(
        '{"type": "direct_response", "direct_response": "Hello from intent JSON"}'
    )
    results = await _collect_handler_results(wcca, intent, "thanks!")
    assert not tracking.calls
    assert any(r.type == "result" and r.content == "Hello from intent JSON" for r in results)


@pytest.mark.asyncio
async def test_handle_clarification_uses_intent_question_without_llm(wcca):
    tracking = TrackingLLM()
    wcca.llm_interface = tracking
    intent = wcca._parse_intent_response(
        '{"type": "clarification_question", "clarification_question": "Which audit report?"}'
    )
    results = await _collect_handler_results(wcca, intent, "summarize it")
    assert not tracking.calls
    assert any(r.type == "result" and r.content == "Which audit report?" for r in results)


@pytest.mark.asyncio
async def test_direct_response_overridden_when_web_search_needed(wcca):
    tracking = TrackingLLM()
    wcca.llm_interface = tracking
    wcca.orchestrator_agent = MockOrchestrator()
    intent = wcca._parse_intent_response(
        '{"type": "direct_response", "direct_response": "From memory: nobody"}'
    )
    results = await _collect_handler_results(
        wcca, intent, "What famous musician died on june 14th 2026?"
    )
    assert not tracking.calls
    assert any(r.type == "result" and r.content == "orchestrator handled it" for r in results)


@pytest.mark.asyncio
async def test_conversation_still_calls_llm_once(wcca):
    tracking = TrackingLLM()
    wcca.llm_interface = tracking
    intent = {
        "type": "conversation",
        "complexity": "simple",
        "requires_tools": False,
        "suggested_response": "direct",
    }
    await _collect_handler_results(wcca, intent, "hi there!")
    assert len(tracking.calls) == 1
    assert "casual conversation" in tracking.calls[0].lower()


# ------------------------------------------------------- documents context

def test_documents_context_lists_files(wcca):
    ctx = wcca._documents_context({"a_report.md": 3})
    assert "a_report.md (3 chunks)" in ctx
    assert "ALREADY ingested" in ctx


def test_documents_context_empty():
    assert "No user documents" in WitsControlCenterAgent._documents_context({})


# ---------------------------------------- agent delegation contract

def test_orchestrator_and_coding_agents_use_user_input_param():
    """WCCA delegates with user_input=; all BaseAgent subclasses must accept it."""
    orch_params = inspect.signature(BaseOrchestratorAgent.run).parameters
    coding_params = inspect.signature(AdvancedCodingAgent.run).parameters
    assert "user_input" in orch_params
    assert "goal" not in orch_params
    assert "user_input" in coding_params
    assert "request" not in coding_params
