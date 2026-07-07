"""Tests for WitsControlCenterAgent intent routing.

Covers the July 7 2026 failures: document questions misrouted to casual
chat ("hi" substring-matched inside "things") or to clarification loops
because the intent analyzer had no knowledge of ingested documents.
"""

from types import SimpleNamespace
from typing import AsyncGenerator

import pytest

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


# ------------------------------------------------------- intent parsing

def test_goal_defined_routes_to_orchestrator(wcca):
    parsed = wcca._parse_intent_response(
        '{"type": "goal_defined", "confidence": 0.9, "goal_statement": "do the thing"}'
    )
    assert parsed["suggested_response"] == "orchestrator"
    assert parsed["requires_tools"] is True
    assert parsed["complexity"] == "moderate"


# ------------------------------------------------------- documents context

def test_documents_context_lists_files(wcca):
    ctx = wcca._documents_context({"a_report.md": 3})
    assert "a_report.md (3 chunks)" in ctx
    assert "ALREADY ingested" in ctx


def test_documents_context_empty():
    assert "No user documents" in WitsControlCenterAgent._documents_context({})
