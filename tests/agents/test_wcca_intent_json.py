"""Tests for WCCA intent JSON robustness (July 2026 roadmap follow-up).

Mirrors orchestrator JSON repair patterns for intent analysis:
- Ollama structured output (format=json) on intent calls
- robust JSON extraction (think blocks, fences, truncated objects)
- repair-reparse round trip when parsing still fails
"""

from collections.abc import AsyncGenerator

import pytest

from agents.wits_control_center_agent import WitsControlCenterAgent
from core.config import WitsV3Config
from core.llm_interface import BaseLLMInterface


class ScriptedLLM(BaseLLMInterface):
    """Returns scripted responses in order and records every call."""

    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls = []

    async def generate_text(self, prompt: str, **kwargs) -> str:
        self.calls.append((prompt, kwargs))
        if self.responses:
            return self.responses.pop(0)
        return (
            '{"type": "direct_response", "confidence": 0.9, '
            '"reasoning": "default", "direct_response": "fallback"}'
        )

    async def stream_text(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        yield "dummy"

    async def get_embedding(self, text, model=None):
        return [0.0] * 8


@pytest.fixture
def wcca():
    return WitsControlCenterAgent(
        agent_name="TestWCCA",
        config=WitsV3Config(),
        llm_interface=ScriptedLLM(),
    )


def test_parses_clean_intent_json(wcca):
    parsed = wcca._parse_intent_response(
        '{"type": "goal_defined", "confidence": 0.9, "goal_statement": "do the thing"}'
    )
    assert parsed["type"] == "goal_defined"
    assert parsed["suggested_response"] == "orchestrator"
    assert "_parse_failed" not in parsed


def test_parses_intent_json_after_think_block(wcca):
    response = (
        "<think>classify this request</think>\n"
        '{"type": "direct_response", "confidence": 0.8, "direct_response": "Hi!"}'
    )
    parsed = wcca._parse_intent_response(response)
    assert parsed["type"] == "direct_response"
    assert parsed["direct_response"] == "Hi!"


def test_parses_intent_json_in_markdown_fence(wcca):
    response = (
        "Here is the intent:\n```json\n"
        '{"type": "clarification_question", "clarification_question": "Which file?"}\n'
        "```"
    )
    parsed = wcca._parse_intent_response(response)
    assert parsed["type"] == "clarification_question"
    assert parsed["clarification_question"] == "Which file?"


def test_repairs_trailing_comma_in_intent_json(wcca):
    response = '{"type": "direct_response", "direct_response": "ok",}'
    parsed = wcca._parse_intent_response(response)
    assert parsed["direct_response"] == "ok"
    assert "_parse_failed" not in parsed


def test_completes_truncated_intent_json(wcca):
    response = '{"type": "goal_defined", "goal_statement": "summarize the audit'
    parsed = wcca._parse_intent_response(response)
    assert parsed["type"] == "goal_defined"
    assert parsed["goal_statement"].startswith("summarize the audit")


def test_missing_type_flags_parse_failure(wcca):
    parsed = wcca._parse_intent_response('{"confidence": 0.5, "reasoning": "no type"}')
    assert parsed["_parse_failed"] is True
    assert "type" in parsed["_parse_error"]


def test_unparseable_response_flags_parse_failure(wcca):
    parsed = wcca._parse_intent_response("definitely not json at all")
    assert parsed["_parse_failed"] is True
    assert parsed["type"] == "goal_defined"


@pytest.fixture
def wcca_plain():
    """WCCA with enhanced meta-reasoning disabled so intent hits the LLM path."""
    agent = WitsControlCenterAgent(
        agent_name="TestWCCA",
        config=WitsV3Config(),
        llm_interface=ScriptedLLM(),
    )
    agent.has_enhanced_capabilities = False
    agent.meta_reasoning = None
    return agent


def _non_casual_message() -> str:
    """Long, non-question phrasing to bypass the casual-conversation shortcut."""
    return (
        "I need a detailed written comparison of mitosis and meiosis "
        "for my biology homework assignment"
    )


@pytest.mark.asyncio
async def test_intent_call_requests_json_format(wcca_plain):
    llm = ScriptedLLM(
        [
            '{"type": "direct_response", "confidence": 0.9, '
            '"reasoning": "explanation", "direct_response": "Hello!"}'
        ]
    )
    wcca_plain.llm_interface = llm

    intent = await wcca_plain._analyze_user_intent(_non_casual_message(), None)

    assert llm.calls
    _, kwargs = llm.calls[0]
    assert kwargs.get("format") == "json"
    assert intent["type"] == "direct_response"
    assert intent["direct_response"] == "Hello!"


@pytest.mark.asyncio
async def test_intent_repair_reparse_round_trip(wcca_plain):
    llm = ScriptedLLM(
        [
            '{"type": "goal_defined" "goal_statement": "broken"}',
            '{"type": "goal_defined", "confidence": 0.9, "goal_statement": "fixed goal"}',
        ]
    )
    wcca_plain.llm_interface = llm

    intent = await wcca_plain._analyze_user_intent(
        "please compare mitosis and meiosis for a biology student", None
    )

    assert len(llm.calls) == 2
    repair_prompt, repair_kwargs = llm.calls[1]
    assert "failed to parse" in repair_prompt
    assert repair_kwargs.get("format") == "json"
    assert intent["type"] == "goal_defined"
    assert intent["goal_statement"] == "fixed goal"
    assert intent["suggested_response"] == "orchestrator"


@pytest.mark.asyncio
async def test_intent_repair_failure_uses_heuristic_without_crashing(wcca_plain):
    llm = ScriptedLLM(
        [
            "not json at all",
            "still not json",
        ]
    )
    wcca_plain.llm_interface = llm

    intent = await wcca_plain._analyze_user_intent(
        "please describe the main themes in nineteenth century literature", None
    )

    assert len(llm.calls) == 2
    assert intent["type"] == "goal_defined"
    assert intent["suggested_response"] == "orchestrator"
