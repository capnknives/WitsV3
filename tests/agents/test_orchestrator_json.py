"""Tests for orchestrator JSON robustness (July 2026 roadmap item #5).

Covers the qwen3 malformed-JSON failures from the ReAct loop
("Failed to parse reasoning response: Expecting ',' delimiter"):
- Ollama structured output (format=json) requested on reasoning calls
- robust JSON extraction (<think> blocks, markdown fences, multiple/truncated objects)
- conservative syntax repair (trailing commas, smart quotes, Python literals)
- repair-reparse round trip when parsing still fails
"""

from typing import AsyncGenerator

import pytest

from agents.llm_driven_orchestrator import LLMDrivenOrchestrator
from core.config import WitsV3Config
from core.llm_interface import BaseLLMInterface, OllamaInterface


class ScriptedLLM(BaseLLMInterface):
    """Returns scripted responses in order and records every call."""

    SAFE_FINAL = '{"thought": "done", "action_type": "final_answer", "final_answer": "scripted default"}'

    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls = []  # (prompt, kwargs) tuples

    async def generate_text(self, prompt: str, **kwargs) -> str:
        self.calls.append((prompt, kwargs))
        if self.responses:
            return self.responses.pop(0)
        return self.SAFE_FINAL

    async def stream_text(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        yield "dummy"

    async def get_embedding(self, text, model=None):
        return [0.0] * 8


def make_orchestrator(llm=None):
    return LLMDrivenOrchestrator(
        agent_name="TestOrchestrator",
        config=WitsV3Config(),
        llm_interface=llm or ScriptedLLM(),
    )


@pytest.fixture
def orchestrator():
    return make_orchestrator()


# ------------------------------------------------------------ parsing

def test_parses_clean_json(orchestrator):
    parsed = orchestrator._parse_reasoning_response(
        '{"thought": "t", "action_type": "tool_call", "tool_name": "document_search", "tool_args": {"query": "audit"}}'
    )
    assert parsed["action_type"] == "tool_call"
    assert parsed["tool_name"] == "document_search"
    assert parsed["tool_args"] == {"query": "audit"}
    assert "_parse_failed" not in parsed


def test_parses_json_after_think_block(orchestrator):
    # qwen3 emits <think>...</think> before the answer
    response = (
        "<think>The user wants the audit report, I should search.\n"
        'Some braces in here: {not json}</think>\n'
        '{"thought": "search docs", "action_type": "tool_call", "tool_name": "document_search", "tool_args": {"query": "audit"}}'
    )
    parsed = orchestrator._parse_reasoning_response(response)
    assert parsed["action_type"] == "tool_call"
    assert parsed["tool_name"] == "document_search"


def test_parses_json_in_markdown_fence(orchestrator):
    response = (
        "Here is my decision:\n```json\n"
        '{"thought": "t", "action_type": "final_answer", "final_answer": "42"}\n'
        "```"
    )
    parsed = orchestrator._parse_reasoning_response(response)
    assert parsed["action_type"] == "final_answer"
    assert parsed["final_answer"] == "42"


def test_skips_invalid_object_and_uses_later_one(orchestrator):
    # The old greedy r'\{.*\}' regex spanned from the first { to the last }
    # and always failed on responses like this.
    response = (
        'I considered {"irrelevant": 1} but decided:\n'
        '{"thought": "t", "action_type": "final_answer", "final_answer": "done"}'
    )
    parsed = orchestrator._parse_reasoning_response(response)
    assert parsed["action_type"] == "final_answer"
    assert parsed["final_answer"] == "done"


def test_repairs_trailing_comma(orchestrator):
    response = '{"thought": "t", "action_type": "final_answer", "final_answer": "ok",}'
    parsed = orchestrator._parse_reasoning_response(response)
    assert parsed["final_answer"] == "ok"
    assert "_parse_failed" not in parsed


def test_repairs_python_literals(orchestrator):
    response = '{"thought": "t", "action_type": "tool_call", "tool_name": "web_search", "tool_args": {"safe": True, "limit": None}}'
    parsed = orchestrator._parse_reasoning_response(response)
    assert parsed["tool_args"] == {"safe": True, "limit": None}


def test_completes_truncated_json(orchestrator):
    # Response cut off mid-string (e.g. num_predict limit)
    response = '{"thought": "t", "action_type": "tool_call", "tool_name": "document_search", "tool_args": {"query": "megafauna audit'
    parsed = orchestrator._parse_reasoning_response(response)
    assert parsed["action_type"] == "tool_call"
    assert parsed["tool_name"] == "document_search"
    assert parsed["tool_args"]["query"].startswith("megafauna audit")


def test_completes_truncated_json_unclosed_string_with_trailing_comma(orchestrator):
    # Truncated inside a string value that ends with a comma — comma must be
    # stripped before the closing quote, not after.
    response = '{"thought": "t", "action_type": "final_answer", "final_answer": "hello,'
    parsed = orchestrator._parse_reasoning_response(response)
    assert parsed["action_type"] == "final_answer"
    assert parsed["final_answer"] == "hello"


def test_missing_tool_args_defaults_to_empty(orchestrator):
    response = '{"thought": "t", "action_type": "tool_call", "tool_name": "think"}'
    parsed = orchestrator._parse_reasoning_response(response)
    assert parsed["tool_args"] == {}


def test_coerces_tool_name_used_as_action_type(orchestrator):
    response = (
        '{"thought": "look it up", "action_type": "web_search", '
        '"tool_args": {"query": "dragonball advent truth"}}'
    )
    parsed = orchestrator._parse_reasoning_response(response)
    assert parsed["action_type"] == "tool_call"
    assert parsed["tool_name"] == "web_search"
    assert parsed["tool_args"]["query"] == "dragonball advent truth"
    assert "_parse_failed" not in parsed


def test_coerces_action_type_with_top_level_query(orchestrator):
    response = (
        '{"thought": "search", "action_type": "web_search", '
        '"query": "dragonball advent truth game report"}'
    )
    parsed = orchestrator._parse_reasoning_response(response)
    assert parsed["action_type"] == "tool_call"
    assert parsed["tool_name"] == "web_search"
    assert parsed["tool_args"]["query"] == "dragonball advent truth game report"


def test_invalid_action_type_flags_failure(orchestrator):
    parsed = orchestrator._parse_reasoning_response(
        '{"thought": "hmm", "action_type": "definitely_not_a_valid_action"}'
    )
    assert parsed["_parse_failed"] is True


def test_unparseable_response_flags_parse_failure(orchestrator):
    parsed = orchestrator._parse_reasoning_response("complete nonsense, no json here at all")
    assert parsed["_parse_failed"] is True
    assert parsed["_parse_error"]
    assert parsed["action_type"] in ("tool_call", "final_answer")


def test_valid_json_missing_required_fields_flags_failure(orchestrator):
    parsed = orchestrator._parse_reasoning_response('{"thought": "no action type here"}')
    assert parsed["_parse_failed"] is True
    assert "action_type" in parsed["_parse_error"]


def test_fallback_strips_think_blocks(orchestrator):
    parsed = orchestrator._parse_reasoning_response(
        "<think>internal monologue</think>The final response text."
    )
    assert "internal monologue" not in parsed["thought"]
    assert "The final response text." in parsed["thought"]


# ------------------------------------------------------------ ollama payload

@pytest.mark.asyncio
async def test_prepare_payload_includes_format():
    iface = OllamaInterface(WitsV3Config())
    try:
        with_format = await iface._prepare_payload("p", format="json")
        without_format = await iface._prepare_payload("p")
        assert with_format["format"] == "json"
        assert "format" not in without_format
    finally:
        await iface.shutdown()


# ------------------------------------------------------------ react loop

async def run_to_completion(orchestrator, user_input="Summarize the audit report"):
    return [sd async for sd in orchestrator.run(user_input)]


@pytest.mark.asyncio
async def test_reasoning_call_requests_json_format():
    llm = ScriptedLLM(['{"thought": "t", "action_type": "final_answer", "final_answer": "all done"}'])
    orchestrator = make_orchestrator(llm)

    stream = await run_to_completion(orchestrator)

    assert llm.calls, "expected at least one LLM call"
    reasoning_prompt, reasoning_kwargs = llm.calls[0]
    assert reasoning_kwargs.get("format") == "json"
    results = [sd for sd in stream if sd.type == "result"]
    assert results and results[0].content == "all done"


@pytest.mark.asyncio
async def test_repair_reparse_round_trip():
    # First reasoning response is malformed (the classic "Expecting ',' delimiter"),
    # the repair call returns valid JSON.
    llm = ScriptedLLM([
        '{"thought": "broken" "action_type": "final_answer", "final_answer": "bad"}',
        '{"thought": "fixed", "action_type": "final_answer", "final_answer": "repaired answer"}',
    ])
    orchestrator = make_orchestrator(llm)

    stream = await run_to_completion(orchestrator)

    assert len(llm.calls) == 2
    repair_prompt, repair_kwargs = llm.calls[1]
    assert "failed to parse" in repair_prompt
    assert repair_kwargs.get("format") == "json"
    results = [sd for sd in stream if sd.type == "result"]
    assert results and results[0].content == "repaired answer"


@pytest.mark.asyncio
async def test_repair_failure_uses_fallback_without_crashing():
    # Both the reasoning response and the repair attempt are unusable prose;
    # the loop should fall back gracefully and still finish.
    llm = ScriptedLLM([
        "just some prose that is definitely not json",
        "still not json, sorry",
    ])
    orchestrator = make_orchestrator(llm)

    stream = await run_to_completion(orchestrator)

    assert len(llm.calls) == 2
    assert not any(sd.type == "error" for sd in stream)
    results = [sd for sd in stream if sd.type == "result"]
    assert results and "just some prose" in results[0].content
