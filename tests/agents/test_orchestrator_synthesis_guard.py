"""Tests for orchestrator final-answer synthesis guard."""

import logging

import pytest

from agents.base_orchestrator_agent import BaseOrchestratorAgent


class _SynthHarness(BaseOrchestratorAgent):
    def _build_reasoning_prompt(self, state):
        return ""

    def _parse_reasoning_response(self, response):
        return {}


def _harness() -> _SynthHarness:
    return _SynthHarness.__new__(_SynthHarness)


def _loop_harness() -> _SynthHarness:
    """Harness wired just enough to drive _execute_react_loop past the cap."""
    h = _SynthHarness.__new__(_SynthHarness)
    h.agent_name = "test-orchestrator"
    h.logger = logging.getLogger("test-orchestrator")
    h.max_iterations = 15
    h.current_iteration = 15  # already at the cap → loop body is skipped

    async def _noop_store(**kwargs):
        return None

    h.store_memory = _noop_store
    return h


@pytest.mark.asyncio
async def test_max_iterations_synthesizes_from_observations():
    """2026-07-08 finding: hitting max_iterations returned only a bare error,
    discarding usable observations. It must now salvage a grounded answer."""
    h = _loop_harness()
    state = {
        "completed": False,
        "goal": "Who died on June 14 2026?",
        "lookup_search_done": True,
        "observations": [
            "web_search results (base your answer on the SOURCES below):\n"
            "tavily summary (usually accurate — use it, but trust the sources "
            "below if any clearly contradicts it): Oliver Tree died June 14, 2026.\n"
            "[1] Example\n    snippet\n    source: https://example.com"
        ],
    }
    results = [sd async for sd in h._execute_react_loop(state, "sess")]
    assert state["completed"] is True
    assert any(sd.type == "result" and "Oliver Tree" in sd.content for sd in results)
    assert not any(sd.type == "error" for sd in results)


@pytest.mark.asyncio
async def test_max_iterations_without_observations_returns_clear_message():
    """With nothing usable to synthesize, the fallback must be an honest,
    actionable message rather than the old bare "maximum iterations" error."""
    h = _loop_harness()
    state = {"completed": False, "goal": "do something", "observations": []}
    results = [sd async for sd in h._execute_react_loop(state, "sess")]
    errors = [sd for sd in results if sd.type == "error"]
    assert errors
    assert "couldn't complete" in errors[0].content.lower()


def _doc_state():
    return {
        "goal": "Summarize the audit report",
        "observations": [
            "document_search results (base your answer on the EXCERPTS below):\n"
            "[1] audit.pdf (chunk 1) relevance=0.9\n"
            "    Revenue grew 12% year over year in Q4."
        ],
        "synthesis_guard_retries": 0,
    }


def test_blocks_denial_when_document_excerpts_exist():
    h = _harness()
    msg = h._validate_final_answer_synthesis(
        "I don't have access to your uploaded documents.",
        _doc_state(),
    )
    assert msg is not None
    assert "document_search" in msg


def test_allows_grounded_document_answer():
    h = _harness()
    msg = h._validate_final_answer_synthesis(
        "Revenue grew 12% year over year in Q4 according to audit.pdf.",
        _doc_state(),
    )
    assert msg is None


def test_resolve_final_answer_retries_once():
    h = _harness()
    state = _doc_state()
    answer, done = h._resolve_final_answer("I cannot access your files.", state)
    assert done is False
    assert state["synthesis_guard_retries"] == 1

    answer, done = h._resolve_final_answer("I cannot access your files.", state)
    assert done is True
    assert "audit.pdf" in answer or "Revenue" in answer


def test_auto_synthesize_from_web_summary():
    h = _harness()
    state = {
        "goal": "Who died on June 14 2026?",
        "lookup_search_done": True,
        "observations": [
            "web_search results (base your answer on the SOURCES below):\n"
            "tavily summary (usually accurate — use it, but trust the "
            "sources below if any clearly contradicts it): Oliver Tree died June 14, 2026.\n"
            "[1] Example\n    snippet\n    source: https://example.com"
        ],
    }
    text = h._auto_synthesize_from_observations(state)
    assert text is not None
    assert "Oliver Tree" in text


def test_blocks_confident_answer_when_no_document_passages():
    h = _harness()
    state = {
        "goal": "Summarize the audit report",
        "observations": [
            "document_search results (base your answer on the EXCERPTS below):\n"
            "(no matching passages — try a broader query before giving up)"
        ],
        "synthesis_guard_retries": 0,
    }
    msg = h._validate_final_answer_synthesis(
        "The audit shows revenue grew 40% and margins improved sharply.",
        state,
    )
    assert msg is not None
    assert "no strong excerpts" in msg


def test_auto_synthesize_returns_insufficient_evidence_message():
    h = _harness()
    state = {
        "goal": "Summarize the audit report",
        "observations": [
            "document_search results (base your answer on the EXCERPTS below):\n"
            "(no matching passages — try a broader query before giving up)"
        ],
    }
    text = h._auto_synthesize_from_observations(state)
    assert text is not None
    assert "didn't find passages" in text


def _codebase_state():
    return {
        "goal": "What can you tell me about your codebase wits?",
        "observations": [
            "Tool read_file result: # WitsV3\n\nA local-first LLM orchestration system. "
            "Talk to it in a browser; it plans with a ReAct loop and calls real tools."
        ],
        "synthesis_guard_retries": 0,
    }


def test_blocks_witwatersrand_hallucination_on_codebase_intro():
    h = _harness()
    msg = h._validate_final_answer_synthesis(
        "Wits University, also known as the University of the Witwatersrand, uses GitHub.",
        _codebase_state(),
    )
    assert msg is not None
    assert "external organization" in msg or "WitsV3" in msg


def test_allows_grounded_codebase_answer():
    h = _harness()
    msg = h._validate_final_answer_synthesis(
        "WitsV3 is a local-first LLM orchestration system using Ollama and a ReAct orchestrator.",
        _codebase_state(),
    )
    assert msg is None


def test_auto_synthesize_from_codebase_read_file():
    h = _harness()
    state = _codebase_state()
    text = h._auto_synthesize_from_observations(state)
    assert text is not None
    assert "WitsV3" in text or "local-first" in text
