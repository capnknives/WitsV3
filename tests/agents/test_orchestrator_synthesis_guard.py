"""Tests for orchestrator final-answer synthesis guard."""

from agents.base_orchestrator_agent import BaseOrchestratorAgent


class _SynthHarness(BaseOrchestratorAgent):
    def _build_reasoning_prompt(self, state):
        return ""

    def _parse_reasoning_response(self, response):
        return {}


def _harness() -> _SynthHarness:
    return _SynthHarness.__new__(_SynthHarness)


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
