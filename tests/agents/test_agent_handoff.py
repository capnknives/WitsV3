"""Agent hand-off graph tests."""

from agents.agent_handoff import handoff_stream_note, resolve_handoff


def test_coding_handoff_has_fallback():
    primary, fallbacks = resolve_handoff("coding")
    assert primary == "AdvancedCodingAgent"
    assert "LLMDrivenOrchestrator" in fallbacks


def test_handoff_stream_note():
    note = handoff_stream_note("coding")
    assert "AdvancedCodingAgent" in note
