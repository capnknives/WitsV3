"""Unit tests for agents/routing_classifier.py — deterministic routing layer."""

import pytest

from agents.routing_classifier import (
    RoutingContext,
    classify_message,
    is_pure_greeting,
    needs_self_repair,
    needs_story_writing,
    requires_orchestrator,
)


def test_pure_greeting_matches_explicit_greetings():
    assert is_pure_greeting("hi") is True
    assert is_pure_greeting("thanks!") is True
    assert is_pure_greeting("how are you today?") is True


def test_pure_greeting_rejects_length_only_casual():
    assert is_pure_greeting("Run self repair") is False
    assert (
        is_pure_greeting("i've updated things, please check the results once more") is False
    )


def test_self_repair_signals():
    assert needs_self_repair("run self repair") is True
    assert needs_self_repair("find and fix any bugs in your code") is True
    assert needs_self_repair("thanks!") is False


def test_story_writing_signals():
    assert needs_story_writing("write a 100 page story about a knight") is True
    assert needs_story_writing("what's the story with this bug?") is False


def test_orchestrator_for_web_search():
    assert requires_orchestrator("search the web for ollama news", {}) is True


def test_orchestrator_for_documents():
    inventory = {"Pleistocene_Megafauna_Audit_Report.md": 3}
    assert requires_orchestrator("summarize the audit report", inventory) is True


def test_classify_self_repair_before_greeting():
    ctx = RoutingContext(message="run self repair", user_role="owner")
    decision = classify_message(ctx)
    assert decision.destination == "self_repair"


def test_classify_greeting():
    ctx = RoutingContext(message="thanks!", user_role="owner")
    decision = classify_message(ctx)
    assert decision.destination == "greeting"
    intent = decision.to_intent()
    assert intent is not None
    assert intent["type"] == "conversation"


def test_classify_ambiguous_needs_intent():
    ctx = RoutingContext(
        message="explain the difference between asyncio and threading",
        user_role="owner",
    )
    decision = classify_message(ctx)
    assert decision.destination == "needs_intent"
    assert decision.to_intent() is None


def test_classify_story_to_book_agent():
    ctx = RoutingContext(
        message="write a short story about space exploration",
        user_role="owner",
    )
    decision = classify_message(ctx)
    assert decision.destination == "book_writing"
    intent = decision.to_intent()
    assert intent["specialized_agent"] == "book_writing"


@pytest.mark.parametrize(
    "message",
    [
        "run self repair",
        "find and fix any bugs in your code",
        "repair yourself",
    ],
)
def test_imperative_commands_not_greetings(message):
    assert is_pure_greeting(message) is False
    ctx = RoutingContext(message=message, user_role="owner")
    assert classify_message(ctx).destination == "self_repair"
