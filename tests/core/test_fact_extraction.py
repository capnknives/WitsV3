"""Tests for heuristic owner-path fact promotion."""

from core.fact_extraction import extract_promotable_facts


def test_extracts_preference_from_user_message():
    facts = extract_promotable_facts("I prefer dark mode in every app", "Sure, noted.")
    assert len(facts) == 1
    assert "dark mode" in facts[0].lower()


def test_skips_explicit_remember_phrasing():
    facts = extract_promotable_facts("Remember that my dog is named Max", "Got it.")
    assert facts == []


def test_skips_casual_messages_without_preference_signal():
    assert extract_promotable_facts("What is the weather?", "Sunny.") == []
