"""Unit tests for the explicit remember-rule matcher."""

from __future__ import annotations

import pytest

from agents.remember_rule import extract_remember_fact

CASES = [
    ("Remember that my favourite colour is green.", "my favourite colour is green"),
    ("REMEMBER THAT x", "x"),
    ("remember: the wifi password is on the fridge", "the wifi password is on the fridge"),
    ("Remember this: Zach prefers texts", "Zach prefers texts"),
    ("please remember that I take my coffee black.", "I take my coffee black"),
    ("Hey Wits, remember that the truck is blue", "the truck is blue"),
    ("don't forget that the dentist is Tuesday", "the dentist is Tuesday"),
    ("Do not forget that rent is due on the 1st.", "rent is due on the 1st"),
    # A task, not a fact: goes down the normal path (no reminder feature here).
    ("remember to buy milk", None),
    ("remember my truck is blue", None),
    ("I remember that day", None),
    ("I can't recall who wrote Dune, can you look it up?", None),
    ("recall what I said yesterday", None),
    ("Do you remember what we discussed about the trip?", None),
    ("Can you remember that for me", None),
    ("Remember that my dog is called Rex?", None),
    ("The manager recalled the product, remembered?", None),
    ("remember that", None),
    ("remember that.", None),
    ("", None),
]


@pytest.mark.parametrize(("message", "expected"), CASES)
def test_extract_remember_fact(message, expected):
    assert extract_remember_fact(message) == expected
