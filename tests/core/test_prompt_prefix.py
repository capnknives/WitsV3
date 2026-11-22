"""Tests for core.prompt_prefix."""

import random

from core.llm_interface import LLMMessage
from core.prompt_prefix import (
    CONTEXT_HEADING,
    build_messages,
    order_tools,
    prefix_fingerprint,
)

SYSTEM = "You are WitsV3."
CATALOG = "Skills: a, b"
TOOLS = [
    {"type": "function", "function": {"name": n, "parameters": {}}}
    for n in ("zeta", "alpha", "mid")
]


def _history_from(messages, reply):
    """Fold a sent turn plus the assistant reply into history, as first sent."""
    return list(messages[1:]) + [LLMMessage(role="assistant", content=reply)]


def test_prefix_stable_across_three_turns():
    contexts = [{"Time": "09:00"}, {"Time": "09:05", "Memory": "likes tea"}, None]
    history: list[LLMMessage] = []
    turns = []
    for i, ctx in enumerate(contexts):
        msgs = build_messages(SYSTEM, CATALOG, history, f"question {i}", ctx)
        turns.append(msgs)
        history = _history_from(msgs, f"answer {i}")
    for a, b in zip(turns, turns[1:], strict=False):
        # The whole of turn N, up to its last message, is the start of turn N+1.
        span = len(a) - 1
        fp_a = prefix_fingerprint(a, TOOLS)
        fp_b = prefix_fingerprint(b[: span + 1], TOOLS)
        assert fp_a == fp_b
        assert b[:span] == a[:span]


def test_context_only_in_last_message():
    msgs = build_messages(SYSTEM, CATALOG, [], "hi", {"Time": "09:00", "Docs": "x.md"})
    assert all(CONTEXT_HEADING not in m.content for m in msgs[:-1])
    assert msgs[-1].content.startswith("hi")
    assert CONTEXT_HEADING in msgs[-1].content
    assert "Time: 09:00" in msgs[-1].content


def test_none_context_leaves_user_text_unchanged():
    msgs = build_messages(SYSTEM, CATALOG, [{"role": "user", "content": "old"}], "hi", None)
    assert msgs[-1] == LLMMessage(role="user", content="hi")
    assert msgs[0].content == f"{SYSTEM}\n\n{CATALOG}"
    assert msgs[1].content == "old"


def test_order_tools_stable_under_shuffle():
    expected = order_tools(TOOLS)
    assert [t["function"]["name"] for t in expected] == ["alpha", "mid", "zeta"]
    for seed in range(5):
        shuffled = TOOLS[:]
        random.Random(seed).shuffle(shuffled)
        assert order_tools(shuffled) == expected
        assert prefix_fingerprint(
            [LLMMessage("system", "s"), LLMMessage("user", "u")], shuffled
        ) == prefix_fingerprint([LLMMessage("system", "s"), LLMMessage("user", "u")], TOOLS)
