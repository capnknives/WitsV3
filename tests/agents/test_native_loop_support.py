"""Pure helpers of the native loop: history window, arguments, assistant message."""

import pytest

from agents.native_loop_support import (
    assistant_tool_message,
    decode_arguments,
    history_window,
)


def _history(n: int) -> list[dict]:
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(n)]


def test_keeps_everything_until_twice_the_window():
    hist = _history(10)
    assert history_window(hist, 5) == hist


@pytest.mark.parametrize("h", [4, 5, 10])
def test_cut_starts_on_user_and_first_message_holds_between_cuts(h):
    previous_first, changes = None, 0
    for n in range(2, 8 * h, 2):  # Reason: each completed turn adds a user/assistant pair.
        window = history_window(_history(n), h)
        assert window[0]["role"] == "user"
        assert len(window) <= 2 * h
        if n > 2 * h:
            assert len(window) >= h - 1
        if window[0]["content"] != previous_first:
            changes += 1
            previous_first = window[0]["content"]
    # Reason: the start moves only at cuts (roughly every h + 1 messages), not every turn.
    turns = len(range(2, 8 * h, 2))
    assert 1 < changes < turns / 2


def test_history_turns_one_keeps_a_full_exchange():
    for n in range(2, 30, 2):
        window = history_window(_history(n), 1)
        assert window[0]["role"] == "user"
        assert [m["role"] for m in window[:2]] == ["user", "assistant"]


def test_window_never_splits_a_tool_group():
    group = [
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "t"}}]},
        {"role": "tool", "content": "r1"},
        {"role": "tool", "content": "r2"},
        {"role": "assistant", "content": "done"},
    ]
    hist = _history(5) + group + _history(2)
    # 11 messages, h=5: the block cut would start at index 6, a tool reply.
    window = history_window(hist, 5)
    assert window[0]["role"] == "user"
    assert window == hist[9:]
    assert not any(m["role"] == "tool" for m in window)


def test_decode_arguments_never_coerces_non_objects():
    assert decode_arguments("") == {}
    assert decode_arguments('{"k": 1}') == {"k": 1}
    assert decode_arguments("[1, 2]") == [1, 2]
    assert decode_arguments("null") is None
    assert decode_arguments("3") == 3
    assert decode_arguments(None) is None
    with pytest.raises(ValueError):
        decode_arguments("{bad")


def test_assistant_message_uses_ollama_shape_and_objects():
    msg = assistant_tool_message(
        "", [{"name": "a", "arguments": '{"k": 1}'}, {"name": "b", "arguments": "{bad"}]
    )
    assert msg.role == "assistant"
    assert msg.tool_calls == [
        {"function": {"name": "a", "arguments": {"k": 1}}},
        {"function": {"name": "b", "arguments": {}}},
    ]


def test_token_budget_cuts_on_user_and_cut_holds_next_turn():
    big = [{"role": r, "content": "x" * 400} for r in ["user", "assistant"] * 3]
    # Six 100-token messages fit a 700 budget; the next exchange does not.
    assert history_window(big, 5, token_budget=700) == big
    grown = big + [{"role": "user", "content": "x" * 400}, {"role": "assistant", "content": "y"}]
    window = history_window(grown, 5, token_budget=700)
    assert window[0]["role"] == "user" and len(window) < len(grown)
    # Reason: without the budget 8 messages are under 2 * h, so this cut is the token one.
    assert history_window(grown, 5) == grown
    small = [{"role": "user", "content": "z"}, {"role": "assistant", "content": "w"}]
    assert history_window(grown + small, 5, token_budget=700)[0] is window[0]


def test_no_budget_matches_count_rule():
    hist = _history(23)
    assert history_window(hist, 5, token_budget=None) == history_window(hist, 5)
