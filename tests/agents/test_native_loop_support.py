"""Pure helpers of the native loop: history window, assistant tool message."""

from agents.native_loop_support import assistant_tool_message, history_window


def _history(n: int) -> list[dict]:
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(n)]


def test_keeps_everything_until_twice_the_window():
    hist = _history(10)
    assert history_window(hist, 5) == hist


def test_cut_leaves_window_and_first_message_holds_between_cuts():
    h = 5
    firsts = {n: history_window(_history(n), h)[0]["content"] for n in range(11, 24)}
    assert len(history_window(_history(11), h)) == h
    # Between cuts (kept part 5..10) the first kept message never changes.
    assert {firsts[n] for n in range(11, 17)} == {"m6"}
    # The next cut happens when the kept part would exceed 2 * h.
    assert firsts[17] == "m12" and len(history_window(_history(17), h)) == h
    assert {firsts[n] for n in range(17, 23)} == {"m12"}


def test_window_never_starts_on_a_tool_reply():
    hist = (
        _history(5)
        + [
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "t"}}]},
            {"role": "tool", "content": "r1"},
            {"role": "tool", "content": "r2"},
            {"role": "assistant", "content": "done"},
        ]
        + _history(2)
    )
    # 11 messages, h=5: the plain cut would start at index 6 (a tool reply).
    window = history_window(hist, 5)
    assert window[0]["role"] == "assistant" and window[0].get("tool_calls")
    assert window == hist[5:]


def test_assistant_message_uses_ollama_shape_and_objects():
    msg = assistant_tool_message(
        "", [{"name": "a", "arguments": '{"k": 1}'}, {"name": "b", "arguments": "{bad"}]
    )
    assert msg.role == "assistant"
    assert msg.tool_calls == [
        {"function": {"name": "a", "arguments": {"k": 1}}},
        {"function": {"name": "b", "arguments": {}}},
    ]
