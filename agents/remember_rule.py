"""The explicit "remember" rule: only a clear instruction at the start of a message.

A message is a remember request only when it opens with an explicit instruction such as
"remember that ...", "remember: ...", "remember this: ...", "don't forget that ..." or
"do not forget that ...", optionally after "please" or "hey wits". Questions and
in-passing mentions ("I can't recall...", "Do you remember...?") are not requests.
"""

from __future__ import annotations

import re

_FILLER = r"(?:(?:hey|hi|ok|okay)\s+wits[\s,:]*)?(?:please[\s,]+)?"
_TRIGGER = (
    r"(?:remember\s+this\s*:|remember\s*:|remember\s+that\b"
    r"|(?:don't|don’t|dont|do\s+not)\s+forget\s+that\b)"
)
_PATTERN = re.compile(rf"^\s*{_FILLER}{_TRIGGER}\s*(?P<fact>.+)$", re.IGNORECASE | re.DOTALL)
_QUESTION_START = re.compile(r"^\s*(?:do|can|did|could|would)\s+you\s+remember\b", re.IGNORECASE)


def extract_remember_fact(message: str) -> str | None:
    """Return the fact text of an explicit remember instruction, or None.

    Args:
        message: The raw user message.

    Returns:
        The text after the trigger phrase, stripped, without a trailing period; None when
        the message is not an explicit instruction or the fact would be empty.
    """
    text = (message or "").strip()
    if not text or text.endswith("?") or _QUESTION_START.match(text):
        return None
    match = _PATTERN.match(text)
    if not match:
        return None
    fact = match.group("fact").strip().rstrip(".").strip()
    return fact or None
