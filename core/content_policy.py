"""Family-friendly content policy for guest chat (input/output preflight)."""

from __future__ import annotations

import re
from typing import Literal

Direction = Literal["input", "output"]

# Teen-oriented blocklist — substring match on normalized lowercase text.
_BLOCKED_TERMS: tuple[str, ...] = (
    "porn",
    "pornography",
    "xxx",
    "hentai",
    "nude",
    "nudes",
    "onlyfans",
    "strip club",
    "how to make meth",
    "how to make cocaine",
    "buy drugs",
    "sell drugs",
    "self harm",
    "kill myself",
    "suicide method",
    "how to build a bomb",
    "make a bomb",
    "child porn",
    "cp ",
)

_BLOCKED_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bhow\s+to\s+(make|build|synthesize)\s+(meth|cocaine|heroin|fentanyl)\b", re.I),
    re.compile(r"\b(buy|sell|get)\s+(weed|marijuana|cocaine|meth|heroin)\b", re.I),
)


def check_guest_content(
    text: str,
    *,
    direction: Direction = "input",
    age_band: str = "teen",
) -> tuple[bool, str | None]:
    """Return (allowed, refusal_message). allowed=False means block the turn."""
    if not text or not text.strip():
        return True, None

    normalized = " ".join(text.lower().split())

    for term in _BLOCKED_TERMS:
        if term in normalized:
            return False, _refusal_message(direction, age_band)

    for pattern in _BLOCKED_PATTERNS:
        if pattern.search(normalized):
            return False, _refusal_message(direction, age_band)

    return True, None


def _refusal_message(direction: Direction, age_band: str) -> str:
    if direction == "input":
        return (
            "I can't help with that request. This guest session is limited to "
            "family-friendly topics. Try asking about homework, hobbies, games, "
            "or general knowledge instead."
        )
    return (
        "I need to keep my answer family-friendly, so I can't share that. "
        "Ask me something else!"
    )
