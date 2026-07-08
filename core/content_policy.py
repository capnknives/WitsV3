"""Age-band content ethics layer for guest chat (child / teen / adult)."""

from __future__ import annotations

import re
from typing import Literal

Direction = Literal["input", "output"]
AgeBand = Literal["child", "teen", "adult"]

VALID_AGE_BANDS: frozenset[str] = frozenset({"child", "teen", "adult"})

# Shared illegal / harm blocklist (all tiers).
_ABSOLUTE_BLOCKED_TERMS: tuple[str, ...] = (
    "child porn",
    "cp ",
    "how to make meth",
    "how to make cocaine",
    "how to build a bomb",
    "make a bomb",
    "suicide method",
    "self harm method",
)

_ABSOLUTE_BLOCKED_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bhow\s+to\s+(make|build|synthesize)\s+(meth|cocaine|heroin|fentanyl)\b", re.I),
)

# Teen + child: family-friendly (no explicit adult content, drugs, etc.).
_FAMILY_BLOCKED_TERMS: tuple[str, ...] = (
    "porn",
    "pornography",
    "xxx",
    "hentai",
    "nude",
    "nudes",
    "onlyfans",
    "strip club",
    "buy drugs",
    "sell drugs",
    "self harm",
    "kill myself",
)

_FAMILY_BLOCKED_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(buy|sell|get)\s+(weed|marijuana|cocaine|meth|heroin)\b", re.I),
)

# Child: strictest — extra topics teens might discuss in supervised contexts.
_CHILD_EXTRA_TERMS: tuple[str, ...] = (
    "alcohol",
    "beer",
    "wine",
    "vodka",
    "whiskey",
    "cigarette",
    "cigarettes",
    "vape",
    "vaping",
    "hookup",
    "tinder",
    "dating app",
    "gore",
    "how to kill",
)


def normalize_age_band(value: str | None, *, default: str = "teen") -> str:
    """Map owner/guest age band strings to child | teen | adult."""
    if not value:
        return default if default in VALID_AGE_BANDS else "teen"
    lowered = value.strip().lower()
    if lowered in ("child", "kid", "kids", "minor"):
        return "child"
    if lowered in ("teen", "teenager", "youth"):
        return "teen"
    if lowered in ("adult", "grownup", "grown-up"):
        return "adult"
    return default if default in VALID_AGE_BANDS else "teen"


def _terms_for_band(age_band: str) -> tuple[str, ...]:
    band = normalize_age_band(age_band)
    terms = list(_ABSOLUTE_BLOCKED_TERMS)
    if band in ("child", "teen"):
        terms.extend(_FAMILY_BLOCKED_TERMS)
    if band == "child":
        terms.extend(_CHILD_EXTRA_TERMS)
    return tuple(terms)


def _patterns_for_band(age_band: str) -> tuple[re.Pattern[str], ...]:
    band = normalize_age_band(age_band)
    patterns = list(_ABSOLUTE_BLOCKED_PATTERNS)
    if band in ("child", "teen"):
        patterns.extend(_FAMILY_BLOCKED_PATTERNS)
    return tuple(patterns)


def check_guest_content(
    text: str,
    *,
    direction: Direction = "input",
    age_band: str = "teen",
) -> tuple[bool, str | None]:
    """Return (allowed, refusal_message). allowed=False means block the turn."""
    if not text or not text.strip():
        return True, None

    band = normalize_age_band(age_band)
    normalized = " ".join(text.lower().split())

    for term in _terms_for_band(band):
        if term in normalized:
            return False, _refusal_message(direction, band)

    for pattern in _patterns_for_band(band):
        if pattern.search(normalized):
            return False, _refusal_message(direction, band)

    return True, None


def _refusal_message(direction: Direction, age_band: str) -> str:
    band = normalize_age_band(age_band)
    if band == "child":
        hint = "Try asking about school, games, hobbies, or fun facts instead."
    elif band == "teen":
        hint = "Try asking about homework, hobbies, games, or general knowledge instead."
    else:
        hint = "Try rephrasing your question."

    if direction == "input":
        return (
            "I can't help with that request. This guest session has content limits "
            f"for {band} users. {hint}"
        )
    return (
        "I need to keep my answer appropriate for this guest session. "
        "Ask me something else!"
    )


def age_band_description(age_band: str) -> str:
    band = normalize_age_band(age_band)
    return {
        "child": "strict family-safe filter (child)",
        "teen": "family-friendly filter (teen)",
        "adult": "standard safety filter (adult — owner-assigned only)",
    }[band]
