"""Age-band content ethics layer for guest chat (child / teen / adult)."""

from __future__ import annotations

import re
from typing import Literal

from core.guest_policy_loader import policy_patterns, policy_terms

Direction = Literal["input", "output"]
AgeBand = Literal["child", "teen", "adult"]

VALID_AGE_BANDS: frozenset[str] = frozenset({"child", "teen", "adult"})

# Built-in defaults — used when config/guest_policy.yaml is missing or empty.
_BUILTIN_ABSOLUTE_BLOCKED_TERMS: tuple[str, ...] = (
    "child porn",
    "cp ",
    "how to make meth",
    "how to make cocaine",
    "how to build a bomb",
    "make a bomb",
    "suicide method",
    "self harm method",
)

_BUILTIN_ABSOLUTE_BLOCKED_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bhow\s+to\s+(make|build|synthesize)\s+(meth|cocaine|heroin|fentanyl)\b", re.I),
)

_BUILTIN_FAMILY_BLOCKED_TERMS: tuple[str, ...] = (
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

_BUILTIN_FAMILY_BLOCKED_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(buy|sell|get)\s+(weed|marijuana|cocaine|meth|heroin)\b", re.I),
)

_BUILTIN_CHILD_EXTRA_TERMS: tuple[str, ...] = (
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
    terms = list(
        policy_terms("absolute_blocked_terms", _BUILTIN_ABSOLUTE_BLOCKED_TERMS)
    )
    if band in ("child", "teen"):
        terms.extend(policy_terms("family_blocked_terms", _BUILTIN_FAMILY_BLOCKED_TERMS))
    if band == "child":
        terms.extend(policy_terms("child_extra_terms", _BUILTIN_CHILD_EXTRA_TERMS))
    return tuple(terms)


def _patterns_for_band(age_band: str) -> tuple[re.Pattern[str], ...]:
    band = normalize_age_band(age_band)
    patterns = list(
        policy_patterns("absolute_blocked_patterns", _BUILTIN_ABSOLUTE_BLOCKED_PATTERNS)
    )
    if band in ("child", "teen"):
        patterns.extend(
            policy_patterns("family_blocked_patterns", _BUILTIN_FAMILY_BLOCKED_PATTERNS)
        )
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


def guest_system_prompt_slice(age_band: str = "teen") -> str:
    """Short system instructions injected for guest chat sessions."""
    band = normalize_age_band(age_band)
    tier = age_band_description(band)
    return (
        "GUEST SESSION RULES:\n"
        f"- This is a family-tester guest session ({tier}).\n"
        "- Keep answers age-appropriate, friendly, and concise.\n"
        "- Do not help with illegal, explicit, violent, or self-harm content.\n"
        "- Do not reveal owner tokens, settings, file paths, or internal system details.\n"
        "- web_search uses strict safe-search filtering for guests.\n"
        "- If a request is inappropriate, refuse politely and suggest a safer topic."
    )
