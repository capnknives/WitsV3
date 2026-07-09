"""Heuristic fact extraction for owner-path knowledge log promotion."""

from __future__ import annotations

import re

_PREFERENCE_RE = re.compile(
    r"\b(?:i always|i prefer|my name is|call me|remember that|don't forget)\b",
    re.I,
)
_FACT_LINE_RE = re.compile(
    r"^(?:User prefers|User name is|Remember:)\s*(.+)$",
    re.I,
)


def extract_promotable_facts(user_message: str, assistant_message: str) -> list[str]:
    """Return short fact strings worth promoting to the knowledge log."""
    facts: list[str] = []
    user = (user_message or "").strip()
    if not user or not _PREFERENCE_RE.search(user):
        return facts

    # Explicit remember phrasing is handled by WCCA; skip duplicate promotion.
    if any(kw in user.lower() for kw in ("remember", "don't forget", "recall")):
        return facts

    snippet = user[:280].strip()
    if len(snippet) >= 12:
        facts.append(snippet)
    return facts[:2]
