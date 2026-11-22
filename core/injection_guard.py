"""Lightweight prompt-injection checks for tool arguments (owner path)."""

from __future__ import annotations

import re
from typing import Any

_INJECTION_PATTERNS = (
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", re.I),
    re.compile(r"disregard\s+(your|the)\s+(system|safety|rules)", re.I),
    re.compile(r"you\s+are\s+now\s+(in\s+)?(developer|admin|root)\s+mode", re.I),
    # Reason: the slash and its trailing spaces form one optional group; two adjacent \s* here
    # backtrack quadratically on "<" followed by many spaces.
    re.compile(r"<\s*(?:/\s*)?system\s*>", re.I),
    re.compile(r"BEGIN\s+SYSTEM\s+PROMPT", re.I),
)

_FILE_WRITE_TOOLS = frozenset({"write_file", "apply_code_fix"})


def check_tool_injection(tool_name: str, tool_args: dict[str, Any]) -> str | None:
    """Return a block reason if tool_args look like an injection attempt."""
    if tool_name not in _FILE_WRITE_TOOLS:
        return None

    blobs: list[str] = []
    for key in ("new_content", "content", "file_path", "reason"):
        val = tool_args.get(key)
        if isinstance(val, str):
            blobs.append(val)

    combined = "\n".join(blobs)
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(combined):
            return (
                f"Blocked {tool_name}: tool arguments match a prompt-injection pattern "
                f"({pattern.pattern[:40]}…)."
            )
    return None


def scan_text(text: str) -> list[str]:
    """Return the patterns that match ``text``, for flagging stored text (never for blocking).

    Workspace text from network agents is stored even when it matches (contract section 8): a
    pattern list can warn but not protect, so a hit only sets ``flagged``.

    Args:
        text: Already-cleaned text to scan.

    Returns:
        The source of each matching pattern, in pattern order; empty when nothing matches or the
        value is not a string.
    """
    if not isinstance(text, str):
        return []
    return [p.pattern for p in _INJECTION_PATTERNS if p.search(text)]
