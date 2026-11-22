"""Tiny key=value config loader used as a target for the golden repair cases."""

from __future__ import annotations


def load_settings(text: str) -> dict[str, str]:
    """Parse ``key=value`` lines, skipping blanks and ``#`` comments."""
    settings: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        settings[key.strip()] = value.strip()
    return settings
