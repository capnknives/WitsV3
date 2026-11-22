"""Tiny date parser used as a target for the golden repair cases."""

from __future__ import annotations

from datetime import date


def parse_date(text: str) -> date:
    """Parse an ISO ``YYYY-MM-DD`` string into a date."""
    year, month, day = (int(part) for part in text.strip().split("-"))
    return date(year, month, day)
