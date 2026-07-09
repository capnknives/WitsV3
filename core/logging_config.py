"""Apply logging level from WitsV3Config (Config Wave 3)."""

from __future__ import annotations

import logging


def apply_logging_level(level_name: str) -> None:
    """Set root logger level from config.yaml `logging_level`."""
    level = getattr(logging, str(level_name or "INFO").upper(), logging.INFO)
    logging.getLogger().setLevel(level)
