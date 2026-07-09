"""Load guest content blocklists from config/guest_policy.yaml."""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_POLICY_PATH = Path(__file__).resolve().parent.parent / "config" / "guest_policy.yaml"


@lru_cache(maxsize=1)
def load_guest_policy(path: str | Path | None = None) -> dict[str, Any]:
    """Return parsed guest policy YAML (cached). Falls back to empty sections."""
    policy_path = Path(path) if path else _DEFAULT_POLICY_PATH
    if not policy_path.is_file():
        logger.debug("Guest policy file not found at %s; using built-in defaults", policy_path)
        return {}
    try:
        data = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.warning("Failed to load guest policy from %s: %s", policy_path, exc)
        return {}


def _compile_patterns(raw: list[str] | None) -> tuple[re.Pattern[str], ...]:
    compiled: list[re.Pattern[str]] = []
    for item in raw or []:
        try:
            compiled.append(re.compile(item, re.I))
        except re.error as exc:
            logger.warning("Skipping invalid guest policy pattern %r: %s", item, exc)
    return tuple(compiled)


def policy_terms(section: str, fallback: tuple[str, ...]) -> tuple[str, ...]:
    data = load_guest_policy()
    items = data.get(section)
    if not items:
        return fallback
    return tuple(str(item).lower() for item in items)


def policy_patterns(
    section: str, fallback: tuple[re.Pattern[str], ...]
) -> tuple[re.Pattern[str], ...]:
    data = load_guest_policy()
    items = data.get(section)
    if not items:
        return fallback
    return _compile_patterns(items)
