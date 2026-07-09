"""SKILL.md-style orchestrator playbooks — fixed tool sequences without ReAct."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

_PLAYBOOKS_DIR = Path(__file__).resolve().parents[1] / "config" / "playbooks"
_CACHE: dict[str, dict[str, Any]] | None = None


def _load_playbooks() -> dict[str, dict[str, Any]]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    playbooks: dict[str, dict[str, Any]] = {}
    if not _PLAYBOOKS_DIR.is_dir():
        _CACHE = playbooks
        return playbooks
    for path in sorted(_PLAYBOOKS_DIR.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        pid = data.get("id") or path.stem
        data["id"] = pid
        playbooks[pid] = data
    _CACHE = playbooks
    return playbooks


def reload_playbooks() -> None:
    """Clear cache (for tests)."""
    global _CACHE
    _CACHE = None


def list_playbooks() -> list[str]:
    return list(_load_playbooks().keys())


def get_playbook(playbook_id: str) -> dict[str, Any] | None:
    return _load_playbooks().get(playbook_id)


def _signals_match(message: str, signals: list[str]) -> bool:
    lowered = message.lower()
    return any(sig.lower() in lowered for sig in signals)


def match_playbook(message: str, *, doc_inventory: dict[str, int] | None = None) -> str | None:
    """Return playbook id when message matches trigger signals."""
    inventory = doc_inventory or {}
    for pid, spec in _load_playbooks().items():
        triggers = spec.get("triggers", {})
        signals = triggers.get("signals", [])
        if signals and _signals_match(message, signals):
            if triggers.get("requires_doc_inventory") and not inventory:
                continue
            return pid
        pattern = triggers.get("pattern")
        if pattern and re.search(pattern, message, re.IGNORECASE):
            if triggers.get("requires_doc_inventory") and not inventory:
                continue
            return pid
    return None


def extract_export_basename(message: str) -> str | None:
    """Pull export filename stem from save/export phrasing."""
    patterns = (
        r"(?:as|to|into)\s+([^\s?\"']+)",
        r"\bsave\s+(?:our|the|this)\s+conversation\s+as\s+([^\s?\"']+)",
        r"\bsave\s+a\s+copy(?:\s+of\s+our\s+conversation)?\s+as\s+([^\s?\"']+)",
    )
    for pat in patterns:
        m = re.search(pat, message, re.IGNORECASE)
        if m:
            name = m.group(1).strip().rstrip(".")
            if not name.endswith((".txt", ".md", ".log", ".json")):
                name = f"{name}.txt"
            return name
    return None
