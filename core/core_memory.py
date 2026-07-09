"""Tiered memory: always-in-prompt core facts block (Letta-style core vs archival)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from core.runtime_paths import data_dir

logger = logging.getLogger(__name__)

_DEFAULT_MAX_TOKENS = 2048
_CHARS_PER_TOKEN = 4  # rough heuristic for local models


class CoreMemoryStore:
    """Structured core memory persisted to var/data/core_memory.json."""

    def __init__(self, path: Path | str | None = None, *, max_tokens: int = _DEFAULT_MAX_TOKENS):
        self.path = Path(path) if path else data_dir() / "core_memory.json"
        self.max_tokens = max_tokens
        self._data: dict[str, Any] = {
            "user_facts": [],
            "active_project": "",
            "preferences": [],
            "last_task_summary": "",
        }
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                self._data.update(loaded)
        except Exception as exc:
            logger.warning("Failed to load core memory from %s: %s", self.path, exc)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    def promote_fact(self, fact: str, *, source: str = "promotion") -> bool:
        text = (fact or "").strip()
        if len(text) < 8:
            return False
        facts: list[str] = self._data.setdefault("user_facts", [])
        lowered = text.lower()
        if any(existing.lower() == lowered for existing in facts):
            return False
        facts.append(text[:500])
        self._trim_lists()
        self._save()
        logger.debug("Promoted core fact (%s): %s", source, text[:80])
        return True

    def set_last_task_summary(self, summary: str) -> None:
        self._data["last_task_summary"] = (summary or "").strip()[:1000]
        self._save()

    def set_active_project(self, name: str) -> None:
        self._data["active_project"] = (name or "").strip()[:200]
        self._save()

    def add_preference(self, pref: str) -> bool:
        text = (pref or "").strip()
        if len(text) < 4:
            return False
        prefs: list[str] = self._data.setdefault("preferences", [])
        if text.lower() in (p.lower() for p in prefs):
            return False
        prefs.append(text[:300])
        self._trim_lists()
        self._save()
        return True

    def _trim_lists(self) -> None:
        max_facts = 40
        facts: list[str] = self._data.get("user_facts") or []
        if len(facts) > max_facts:
            self._data["user_facts"] = facts[-max_facts:]
        prefs: list[str] = self._data.get("preferences") or []
        if len(prefs) > 20:
            self._data["preferences"] = prefs[-20:]

    def as_prompt_block(self) -> str:
        """Render core memory for injection into orchestrator / WCCA context."""
        lines: list[str] = []
        project = (self._data.get("active_project") or "").strip()
        if project:
            lines.append(f"Active project: {project}")
        facts = self._data.get("user_facts") or []
        if facts:
            lines.append("Known facts:")
            for fact in facts[-15:]:
                lines.append(f"- {fact}")
        prefs = self._data.get("preferences") or []
        if prefs:
            lines.append("Preferences:")
            for pref in prefs[-8:]:
                lines.append(f"- {pref}")
        summary = (self._data.get("last_task_summary") or "").strip()
        if summary:
            lines.append(f"Last task context: {summary}")
        if not lines:
            return ""
        block = "\n".join(lines)
        max_chars = self.max_tokens * _CHARS_PER_TOKEN
        if len(block) > max_chars:
            return block[: max_chars - 1] + "…"
        return block

    def to_dict(self) -> dict[str, Any]:
        return dict(self._data)


def get_core_memory_store(config: Any | None = None) -> CoreMemoryStore:
    max_tokens = _DEFAULT_MAX_TOKENS
    if config is not None:
        mm = getattr(config, "memory_manager", None)
        if mm is not None:
            max_tokens = getattr(mm, "core_max_tokens", _DEFAULT_MAX_TOKENS)
    return CoreMemoryStore(max_tokens=max_tokens)
