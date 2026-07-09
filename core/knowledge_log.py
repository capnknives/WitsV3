"""Cross-session knowledge aggregation: recurring errors and durable project facts.

Separate from per-guest profiles (core/guest_user_profile.py) and from the
general conversational memory (core/memory_manager.py) — this store answers
"have we seen this before" across restarts, for two things that otherwise
reset every run:

  - Errors: tools/self_repair_tools.py's DiagnoseLogErrorsTool re-scans
    logs/witsv3.log fresh on every call and discards the result. This store
    lets the same scan accumulate occurrence counts over time instead.
  - Facts: durable, owner-confirmed facts about the project (not casual
    guest chat facts, and not raw conversational memory).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("WitsV3.KnowledgeLog")

DEFAULT_PATH = Path("var/data/knowledge_log.json")
MAX_FACT_LEN = 300


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_message(message: str) -> str:
    """Strip volatile bits (numbers, hex addresses, quoted values) so the same
    underlying error still hashes the same across occurrences."""
    text = (message or "").strip().lower()
    text = re.sub(r"0x[0-9a-f]+", "<hex>", text)
    text = re.sub(r"\d+", "<n>", text)
    text = re.sub(r"'[^']*'", "<val>", text)
    text = re.sub(r'"[^"]*"', "<val>", text)
    return text


def _error_signature(issue: dict[str, Any]) -> str:
    """Stable key for an issue dict shaped like parse_traceback_issues() output."""
    file_part = issue.get("file") or ""
    normalized = _normalize_message(issue.get("message", ""))
    raw = f"{file_part}|{normalized}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _empty_log() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "updated_at": _now_iso(),
        "errors": {},
        "facts": [],
    }


class KnowledgeLogStore:
    """JSON-file-backed aggregation of recurring errors and durable facts."""

    def __init__(
        self,
        path: Path | str | None = None,
        *,
        max_error_signatures: int = 200,
        max_facts: int = 200,
    ):
        self.path = Path(path) if path else DEFAULT_PATH
        self.max_error_signatures = max_error_signatures
        self.max_facts = max_facts
        self._lock = asyncio.Lock()

    def _load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return _empty_log()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Failed to load knowledge log from %s: %s", self.path, e)
            return _empty_log()
        data.setdefault("schema_version", 1)
        data.setdefault("errors", {})
        data.setdefault("facts", [])
        return data

    def _save(self, data: dict[str, Any]) -> None:
        data["updated_at"] = _now_iso()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    def _prune_errors(self, errors: dict[str, Any]) -> None:
        if len(errors) <= self.max_error_signatures:
            return
        # Rank most keep-worthy first: unresolved before resolved, then by
        # occurrence count (desc), then most recently seen first. Stable sort
        # lets us apply keys least-significant-first.
        ordered = sorted(errors.items(), key=lambda kv: kv[1].get("last_seen", ""), reverse=True)
        ordered.sort(key=lambda kv: -kv[1].get("occurrences", 0))
        ordered.sort(key=lambda kv: kv[1].get("resolved", False))
        for sig, _ in ordered[self.max_error_signatures :]:
            errors.pop(sig, None)

    def record_error_issues(self, issues: list[dict[str, Any]]) -> int:
        """Record/bump occurrence counts for a batch of issues (see
        tools/self_repair_tools.py:parse_traceback_issues for the input shape).
        Returns the number of distinct signatures touched."""
        if not issues:
            return 0
        data = self._load()
        errors: dict[str, Any] = data["errors"]
        now = _now_iso()
        touched = 0
        for issue in issues:
            sig = _error_signature(issue)
            entry = errors.get(sig)
            if entry is None:
                errors[sig] = {
                    "signature": (issue.get("message") or "")[:120],
                    "file": issue.get("file"),
                    "line": issue.get("line"),
                    "message": issue.get("message", ""),
                    "kind": issue.get("kind", "log_line"),
                    "first_seen": now,
                    "last_seen": now,
                    "occurrences": 1,
                    "resolved": False,
                    "resolved_at": None,
                }
            else:
                entry["occurrences"] = entry.get("occurrences", 0) + 1
                entry["last_seen"] = now
                entry["line"] = issue.get("line", entry.get("line"))
                # A fresh occurrence means it's back — clear any prior resolution.
                entry["resolved"] = False
                entry["resolved_at"] = None
            touched += 1
        self._prune_errors(errors)
        self._save(data)
        return touched

    def mark_error_resolved(self, issue: dict[str, Any]) -> bool:
        """Flag an issue's signature as resolved after a verified self-repair fix."""
        data = self._load()
        errors: dict[str, Any] = data["errors"]
        sig = _error_signature(issue)
        entry = errors.get(sig)
        if entry is None:
            return False
        entry["resolved"] = True
        entry["resolved_at"] = _now_iso()
        self._save(data)
        return True

    def add_fact(self, text: str, source: str, category: str = "project") -> bool:
        """Append a durable fact, deduped by exact text. Returns True if newly added."""
        text = (text or "").strip()[:MAX_FACT_LEN]
        if not text:
            return False
        data = self._load()
        facts: list[dict[str, Any]] = data["facts"]
        if any(f.get("text") == text for f in facts):
            return False
        facts.append({"text": text, "ts": _now_iso(), "source": source, "category": category})
        if len(facts) > self.max_facts:
            data["facts"] = facts[-self.max_facts :]
        self._save(data)
        return True

    @staticmethod
    def _cross_guest_patterns(
        guest_profile_summaries: list[dict[str, Any]] | None,
    ) -> list[tuple[str, int, int]]:
        """(label, guest_count, total_mentions) for interests shared by >=2 guests."""
        if not guest_profile_summaries:
            return []
        by_label: dict[str, dict[str, int]] = {}
        for summary in guest_profile_summaries:
            for interest in summary.get("top_interests") or []:
                label = interest.get("label")
                count = interest.get("count", 0)
                if not label:
                    continue
                stats = by_label.setdefault(label, {"guests": 0, "mentions": 0})
                stats["guests"] += 1
                stats["mentions"] += count
        shared = [
            (label, stats["guests"], stats["mentions"])
            for label, stats in by_label.items()
            if stats["guests"] >= 2
        ]
        shared.sort(key=lambda t: (-t[1], -t[2], t[0]))
        return shared

    def format_owner_summary(
        self, guest_profile_summaries: list[dict[str, Any]] | None = None
    ) -> str:
        data = self._load()
        errors: dict[str, Any] = data.get("errors", {})
        facts: list[dict[str, Any]] = data.get("facts", [])

        unresolved = [e for e in errors.values() if not e.get("resolved", False)]
        unresolved.sort(key=lambda e: -e.get("occurrences", 0))

        lines = ["Accumulated project knowledge:"]

        if unresolved:
            lines.append("\nRecurring issues (unresolved, most frequent first):")
            for e in unresolved[:10]:
                loc = f" ({e['file']}:{e['line']})" if e.get("file") else ""
                lines.append(f"  - [{e.get('occurrences', 1)}x] {e.get('message', '')[:160]}{loc}")
        else:
            lines.append("\nRecurring issues: none tracked yet.")

        if facts:
            lines.append("\nDurable facts:")
            for f in facts[-15:]:
                lines.append(f"  - {f.get('text', '')}")
        else:
            lines.append("\nDurable facts: none saved yet.")

        shared = self._cross_guest_patterns(guest_profile_summaries)
        if shared:
            lines.append("\nShared guest interests (mentioned by 2+ guests):")
            for label, guest_count, mentions in shared[:10]:
                lines.append(f"  - {label}: {guest_count} guests, {mentions} total mentions")

        return "\n".join(lines)
