"""Pure helpers for the golden runner's failure files (no I/O beyond the files named)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

FAILURES_DIRNAME = "failures"
REASON_LIMIT = 80


def reply_text(events: list[Any]) -> str:
    """Join the text of every result and error event, in order, with newlines."""
    return "\n".join(str(e.content) for e in events if e.type in ("result", "error"))


def eval_tokens_of(turn: Any) -> int | None:
    """Sum eval_tokens over a turn record's llm_calls; None when there is no record."""
    if turn is None:
        return None
    return sum(c.get("eval_tokens") or 0 for c in turn.llm_calls)


def shorten_reason(reason: str, limit: int = REASON_LIMIT) -> str:
    """Make a failure string safe for a one-line table cell, cut to limit characters."""
    text = " ".join(reason.split("\n")).replace("\r", " ").replace("|", "/")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def build_record(
    case_id: str,
    run: int,
    message: str,
    failures: list[str],
    events: list[Any],
    tools: list[str],
    first_result_ms: float | None,
    eval_tokens: int | None,
) -> dict[str, Any]:
    """Build the failure record for one failed run (exact key set is a contract)."""
    reply = reply_text(events)
    return {
        "case_id": case_id,
        "run": run,
        "message": message,
        "failures": list(failures),
        "reply": reply,
        "reply_chars": len(reply),
        "tools_seen": list(tools),
        "first_result_ms": first_result_ms,
        "eval_tokens": eval_tokens,
    }


def clear_failures(sandbox: Path) -> Path:
    """Create DIR/failures and delete only *.run*.json files in it; return the folder."""
    folder = sandbox / FAILURES_DIRNAME
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.run*.json"):
        if old.is_file():
            old.unlink()
    return folder


def write_record(folder: Path, record: dict[str, Any]) -> Path:
    """Write one record as <case-id>.run<N>.json (UTF-8, indent 2); return the path."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{record['case_id']}.run{record['run']}.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
