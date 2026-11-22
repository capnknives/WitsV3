"""Per-turn timing record, appended as one JSON line per turn.

Why: later conversation-core packages need numbers to compare against the old path.
Only names, counts and times are written; never message, prompt, reply or tool text.
Nothing here may raise into a caller, so every public function swallows failures.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import time
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from core.runtime_paths import logs_dir

logger = logging.getLogger(__name__)

_NS_PER_MS = 1_000_000.0
_current: ContextVar[TurnMetrics | None] = ContextVar("turn_metrics_current", default=None)
_cached_settings: Any = None


@dataclass
class TurnMetrics:
    """One turn's record. Fields match the C1 contract."""

    ts: int
    turn_id: str
    session_id: str | None
    entry: str
    path: str
    input_chars: int
    first_event_ms: float | None = None
    first_result_ms: float | None = None
    first_token_ms: float | None = None
    wall_ms: float = 0.0
    llm_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    outcome: str = "ok"
    _t0: float = field(default=0.0, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        """Return the contract fields only (no private timing state)."""
        data = asdict(self)
        data.pop("_t0", None)
        return data


def _settings() -> Any:
    """Return the turn_metrics settings; defaults if config cannot be loaded."""
    global _cached_settings
    if _cached_settings is None:
        try:
            from core.config import TurnMetricsSettings, load_config

            try:
                _cached_settings = load_config().turn_metrics
            except Exception:
                _cached_settings = TurnMetricsSettings()
        except Exception:
            logger.debug("turn_metrics settings unavailable", exc_info=True)
            return None
    return _cached_settings


def _elapsed_ms(turn: TurnMetrics) -> float:
    return (time.perf_counter() - turn._t0) * 1000.0


def start_turn(
    entry: str, session_id: str | None, input_chars: int, path: str = "legacy"
) -> TurnMetrics:
    """Begin a turn and make it the current one for this task context."""
    turn = TurnMetrics(
        ts=int(time.time()),
        turn_id="trn_" + secrets.token_hex(8),
        session_id=session_id,
        entry=entry,
        path=path,
        input_chars=input_chars,
    )
    turn._t0 = time.perf_counter()
    _current.set(turn)
    return turn


def current_turn() -> TurnMetrics | None:
    """Return the active turn, or None outside a turn."""
    return _current.get()


def _ms(value: Any) -> float | None:
    """Convert an Ollama nanosecond duration to milliseconds."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return round(value / _NS_PER_MS, 3)
    return None


def _count(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def record_llm_call(endpoint: str, model: str, wall_ms: float, response: dict | None) -> None:
    """Add one model call; ``response`` is Ollama's final JSON object, if any."""
    try:
        turn = _current.get()
        if turn is None:
            return
        resp = response if isinstance(response, dict) else {}
        turn.llm_calls.append(
            {
                "endpoint": endpoint,
                "model": model,
                "wall_ms": round(wall_ms, 3),
                "load_ms": _ms(resp.get("load_duration")),
                "prompt_ms": _ms(resp.get("prompt_eval_duration")),
                "prompt_tokens": _count(resp.get("prompt_eval_count")),
                "eval_ms": _ms(resp.get("eval_duration")),
                "eval_tokens": _count(resp.get("eval_count")),
            }
        )
    except Exception:
        logger.debug("record_llm_call failed", exc_info=True)


def record_first_token() -> None:
    """Set first_token_ms once; later calls are ignored."""
    try:
        turn = _current.get()
        if turn is not None and turn.first_token_ms is None:
            turn.first_token_ms = round(_elapsed_ms(turn), 3)
    except Exception:
        logger.debug("record_first_token failed", exc_info=True)


def record_tool(name: str, ok: bool, ms: float) -> None:
    """Add one tool execution (name, success flag, duration only)."""
    try:
        turn = _current.get()
        if turn is not None:
            turn.tool_calls.append({"name": name, "ok": bool(ok), "ms": round(ms, 3)})
    except Exception:
        logger.debug("record_tool failed", exc_info=True)


def note_event(event_type: str) -> None:
    """Stamp first_event_ms on the first event and first_result_ms on result/error."""
    try:
        turn = _current.get()
        if turn is None:
            return
        elapsed = round(_elapsed_ms(turn), 3)
        if turn.first_event_ms is None:
            turn.first_event_ms = elapsed
        if event_type in ("result", "error") and turn.first_result_ms is None:
            turn.first_result_ms = elapsed
    except Exception:
        logger.debug("note_event failed", exc_info=True)


def _prune(directory: Path, retention_days: int) -> None:
    """Delete day files older than the retention window."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).strftime("%Y-%m-%d")
    for old in directory.glob("*.jsonl"):
        if old.stem < cutoff:
            old.unlink(missing_ok=True)


def _append(line: str, retention_days: int) -> None:
    directory = Path(logs_dir()) / "turn_metrics"
    directory.mkdir(parents=True, exist_ok=True)
    day_file = directory / (datetime.now(timezone.utc).strftime("%Y-%m-%d") + ".jsonl")
    is_new_day = not day_file.exists()
    with day_file.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    if is_new_day:
        _prune(directory, retention_days)


async def finish_turn(turn: TurnMetrics, outcome: str) -> None:
    """Close the turn, append its line (if enabled) and clear the context var."""
    try:
        turn.wall_ms = round(_elapsed_ms(turn), 3)
        turn.outcome = outcome
        _current.set(None)
        settings = _settings()
        if settings is None or not settings.enabled:
            return
        line = json.dumps(turn.to_dict(), ensure_ascii=False)
        await asyncio.to_thread(_append, line, settings.retention_days)
    except Exception:
        logger.debug("finish_turn failed", exc_info=True)
