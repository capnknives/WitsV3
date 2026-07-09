"""Lightweight in-process tool usage analytics (Phase 2.3)."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock
from typing import Any


@dataclass
class _ToolStats:
    calls: int = 0
    failures: int = 0
    total_ms: float = 0.0
    last_called_at: float | None = None
    last_error: str | None = None


class ToolMetricsRecorder:
    """Thread-safe recorder for tool call latency and failure counts."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._stats: dict[str, _ToolStats] = defaultdict(_ToolStats)

    def record(self, tool_name: str, elapsed_ms: float, *, success: bool, error: str | None = None) -> None:
        with self._lock:
            stats = self._stats[tool_name]
            stats.calls += 1
            stats.total_ms += elapsed_ms
            stats.last_called_at = time.time()
            if not success:
                stats.failures += 1
                stats.last_error = error

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            rows: list[dict[str, Any]] = []
            for name, stats in sorted(self._stats.items()):
                avg_ms = (stats.total_ms / stats.calls) if stats.calls else 0.0
                rows.append(
                    {
                        "tool": name,
                        "calls": stats.calls,
                        "failures": stats.failures,
                        "avg_latency_ms": round(avg_ms, 2),
                        "last_error": stats.last_error,
                        "last_called_at": stats.last_called_at,
                    }
                )
            return rows

    def clear(self) -> None:
        with self._lock:
            self._stats.clear()


# Process-wide singleton used by ToolRegistry and /api/metrics/tools.
tool_metrics = ToolMetricsRecorder()
