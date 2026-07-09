"""Optional smoke-test metrics (gated by WITS_SMOKE_METRICS=1)."""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


def smoke_metrics_enabled() -> bool:
    return os.environ.get("WITS_SMOKE_METRICS", "").strip().lower() in ("1", "true", "yes")


@dataclass
class SmokeMetrics:
    """Per-scenario counters collected during a smoke run."""

    scenario_id: str = ""
    route: str = ""
    llm_calls: int = 0
    react_iterations: int = 0
    tools_called: list[str] = field(default_factory=list)
    wall_ms: float = 0.0
    passed: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def record_llm_call(self) -> None:
        self.llm_calls += 1

    def record_react_iteration(self) -> None:
        self.react_iterations += 1

    def record_tool(self, name: str) -> None:
        if name and name not in self.tools_called:
            self.tools_called.append(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "route": self.route,
            "llm_calls": self.llm_calls,
            "react_iterations": self.react_iterations,
            "tools_called": list(self.tools_called),
            "wall_ms": round(self.wall_ms, 1),
            "tokens_estimated": None,
            "passed": self.passed,
            **self.extra,
        }


_current: SmokeMetrics | None = None
_report: list[dict[str, Any]] = []


def get_current() -> SmokeMetrics | None:
    return _current


def reset_report() -> None:
    global _report
    _report = []


def get_report() -> list[dict[str, Any]]:
    return list(_report)


def append_report(entry: dict[str, Any]) -> None:
    _report.append(entry)


@contextmanager
def scenario_metrics(scenario_id: str, route: str = ""):
    """Context manager for per-scenario timing and counters."""
    global _current
    if not smoke_metrics_enabled():
        yield None
        return

    metrics = SmokeMetrics(scenario_id=scenario_id, route=route)
    _current = metrics
    start = time.perf_counter()
    try:
        yield metrics
    finally:
        metrics.wall_ms = (time.perf_counter() - start) * 1000
        append_report(metrics.to_dict())
        _current = None


def dump_report_json() -> str:
    return json.dumps(get_report(), indent=2)
