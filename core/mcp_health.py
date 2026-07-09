"""In-process MCP server health snapshots for the web UI (Phase 2.2)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

_server_health: dict[str, dict[str, Any]] = {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def record_connect_success(name: str) -> None:
    """Record a successful MCP server connection."""
    entry = _server_health.setdefault(name, {})
    entry.update(
        {
            "connected": True,
            "last_error": None,
            "last_connect_at": _now_iso(),
            "last_success_at": _now_iso(),
        }
    )


def record_connect_failure(name: str, error: str) -> None:
    """Record a failed MCP connect attempt."""
    entry = _server_health.setdefault(name, {})
    entry.update(
        {
            "connected": False,
            "last_error": error,
            "last_connect_at": _now_iso(),
        }
    )


def record_disconnect(name: str) -> None:
    """Mark a server as disconnected (preserves last error for diagnostics)."""
    entry = _server_health.setdefault(name, {})
    entry["connected"] = False
    entry["last_disconnect_at"] = _now_iso()


def get_server_health(name: str) -> dict[str, Any]:
    """Health snapshot for one configured server."""
    return dict(_server_health.get(name, {}))


def all_server_health() -> dict[str, dict[str, Any]]:
    """Copy of all tracked server health entries."""
    return {k: dict(v) for k, v in _server_health.items()}


def clear_server_health(name: str | None = None) -> None:
    """Clear health for one server or all (mainly for tests)."""
    if name is None:
        _server_health.clear()
    else:
        _server_health.pop(name, None)
