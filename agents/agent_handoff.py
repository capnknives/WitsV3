"""Explicit agent hand-off graph (lightweight state machine)."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_GRAPH_PATH = Path(__file__).resolve().parent.parent / "config" / "agent_graph.yaml"


@lru_cache(maxsize=1)
def load_agent_graph(path: str | Path | None = None) -> dict[str, Any]:
    graph_path = Path(path) if path else _GRAPH_PATH
    if not graph_path.is_file():
        return {"handoffs": {}}
    try:
        data = yaml.safe_load(graph_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {"handoffs": {}}
    except Exception as exc:
        logger.warning("Failed to load agent graph from %s: %s", graph_path, exc)
        return {"handoffs": {}}


def resolve_handoff(intent_key: str) -> tuple[str | None, list[str]]:
    """Return (primary_agent, fallback_agents) for an intent routing key."""
    graph = load_agent_graph()
    handoffs = graph.get("handoffs") or {}
    entry = handoffs.get(intent_key) if isinstance(handoffs, dict) else None
    if not isinstance(entry, dict):
        return None, []
    primary = entry.get("primary")
    fallbacks = entry.get("fallbacks") or []
    if not isinstance(fallbacks, list):
        fallbacks = []
    return (str(primary) if primary else None, [str(f) for f in fallbacks])


def handoff_stream_note(intent_key: str) -> str:
    primary, fallbacks = resolve_handoff(intent_key)
    if not primary:
        return ""
    if fallbacks:
        return f"Handoff: {primary} (fallback: {', '.join(fallbacks)})"
    return f"Handoff: {primary}"
