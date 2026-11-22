"""Golden turn cases: loading, validation and a code-only check over StreamData events.

A case lists expectations about one turn (tools called, routes taken, reply present).
No model judges anything; every check reads the ``StreamData`` events of the turn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

VALID_TAGS = {
    "must-not-route",
    "chat",
    "tool",
    "coding",
    "repair",
    "writing",
    "command",
    "follow-up",
}
VALID_ROUTES = {"coding", "self_repair", "book", "orchestrator"}
VALID_REPLY = {"required", "optional"}
CASE_KEYS = {"id", "message", "tags", "expect", "history"}
EXPECT_KEYS = {"tools", "forbid_tools", "forbid_routes", "reply"}

# Substrings of an event's ``source`` (lower-cased) that identify each route.
ROUTE_MARKERS = {
    "coding": ("coder", "coding"),
    "self_repair": ("repair", "doctor"),
    "book": ("book",),
    "orchestrator": ("orchestrator",),
}


@dataclass
class GoldenCase:
    """One golden turn: a message, optional history and its expectations."""

    id: str
    message: str
    tags: list[str]
    tools: str | list[str] = "none"
    forbid_tools: list[str] = field(default_factory=list)
    forbid_routes: list[str] = field(default_factory=list)
    reply: str = "required"
    history: list[dict[str, str]] = field(default_factory=list)


@dataclass
class GoldenResult:
    """Outcome of checking one case against the events of a turn."""

    passed: bool
    failures: list[str]


def _check_case(raw: Any, seen: set[str]) -> GoldenCase:
    if not isinstance(raw, dict):
        raise ValueError(f"case is not a mapping: {raw!r}")
    cid = raw.get("id")
    if not isinstance(cid, str) or not cid:
        raise ValueError(f"case without a string id: {raw!r}")
    if cid in seen:
        raise ValueError(f"case {cid}: duplicate id")
    seen.add(cid)
    unknown = set(raw) - CASE_KEYS
    if unknown:
        raise ValueError(f"case {cid}: unknown key {sorted(unknown)}")
    if not isinstance(raw.get("message"), str) or not raw["message"]:
        raise ValueError(f"case {cid}: message must be a non-empty string")
    tags = raw.get("tags") or []
    bad = [t for t in tags if t not in VALID_TAGS]
    if not tags or bad:
        raise ValueError(f"case {cid}: bad tags {bad or 'empty'}")
    expect = raw.get("expect") or {}
    unknown = set(expect) - EXPECT_KEYS
    if unknown:
        raise ValueError(f"case {cid}: unknown expect key {sorted(unknown)}")
    tools = expect.get("tools", "none")
    if tools not in ("none", "any") and not (
        isinstance(tools, list) and tools and all(isinstance(t, str) for t in tools)
    ):
        raise ValueError(f"case {cid}: bad expect.tools {tools!r}")
    forbid_tools = expect.get("forbid_tools", [])
    if not isinstance(forbid_tools, list):
        raise ValueError(f"case {cid}: forbid_tools must be a list")
    forbid_routes = expect.get("forbid_routes", [])
    if not isinstance(forbid_routes, list) or set(forbid_routes) - VALID_ROUTES:
        raise ValueError(f"case {cid}: bad forbid_routes {forbid_routes!r}")
    reply = expect.get("reply", "required")
    if reply not in VALID_REPLY:
        raise ValueError(f"case {cid}: bad reply {reply!r}")
    history = raw.get("history") or []
    for turn in history:
        if not isinstance(turn, dict) or set(turn) != {"role", "content"}:
            raise ValueError(f"case {cid}: history items need role and content")
    return GoldenCase(
        cid, raw["message"], list(tags), tools, forbid_tools, forbid_routes, reply, history
    )


def load_cases(path: str | Path) -> list[GoldenCase]:
    """Load and validate golden cases from a YAML file.

    Args:
        path: YAML file holding a list of case mappings.

    Returns:
        The cases in file order.

    Raises:
        ValueError: A case has an unknown key, a duplicate id or a bad enum value.
    """
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("golden file must be a list of cases")
    seen: set[str] = set()
    return [_check_case(raw, seen) for raw in data]


def tools_seen(events: list[Any]) -> list[str]:
    """Return tool names from tool_call and action events' metadata, in order."""
    names: list[str] = []
    for ev in events:
        if ev.type not in ("tool_call", "action"):
            continue
        meta = ev.metadata or {}
        name = meta.get("tool_name") or meta.get("tool")
        if name:
            names.append(str(name))
    return names


def routes_seen(events: list[Any]) -> set[str]:
    """Return the routes whose marker appears in any event's source."""
    found: set[str] = set()
    for ev in events:
        source = (ev.source or "").lower()
        for route, markers in ROUTE_MARKERS.items():
            if any(m in source for m in markers):
                found.add(route)
    return found


def evaluate(case: GoldenCase, events: list[Any]) -> GoldenResult:
    """Check one turn's events against a case's expectations.

    Args:
        case: The golden case.
        events: ``StreamData`` events of the turn, in order.

    Returns:
        A result with one failure line per broken expectation.
    """
    failures: list[str] = []
    seen = tools_seen(events)
    if case.tools == "none":
        if seen:
            failures.append(f"tools: expected none, saw {seen}")
    elif case.tools == "any":
        if not seen:
            failures.append("tools: expected any tool call; not observable or none made")
    else:
        missing = [t for t in case.tools if t not in seen]
        if missing:
            note = "" if seen else " (not observable: no tool events carry a name)"
            failures.append(f"tools: missing {missing}{note}")
    hit = sorted(set(case.forbid_tools) & set(seen))
    if hit:
        failures.append(f"forbid_tools: called {hit}")
    routed = sorted(set(case.forbid_routes) & routes_seen(events))
    if routed:
        failures.append(f"forbid_routes: took {routed}")
    if case.reply == "required":
        has_reply = any(ev.type == "result" and (ev.content or "").strip() for ev in events)
        if not has_reply:
            failures.append("reply: required but no non-empty result event")
    return GoldenResult(not failures, failures)
