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
    "owner-regression",
    "refusal",
    "arithmetic",
    "self-knowledge",
}
VALID_ROUTES = {"coding", "self_repair", "book", "orchestrator"}
VALID_REPLY = {"required", "optional"}
CASE_KEYS = {"id", "message", "tags", "expect", "history"}
EXPECT_KEYS = {
    "tools",
    "forbid_tools",
    "forbid_routes",
    "reply",
    "tools_any_of",
    "reply_contains_any",
    "reply_not_contains",
    "reply_min_words",
    "max_first_result_ms",
    "timeout_seconds",
    "written_min_words",
}

# Substrings of an event's ``source`` (lower-cased) that identify each route.
# Real agent names: CodingExpert, AdvancedCoder, SystemDoctor, BookWriter, LLMOrchestrator,
# NativeToolOrchestrator.
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
    tools_any_of: list[str] = field(default_factory=list)
    reply_contains_any: list[str] = field(default_factory=list)
    reply_not_contains: list[str] = field(default_factory=list)
    reply_min_words: int = 0
    # Reason: carried for scripts/golden_run.py, which reports "slow"; evaluate() ignores it.
    max_first_result_ms: int = 0
    # Reason: carried for the runner, which uses it instead of --timeout-seconds when set.
    timeout_seconds: int = 0
    # Words the run's new files under the sandbox projects folder must hold in total.
    written_min_words: int = 0


@dataclass
class GoldenResult:
    """Outcome of checking one case against the events of a turn."""

    passed: bool
    failures: list[str]
    unobserved: list[str] = field(default_factory=list)


def _check_extra(cid: str, expect: dict[str, Any]) -> dict[str, Any]:
    """Validate the list and int expectation keys; return them as GoldenCase kwargs."""
    out: dict[str, Any] = {}
    for key in ("tools_any_of", "reply_contains_any", "reply_not_contains"):
        val = expect.get(key, [])
        if not isinstance(val, list) or not all(isinstance(v, str) and v for v in val):
            raise ValueError(f"case {cid}: bad {key} {val!r}")
        out[key] = val
    for key in ("reply_min_words", "max_first_result_ms", "timeout_seconds", "written_min_words"):
        val = expect.get(key, 0)
        if isinstance(val, bool) or not isinstance(val, int) or val < 0:
            raise ValueError(f"case {cid}: bad {key} {val!r}")
        out[key] = val
    return out


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
    # Reason: tools_any_of alone must not inherit the "none" default, which would contradict it;
    # forbid_tools alone means "anything but these", so other tools are allowed.
    if expect.get("tools_any_of"):
        default_tools = "any"
    elif expect.get("forbid_tools"):
        default_tools = "optional"
    else:
        default_tools = "none"
    tools = expect.get("tools", default_tools)
    if tools not in ("none", "any", "optional") and not (
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
    extra = _check_extra(cid, expect)
    history = raw.get("history") or []
    for turn in history:
        if not isinstance(turn, dict) or set(turn) != {"role", "content"}:
            raise ValueError(f"case {cid}: history items need role and content")
    return GoldenCase(
        cid, raw["message"], list(tags), tools, forbid_tools, forbid_routes, reply, history, **extra
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


def _reply_text(events: list[Any]) -> str:
    """Return the concatenated content of ``result`` events."""
    return "".join((ev.content or "") for ev in events if ev.type == "result")


def evaluate(
    case: GoldenCase,
    events: list[Any],
    tool_names: list[str] | None = None,
    *,
    written_words: int | None = None,
) -> GoldenResult:
    """Check one turn's events against a case's expectations.

    Args:
        case: The golden case.
        events: ``StreamData`` events of the turn, in order.
        tool_names: Authoritative tools called (from the turn-metrics record). When given
            it replaces the names read from events.
        written_words: Words in files the run created under the projects folder, measured
            by the runner; None when it could not measure.

    Returns:
        A result with one failure line per broken expectation. ``unobserved`` names tool
        expectations that passed only because no event carried a tool name.
    """
    failures: list[str] = []
    unobserved: list[str] = []
    seen = list(tool_names) if tool_names is not None else tools_seen(events)
    if tool_names is None and not seen:
        if case.tools == "none":
            unobserved.append("tools")
        if case.forbid_tools:
            unobserved.append("forbid_tools")
    if case.tools == "none":
        if seen:
            failures.append(f"tools: expected none, saw {seen}")
    elif case.tools == "optional":
        pass
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
    if case.tools_any_of and not set(case.tools_any_of) & set(seen):
        failures.append(f"tools_any_of: none of {case.tools_any_of} called, saw {seen}")
    text = _reply_text(events)
    if case.reply == "required" and not text.strip():
        failures.append("reply: required but no non-empty result event")
    # Reason: strip thousands separators so "885.5" matches "885.507" and "1,885" matches "1885".
    plain = text.replace(",", "").lower()
    if case.reply_contains_any and not any(
        s.replace(",", "").lower() in plain for s in case.reply_contains_any
    ):
        failures.append(f"reply_contains_any: none of {case.reply_contains_any} in reply")
    found = [s for s in case.reply_not_contains if s.lower() in text.lower()]
    if found:
        failures.append(f"reply_not_contains: reply has {found}")
    if len(text.split()) < case.reply_min_words:
        failures.append(f"reply_min_words: {len(text.split())} < {case.reply_min_words}")
    if case.written_min_words:
        if written_words is None:
            failures.append("written_min_words: not observable (no written word count)")
        elif written_words < case.written_min_words:
            failures.append(f"written_min_words: {written_words} < {case.written_min_words}")
    return GoldenResult(not failures, failures, unobserved)
