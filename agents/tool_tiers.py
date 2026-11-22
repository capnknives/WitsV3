"""Tool tiers for the native loop (C4b): which tools are sent, found or refused.

Pure logic, no I/O. The loop (agents/native_tool_orchestrator.py) sends only the
``always`` tier, answers ``find_tools`` itself from the ``lookup`` tier, and unwraps
``use_tool`` into a call to the real tool, so the guard, preflight, tool context and
events all see the real name. ``owner_approval`` tools are refused by both routes.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from agents.native_loop_support import build_tool_specs
from core.prompt_prefix import order_tools
from core.tool_call_guard import Verdict

logger = logging.getLogger(__name__)

FIND_TOOLS = "find_tools"
USE_TOOL = "use_tool"
LOOP_TOOLS = (FIND_TOOLS, USE_TOOL)
MAX_MATCHES = 5
# Reason: the third request for one refused tool in a turn means the model will not stop
# asking, so the turn ends instead of burning the remaining steps.
MAX_APPROVAL_REQUESTS = 3

OWNER_APPROVAL_REJECT = (
    "'{name}' needs the owner's approval, and approval is not available yet, so it was "
    "not run. Do not call it again; tell the owner what you would have done."
)
OWNER_APPROVAL_STOP = (
    "That needs the owner's approval ('{name}'), which is not available yet, so I did "
    "not run it."
)

_LOOP_SPECS: dict[str, tuple[str, dict[str, Any]]] = {
    FIND_TOOLS: (
        "Find extra tools that are not in this list (guest accounts, knowledge log "
        "summary, running tests, code fixes, Python, MCP tools and more). Use it when no "
        "listed tool fits the request; do not use it for a tool you can already see. "
        "Returns up to five tools with their parameters; run one with use_tool. "
        'Example: {"query": "knowledge log summary"}.',
        {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "minLength": 1,
                    "description": "A few plain words for what the tool should do, "
                    "such as 'guest accounts list' or 'run tests'",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    ),
    USE_TOOL: (
        "Run a tool that find_tools returned. Use it only for tools found with "
        "find_tools; call the tools in this list directly instead. Example: "
        '{"name": "knowledge_log_summary", "arguments": {}}.',
        {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Exact tool name as find_tools returned it, "
                    "such as 'knowledge_log_summary'",
                },
                "arguments": {
                    "type": "object",
                    "description": "The tool's arguments as one JSON object matching "
                    "the parameters find_tools showed; {} when it takes none",
                },
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    ),
}

_logged_unknown: set[str] = set()
_WORD_RE = re.compile(r"[a-z0-9]+")
_STOP_WORDS = frozenset(
    "a an and are be can do for from i in is it me my of on or please the this to tool "
    "tools what with you".split()
)


@dataclass(frozen=True)
class ToolTiers:
    """Resolved tier membership. ``always`` may hold find_tools and use_tool."""

    always: frozenset[str]
    lookup: frozenset[str]
    owner_approval: frozenset[str]


@dataclass(frozen=True)
class TierSpecs:
    """What one turn of the loop needs from the registry.

    Attributes:
        tiers: Resolved tiers.
        payload: Tool list sent to the model: the ``always`` tier, sorted by name.
        schemas: Name to schema the guard checks: ``always`` and ``lookup`` tools, never
            ``owner_approval`` ones.
        lookup: ``{"name", "description", "parameters"}`` for each ``lookup`` tool.
    """

    tiers: ToolTiers
    payload: list[dict[str, Any]]
    schemas: dict[str, dict[str, Any]]
    lookup: list[dict[str, Any]]


@dataclass(frozen=True)
class Route:
    """Where one model call goes after unwrapping ``use_tool``.

    Attributes:
        name: The real tool name to check and run.
        arguments: The real tool's arguments, still undecoded.
        reject: When set, send this back to the model and do nothing else.
    """

    name: Any
    arguments: Any
    reject: str = ""


def _log_unknown_once(name: str) -> None:
    if name not in _logged_unknown:
        _logged_unknown.add(name)
        logger.warning("conversation.tools names %r, which is not a registered tool", name)


def resolve_tiers(
    registered: Iterable[str], always: Iterable[str], owner_approval: Iterable[str]
) -> ToolTiers:
    """Sort registered tools into tiers.

    Args:
        registered: Names of the registered tools the loop may use.
        always: ``conversation.tools.always``.
        owner_approval: ``conversation.tools.owner_approval``.

    Returns:
        The tiers. A configured name that is not registered (other than find_tools and
        use_tool) is logged once and ignored; a name in both lists is ``owner_approval``,
        the safer choice; every other registered tool is ``lookup``.
    """
    registered = set(registered)
    always, owner_approval = list(always), list(owner_approval)
    for name in always + owner_approval:
        if name not in registered and name not in LOOP_TOOLS:
            _log_unknown_once(name)
    approval = frozenset(n for n in owner_approval if n in registered)
    sent = frozenset(
        n for n in always if (n in registered or n in LOOP_TOOLS) and n not in approval
    )
    return ToolTiers(sent, frozenset(registered - sent - approval), approval)


def build_tier_specs(registry: Any, exclude: frozenset[str], settings: Any) -> TierSpecs:
    """Build the payload, guard schemas and lookup entries for one turn.

    Args:
        registry: Tool registry with a ``tools`` mapping, or None.
        exclude: Tool names the loop never offers.
        settings: ``conversation.tools`` (``always`` and ``owner_approval`` lists).

    Returns:
        The specs; the payload is the same bytes for as long as the registry and
        settings are unchanged, so Ollama keeps its cached prefix.
    """
    payload, schemas = build_tool_specs(registry, exclude)
    registered = [n for n in (registry.tools if registry else {}) if n not in exclude]
    tiers = resolve_tiers(registered, settings.always, settings.owner_approval)
    sent = [t for t in payload if t["function"]["name"] in tiers.always]
    guard = {n: s for n, s in schemas.items() if n not in tiers.owner_approval}
    for name in LOOP_TOOLS:
        if name in tiers.always:
            description, parameters = _LOOP_SPECS[name]
            sent.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": description,
                        "parameters": parameters,
                    },
                }
            )
            guard[name] = parameters
    lookup = [dict(t["function"]) for t in payload if t["function"]["name"] in tiers.lookup]
    return TierSpecs(tiers, order_tools(sent), guard, lookup)


def _words(text: str) -> set[str]:
    words = set()
    for word in _WORD_RE.findall(text.lower()):
        if word in _STOP_WORDS:
            continue
        # Reason: "guests" should find "guest"; a crude plural cut is enough here.
        words.add(word[:-1] if len(word) > 3 and word.endswith("s") else word)
    return words


def search_lookup(query: str, lookup: list[dict[str, Any]], limit: int = MAX_MATCHES) -> list:
    """Return up to ``limit`` lookup entries ranked by word overlap with ``query``.

    A word found in the tool name counts twice, one in the description once; entries
    with no overlap are left out. Ties go by name so the result is stable.

    Args:
        query: The model's search words.
        lookup: ``TierSpecs.lookup``; owner_approval tools are never in it.
        limit: Most entries to return.

    Returns:
        The matching entries, best first.
    """
    wanted = _words(query)
    scored = []
    for entry in lookup:
        score = 2 * len(wanted & _words(entry["name"])) + len(
            wanted & _words(entry.get("description", ""))
        )
        if score:
            scored.append((-score, entry["name"], entry))
    return [entry for _, _, entry in sorted(scored, key=lambda s: (s[0], s[1]))[:limit]]


def find_tools_reply(query: Any, lookup: list[dict[str, Any]]) -> str:
    """The tool reply for a guard-approved find_tools call."""
    matches = search_lookup(str(query), lookup)
    if not matches:
        return f"No extra tool matches {str(query)[:80]!r}. Try other words, or answer."
    return json.dumps(
        {"matches": matches, "next": "Run one with use_tool(name, arguments)."},
        ensure_ascii=False,
    )


def route_call(name: Any, arguments: Any, tiers: ToolTiers) -> Route:
    """Unwrap a ``use_tool`` call into the real tool call; other calls pass through.

    A malformed ``use_tool`` call (not an object, no string name, extra keys) is passed
    on as ``use_tool`` so the guard records the schema failure against it. Arguments
    that are not an object are passed on to fail the named tool's schema.

    Args:
        name: Tool name the model called.
        arguments: Its arguments (a JSON string or decoded value).
        tiers: Resolved tiers.

    Returns:
        The route for the loop.
    """
    if name != USE_TOOL or USE_TOOL not in tiers.always:
        return Route(name, arguments)
    outer = arguments
    if isinstance(outer, str):
        try:
            outer = json.loads(outer) if outer.strip() else {}
        except (ValueError, RecursionError):
            return Route(name, arguments)
    if (
        not isinstance(outer, dict)
        or not isinstance(outer.get("name"), str)
        or set(outer) - {"name", "arguments"}
    ):
        return Route(name, arguments)
    real = outer["name"]
    if real in LOOP_TOOLS:
        return Route(real, None, f"use_tool cannot run '{real}'; call {FIND_TOOLS} directly.")
    if real in tiers.always:
        return Route(real, None, f"'{real}' is already in your tool list; call it directly.")
    return Route(real, outer.get("arguments", {}))


class ApprovalGate:
    """Counts one turn's requests for ``owner_approval`` tools; never runs them."""

    def __init__(self) -> None:
        """Start with no requests counted."""
        self.requests: dict[str, int] = {}

    def check(self, name: str) -> Verdict:
        """Reject the request, or stop the turn on the third one for the same tool.

        Rejections are policy, not formatting errors, so they never count as schema
        failures in the guard.
        """
        self.requests[name] = self.requests.get(name, 0) + 1
        if self.requests[name] >= MAX_APPROVAL_REQUESTS:
            return Verdict("stop", OWNER_APPROVAL_STOP.format(name=name))
        return Verdict("reject", OWNER_APPROVAL_REJECT.format(name=name))
