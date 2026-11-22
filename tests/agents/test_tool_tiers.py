"""Tool tiers (C4b): what the native loop sends, finds and refuses."""

from __future__ import annotations

import json
import logging

import pytest

from agents.native_tool_orchestrator import NativeToolOrchestrator
from agents.tool_tiers import (
    LOOP_TOOLS,
    OWNER_APPROVAL_STOP,
    build_tier_specs,
    resolve_tiers,
    route_call,
    search_lookup,
)
from core.config import ToolTierSettings, WitsV3Config
from core.llm_interface import LLMResponse
from core.tool_call_guard import APOLOGY
from core.tool_registry import ToolRegistry

pytestmark = pytest.mark.asyncio

_OBJ = {"type": "object", "properties": {}}


class _Tool:
    def __init__(self, name: str, description: str, schema: dict):
        self.name, self._description, self._schema = name, description, schema

    def get_schema(self) -> dict:
        return self._schema

    def get_llm_description(self) -> dict:
        return {"description": self._description}


class _Registry:
    def __init__(self):
        tools = [
            _Tool(
                "read_file",
                "Read a file",
                {
                    "type": "object",
                    "properties": {"file_path": {"type": "string"}},
                    "required": ["file_path"],
                },
            ),
            _Tool(
                "knowledge_log_summary",
                "Summarise the knowledge log of saved facts",
                {
                    "type": "object",
                    "properties": {"limit": {"type": "integer"}},
                },
            ),
            _Tool("guest_accounts_list", "List guest accounts", _OBJ),
            _Tool("restart_app", "Restart the app process", _OBJ),
        ]
        self.tools = {t.name: t for t in tools}
        self.executed: list[tuple[str, dict]] = []

    def get_tool(self, name):
        return self.tools.get(name)

    async def execute_tool(self, name, **kwargs):
        self.executed.append((name, kwargs))
        return {"success": True, "result": f"{name} ran"}


class _LLM:
    def __init__(self, replies: list):
        self.replies, self.calls = list(replies), []

    async def generate_chat(self, messages, tools=None, model=None, temperature=None):
        self.calls.append({"messages": list(messages), "tools": tools})
        return self.replies.pop(0) if self.replies else LLMResponse(content="done")


def _tools(*calls) -> LLMResponse:
    return LLMResponse(content="", tool_calls=[{"name": n, "arguments": a} for n, a in calls])


def _make(replies: list, registry=None):
    llm = _LLM(replies)
    registry = registry or _Registry()
    orch = NativeToolOrchestrator("native", WitsV3Config(), llm, None, registry)
    return orch, llm, registry


async def _run(orch, message: str = "do the thing") -> list:
    return [e async for e in orch.run(message)]


def _replies(llm) -> list[str]:
    return [m.content for m in llm.calls[-1]["messages"] if m.role == "tool"]


# --- pure parts -------------------------------------------------------------


def test_resolve_tiers_defaults_and_unknown_names_logged_once(caplog):
    settings = ToolTierSettings()
    with caplog.at_level(logging.WARNING, logger="agents.tool_tiers"):
        tiers = resolve_tiers(["read_file", "restart_app", "think"], ["read_file", "ghost"], [])
        resolve_tiers(["read_file"], ["ghost"], [])
    assert [r.message.count("ghost") for r in caplog.records] == [1]
    assert tiers.always == {"read_file"} and tiers.lookup == {"restart_app", "think"}
    tiers = resolve_tiers(["read_file", "restart_app"], settings.always, settings.owner_approval)
    assert tiers.always == {"read_file", *LOOP_TOOLS}
    assert tiers.owner_approval == {"restart_app"} and tiers.lookup == frozenset()


def test_name_in_both_lists_is_owner_approval():
    tiers = resolve_tiers(["x"], ["x"], ["x"])
    assert tiers.owner_approval == {"x"} and "x" not in tiers.always


def test_search_is_word_overlap_best_first_and_capped():
    lookup = [
        {"name": f"tool_{i}", "description": "guest helper", "parameters": _OBJ} for i in range(7)
    ] + [{"name": "guest_accounts_list", "description": "List guests", "parameters": _OBJ}]
    found = search_lookup("list the guests", lookup)
    assert len(found) == 5 and found[0]["name"] == "guest_accounts_list"
    assert search_lookup("weather radar", lookup) == []


def test_route_call_unwraps_and_rejects():
    tiers = resolve_tiers(["read_file", "x"], ["read_file", *LOOP_TOOLS], [])
    assert route_call("use_tool", '{"name": "x", "arguments": {"a": 1}}', tiers).name == "x"
    assert route_call("use_tool", {"name": "x"}, tiers).arguments == {}
    assert "directly" in route_call("use_tool", {"name": "read_file"}, tiers).reject
    assert route_call("use_tool", {"name": "use_tool"}, tiers).reject
    assert route_call("use_tool", {"name": "find_tools"}, tiers).reject
    bad = route_call("use_tool", {"tool": "x"}, tiers)
    assert bad.name == "use_tool" and not bad.reject
    assert route_call("read_file", {}, tiers).name == "read_file"


def test_real_registry_tiers():
    specs = build_tier_specs(ToolRegistry(), frozenset(), ToolTierSettings())
    sent = {t["function"]["name"] for t in specs.payload}
    assert sent == set(ToolTierSettings().always)
    assert specs.tiers.owner_approval == set(ToolTierSettings().owner_approval)
    assert not specs.tiers.owner_approval & set(specs.schemas)
    assert not specs.tiers.owner_approval & {e["name"] for e in specs.lookup}
    for tool in specs.payload:
        for name, prop in tool["function"]["parameters"]["properties"].items():
            assert len(prop.get("description", "")) >= 40, (tool["function"]["name"], name)


# --- the loop ---------------------------------------------------------------


async def test_payload_is_identical_across_rounds_and_turns_and_holds_the_always_tier():
    orch, llm, _ = _make(
        [_tools(("find_tools", {"query": "knowledge log"})), LLMResponse("one")]
        + [_tools(("read_file", {"file_path": "a"})), LLMResponse("two")]
    )
    await _run(orch, "first")
    await _run(orch, "second")
    payloads = [json.dumps(call["tools"]) for call in llm.calls]
    assert len(payloads) == 4 and len(set(payloads)) == 1
    names = [t["function"]["name"] for t in llm.calls[0]["tools"]]
    assert names == ["find_tools", "read_file", "use_tool"]


async def test_find_then_use_tool_runs_the_real_tool_and_events_carry_its_name():
    orch, llm, registry = _make(
        [
            _tools(("find_tools", {"query": "knowledge log summary"})),
            _tools(("use_tool", {"name": "knowledge_log_summary", "arguments": {"limit": 3}})),
            LLMResponse("Here it is."),
        ]
    )
    events = await _run(orch)
    found = json.loads(_replies(llm)[0])
    assert found["matches"][0]["name"] == "knowledge_log_summary"
    assert registry.executed == [("knowledge_log_summary", {"limit": 3})]
    names = [e.metadata["tool_name"] for e in events if e.type == "tool_call"]
    assert names == ["find_tools", "knowledge_log_summary"]


async def test_use_tool_bad_arguments_count_against_the_real_tool():
    bad = ("use_tool", {"name": "knowledge_log_summary", "arguments": {"limit": "x"}})
    orch, llm, registry = _make([_tools(bad), _tools(bad)])
    events = await _run(orch)
    assert registry.executed == []
    assert "Invalid arguments for 'knowledge_log_summary'" in _replies(llm)[0]
    assert events[-1].type == "error" and events[-1].content == APOLOGY


async def test_use_tool_arguments_not_an_object_fail_the_named_tool():
    orch, llm, _ = _make([_tools(("use_tool", {"name": "guest_accounts_list", "arguments": [1]}))])
    await _run(orch)
    assert "Invalid arguments for 'guest_accounts_list'" in _replies(llm)[0]


async def test_use_tool_naming_always_or_unknown_tool_is_rejected():
    orch, llm, registry = _make(
        [
            _tools(
                ("use_tool", {"name": "read_file", "arguments": {"file_path": "a"}}),
                ("use_tool", {"name": "no_such_tool"}),
            )
        ]
    )
    await _run(orch)
    first, second = _replies(llm)
    assert "call it directly" in first and "Unknown tool 'no_such_tool'" in second
    assert registry.executed == []


async def test_restart_app_refused_both_routes_and_third_request_ends_turn():
    direct = ("restart_app", {})
    wrapped = ("use_tool", {"name": "restart_app", "arguments": {}})
    orch, llm, registry = _make([_tools(direct), _tools(wrapped), _tools(direct)])
    events = await _run(orch)
    assert registry.executed == []
    assert all("owner's approval" in r for r in _replies(llm))
    assert not [e for e in events if e.type == "tool_call"]
    assert events[-1].type == "result"
    assert events[-1].content == OWNER_APPROVAL_STOP.format(name="restart_app")


async def test_find_tools_never_returns_an_owner_approval_tool():
    orch, llm, _ = _make([_tools(("find_tools", {"query": "restart app process"}))])
    await _run(orch)
    assert "restart_app" not in _replies(llm)[0]
