"""Tool tiers (C4b): what the native loop sends, finds and refuses."""

from __future__ import annotations

import json
import logging

import pytest

from agents.native_tool_orchestrator import NativeToolOrchestrator
from agents.tool_tiers import (
    LOOP_TOOLS,
    OWNER_APPROVAL_FLOOR,
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
from core.turn_metrics import start_turn
from tools.ask_claude_tool import AskClaudeTool
from tools.guest_audit_tool import GuestSetAgeBandTool
from tools.network_control_tool import NetworkControlTool
from tools.self_repair_tools import RestartAppTool

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
        self.replies, self.calls, self.forever = list(replies), [], None

    async def generate_chat(self, messages, tools=None, model=None, temperature=None):
        self.calls.append({"messages": list(messages), "tools": tools})
        if tools is None:
            return LLMResponse(content="final answer")
        if self.forever is not None:
            return self.forever
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
        tiers = resolve_tiers(["read_file", "think"], ["read_file", "ghost"], [])
        resolve_tiers(["read_file"], ["ghost"], [])
    assert [r.message.count("ghost") for r in caplog.records] == [1]
    assert tiers.always == {"read_file"} and tiers.lookup == {"think"}
    tiers = resolve_tiers(["read_file", "restart_app"], settings.always, settings.owner_approval)
    assert tiers.always == {"read_file", *LOOP_TOOLS}
    assert tiers.owner_approval == {"restart_app"} and tiers.lookup == frozenset()


def test_name_in_both_lists_is_owner_approval():
    tiers = resolve_tiers(["x"], ["x"], ["x"])
    assert tiers.owner_approval == {"x"} and "x" not in tiers.always


@pytest.mark.parametrize("always,approval", [([], []), (None, None), (["restart_app"], [])])
def test_empty_settings_never_move_a_floor_tool_out(always, approval):
    names = ["restart_app", "network_control", "guest_set_age_band", "ask_claude", "think"]
    tiers = resolve_tiers(names, always, approval)
    assert tiers.owner_approval == OWNER_APPROVAL_FLOOR
    assert tiers.lookup == {"think"} and not tiers.always & OWNER_APPROVAL_FLOOR


def test_typo_settings_warn_and_names_match_after_strip_and_casefold(caplog):
    with caplog.at_level(logging.WARNING, logger="agents.tool_tiers"):
        tiers = resolve_tiers(
            ["read_file", "think", "python_execute"],
            [" Read_File ", "reed_file_typo"],
            ["PYTHON_EXECUTE", "thinkk_typo"],
        )
    assert tiers.always == {"read_file"} and tiers.owner_approval == {"python_execute"}
    warned = " ".join(r.message for r in caplog.records)
    assert "reed_file_typo" in warned and "thinkk_typo" in warned


def test_requires_owner_approval_flag_puts_a_tool_in_the_tier():
    class _Flagged(_Tool):
        requires_owner_approval = True

    registry = _Registry()
    registry.tools["wipe_disk"] = _Flagged("wipe_disk", "Wipe", _OBJ)
    settings = ToolTierSettings(always=["wipe_disk", "read_file"], owner_approval=[])
    specs = build_tier_specs(registry, frozenset(), settings)
    assert "wipe_disk" in specs.tiers.owner_approval and "wipe_disk" not in specs.schemas
    for cls in (RestartAppTool, NetworkControlTool, GuestSetAgeBandTool, AskClaudeTool):
        assert cls.requires_owner_approval is True


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
    for odd in (["x"], {"n": 1}, None):
        assert route_call(odd, {}, tiers).name == odd


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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
async def test_use_tool_bad_arguments_count_against_the_real_tool():
    bad = ("use_tool", {"name": "knowledge_log_summary", "arguments": {"limit": "x"}})
    orch, llm, registry = _make([_tools(bad), _tools(bad)])
    events = await _run(orch)
    assert registry.executed == []
    assert "Invalid arguments for 'knowledge_log_summary'" in _replies(llm)[0]
    assert events[-1].type == "error" and events[-1].content == APOLOGY


@pytest.mark.asyncio
async def test_use_tool_arguments_not_an_object_fail_the_named_tool():
    orch, llm, _ = _make([_tools(("use_tool", {"name": "guest_accounts_list", "arguments": [1]}))])
    await _run(orch)
    assert "Invalid arguments for 'guest_accounts_list'" in _replies(llm)[0]


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
async def test_find_tools_never_returns_an_owner_approval_tool():
    orch, llm, _ = _make([_tools(("find_tools", {"query": "restart app process"}))])
    await _run(orch)
    assert "restart_app" not in _replies(llm)[0]


@pytest.mark.asyncio
async def test_name_variants_never_run_an_owner_approval_tool():
    variants = ["Restart_App", " restart_app ", "restart\u200b_app", "RESTART_APP\ufeff"]
    calls = [(v, {}) for v in variants] + [("use_tool", {"name": v}) for v in variants]
    for name, args in calls:
        orch, llm, registry = _make([_tools((name, args))])
        await _run(orch)
        assert registry.executed == []
        assert "owner's approval" in _replies(llm)[0], name


@pytest.mark.asyncio
async def test_non_string_names_are_rejected_as_unknown_without_raising():
    orch, llm, registry = _make([_tools((["restart_app"], {}), (None, {}), ({"a": 1}, {}))])
    events = await _run(orch)
    assert registry.executed == [] and events[-1].type in ("result", "error")
    assert all("Unknown tool" in r for r in _replies(llm))


@pytest.mark.asyncio
async def test_alternating_two_owner_approval_tools_ends_the_turn():
    orch, llm, registry = _make(
        [_tools(("restart_app", {})), _tools(("ask_claude", {}))] * 3,
        registry=_RegistryWithAsk(),
    )
    events = await _run(orch)
    assert registry.executed == [] and len(llm.calls) == 5
    assert events[-1].content == OWNER_APPROVAL_STOP.format(name="restart_app")


@pytest.mark.asyncio
async def test_only_rejected_tier_calls_end_at_max_steps_with_an_answer():
    orch, llm, registry = _make([])
    llm.forever = _tools(("use_tool", {"name": "read_file", "arguments": {"file_path": "a"}}))
    events = await _run(orch)
    steps = orch.config.conversation.max_steps
    assert registry.executed == []
    assert len(llm.calls) == steps + 2 and llm.calls[-1]["tools"] is None
    assert events[-1].type == "result" and events[-1].content == "final answer"


@pytest.mark.asyncio
async def test_malformed_use_tool_is_never_run_and_counts_as_schema_failure():
    bad = ("use_tool", {"tool": "knowledge_log_summary"})
    orch, llm, registry = _make([_tools(bad), _tools(bad)])
    events = await _run(orch)
    assert registry.executed == [] and events[-1].content == APOLOGY
    assert "Invalid arguments for 'use_tool'" in _replies(llm)[0]


@pytest.mark.asyncio
async def test_repeat_cap_applies_through_use_tool():
    call = ("use_tool", {"name": "guest_accounts_list", "arguments": {}})
    orch, llm, registry = _make([_tools(call), _tools(call), _tools(call)])
    await _run(orch)
    assert [n for n, _ in registry.executed] == ["guest_accounts_list"] * 2
    assert "already called" in _replies(llm)[-1]


@pytest.mark.asyncio
async def test_metrics_and_event_carry_the_real_name_for_use_tool():
    turn = start_turn("test", "s1", 4)
    orch, _, _ = _make(
        [_tools(("use_tool", {"name": "think", "arguments": {"thought": "hm"}}))],
        registry=ToolRegistry(),
    )
    events = await _run(orch)
    assert [c["name"] for c in turn.tool_calls] == ["think"]
    assert [e.metadata["tool_name"] for e in events if e.type == "tool_call"] == ["think"]


@pytest.mark.asyncio
async def test_find_tools_on_a_real_registry_never_returns_owner_approval_tools():
    query = "restart app network control guest age band ask claude question"
    orch, llm, _ = _make([_tools(("find_tools", {"query": query}))], registry=ToolRegistry())
    await _run(orch)
    reply = _replies(llm)[0]
    assert not [n for n in OWNER_APPROVAL_FLOOR if n in reply]
    assert "guest_accounts_list" in reply


class _RegistryWithAsk(_Registry):
    def __init__(self):
        super().__init__()
        self.tools["ask_claude"] = _Tool("ask_claude", "Ask Claude", _OBJ)


@pytest.mark.asyncio
async def test_a_loop_only_name_reaching_execution_is_rejected(monkeypatch):
    import agents.native_tool_orchestrator as loop
    from agents.tool_tiers import Route

    # Reason: simulate a routing bug that lets use_tool through to the guard unchanged.
    monkeypatch.setattr(loop, "route_call", lambda name, args, tiers: Route(name, args))
    orch, llm, registry = _make([_tools(("use_tool", {"name": "guest_accounts_list"}))])
    orch._run_tool = None  # Reason: any attempt to run it would raise here.
    await _run(orch)
    assert registry.executed == [] and "cannot run here" in _replies(llm)[0]
