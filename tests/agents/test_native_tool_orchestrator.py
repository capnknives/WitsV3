"""Native tool loop: scripted fake LLM, fake registry, no Ollama."""

from __future__ import annotations

import json

import pytest

from agents.native_tool_orchestrator import NativeToolOrchestrator
from core.config import WitsV3Config
from core.llm_interface import LLMResponse
from core.prompt_prefix import CONTEXT_HEADING, prefix_fingerprint
from core.tool_call_guard import APOLOGY

pytestmark = pytest.mark.asyncio


class _FakeTool:
    def __init__(self, name: str, schema: dict):
        self.name = name
        self._schema = schema

    def get_schema(self) -> dict:
        return self._schema

    def get_llm_description(self) -> dict:
        return {"description": f"{self.name} tool"}


class _FakeRegistry:
    def __init__(self):
        self.tools = {
            "lookup": _FakeTool(
                "lookup",
                {
                    "name": "lookup",
                    "parameters": {
                        "type": "object",
                        "properties": {"term": {"type": "string"}},
                        "required": ["term"],
                    },
                },
            ),
            "clock": _FakeTool("clock", {"type": "object", "properties": {}}),
        }
        self.executed: list[tuple[str, dict]] = []

    def get_tool(self, name):
        return self.tools.get(name)

    async def execute_tool(self, name, **kwargs):
        self.executed.append((name, kwargs))
        return {"success": True, "result": f"{name} says {kwargs.get('term', 'noon')}"}


class _ScriptedLLM:
    """Returns the scripted replies in order and records what each call was sent."""

    def __init__(self, replies: list[LLMResponse]):
        self.replies = list(replies)
        self.calls: list[dict] = []

    async def generate_chat(self, messages, tools=None, model=None, temperature=None):
        self.calls.append({"messages": list(messages), "tools": tools})
        if not self.replies:
            return LLMResponse(content="out of script")
        return self.replies.pop(0)


def _call(name: str, arguments) -> dict:
    return {"name": name, "arguments": arguments}


def _tools(*calls: dict) -> LLMResponse:
    return LLMResponse(content="", tool_calls=list(calls))


def _make(replies: list[LLMResponse], **conversation) -> tuple:
    config = WitsV3Config()
    for key, value in conversation.items():
        setattr(config.conversation, key, value)
    llm = _ScriptedLLM(replies)
    registry = _FakeRegistry()
    orch = NativeToolOrchestrator("native", config, llm, None, registry)
    return orch, llm, registry


async def _run(orch, message: str = "what is the weather word") -> list:
    return [event async for event in orch.run(message)]


def _types(events) -> list[str]:
    return [e.type for e in events]


def _tool_messages(llm) -> list[str]:
    return [m.content for m in llm.calls[-1]["messages"] if m.role == "tool"]


async def test_plain_answer_one_call_no_tool_one_result():
    orch, llm, registry = _make([LLMResponse(content="Hello there.")])
    events = await _run(orch, "hi")
    assert len(llm.calls) == 1
    assert registry.executed == []
    assert _types(events).count("result") == 1
    assert events[-1].type == "result" and events[-1].content == "Hello there."
    assert "thinking" not in _types(events)


async def test_one_tool_then_answer():
    orch, llm, registry = _make([_tools(_call("lookup", {"term": "sun"})), LLMResponse("Sunny.")])
    events = await _run(orch)
    assert registry.executed == [("lookup", {"term": "sun"})]
    tool_events = [e for e in events if e.type == "tool_call"]
    assert [e.metadata for e in tool_events] == [{"tool_name": "lookup", "ok": True}]
    assert events[-1].type == "result" and events[-1].content == "Sunny."
    assert len(llm.calls) == 2


async def test_two_tools_in_one_round():
    orch, llm, registry = _make(
        [_tools(_call("lookup", '{"term": "sun"}'), _call("clock", {})), LLMResponse("Both.")]
    )
    events = await _run(orch)
    assert [name for name, _ in registry.executed] == ["lookup", "clock"]
    assert [e.metadata["tool_name"] for e in events if e.type == "tool_call"] == [
        "lookup",
        "clock",
    ]
    assert len(_tool_messages(llm)) == 2
    assert events[-1].type == "result"


async def test_bad_arguments_once_then_corrected():
    orch, llm, registry = _make(
        [
            _tools(_call("lookup", {"wrong": 1})),
            _tools(_call("lookup", {"term": "sun"})),
            LLMResponse("Fixed."),
        ]
    )
    events = await _run(orch)
    assert registry.executed == [("lookup", {"term": "sun"})]
    assert "Invalid arguments for 'lookup'" in llm.calls[1]["messages"][-1].content
    assert llm.calls[1]["messages"][-1].role == "tool"
    assert events[-1].type == "result" and events[-1].content == "Fixed."


async def test_bad_arguments_twice_ends_with_error():
    orch, llm, registry = _make(
        [_tools(_call("lookup", {"wrong": 1})), _tools(_call("lookup", {"wrong": 2}))]
    )
    events = await _run(orch)
    assert registry.executed == []
    assert events[-1].type == "error" and events[-1].content == APOLOGY
    assert "result" not in _types(events)


async def test_undecodable_arguments_are_a_schema_failure():
    orch, llm, registry = _make(
        [_tools(_call("lookup", "{not json")), _tools(_call("lookup", "{still not"))]
    )
    events = await _run(orch)
    assert registry.executed == []
    assert "not valid JSON" in llm.calls[1]["messages"][-1].content
    assert events[-1].type == "error" and events[-1].content == APOLOGY


async def test_identical_call_not_executed_a_third_time():
    same = _call("lookup", {"term": "sun"})
    orch, llm, registry = _make(
        [_tools(same), _tools(same), _tools(same), LLMResponse("Done looking.")]
    )
    events = await _run(orch)
    assert len(registry.executed) == 2
    assert "already called with these exact arguments" in _tool_messages(llm)[-1]
    assert len([e for e in events if e.type == "tool_call"]) == 2
    assert events[-1].type == "result" and events[-1].content == "Done looking."


async def test_max_steps_reached_stops_with_answer():
    replies = [_tools(_call("lookup", {"term": f"t{i}"})) for i in range(3)]
    replies.append(LLMResponse("Answer from what I found."))
    orch, llm, registry = _make(replies, max_steps=2)
    events = await _run(orch)
    assert len(registry.executed) == 2
    # Two tool rounds, one round whose call is stopped, then one tool-less answer call.
    assert len(llm.calls) == 4
    assert llm.calls[-1]["tools"] is None
    assert events[-1].type == "result" and events[-1].content == "Answer from what I found."


async def test_plain_text_tool_call_hidden_and_counts_as_schema_failure():
    text = '<tool_call>{"name": "lookup", "arguments": {"term": "sun"}}</tool_call>'
    orch, llm, registry = _make([LLMResponse(text), LLMResponse(text)])
    events = await _run(orch)
    assert registry.executed == []
    assert all("<tool_call>" not in (e.content or "") for e in events)
    assert len(llm.calls) == 2
    assert events[-1].type == "error" and events[-1].content == APOLOGY


async def test_plain_text_tool_call_once_then_answer():
    text = '<tool_call>{"name": "lookup", "arguments": {}}</tool_call>'
    orch, llm, registry = _make([LLMResponse(text), LLMResponse("Plain answer.")])
    events = await _run(orch)
    assert llm.calls[1]["messages"][-1].role == "tool"
    assert all("<tool_call>" not in (e.content or "") for e in events)
    assert events[-1].content == "Plain answer."


async def test_prefix_fingerprint_equal_across_two_turns():
    orch, llm, _ = _make([LLMResponse("One."), LLMResponse("Two.")])

    inventories = iter([{"a.md": 1}, {"a.md": 1, "b.md": 3}])

    async def _inventory():
        return next(inventories)

    orch._get_document_inventory = _inventory
    await _run(orch, "first question")
    await _run(orch, "a different second question")
    first, second = llm.calls[0], llm.calls[1]
    assert prefix_fingerprint(first["messages"], first["tools"]) == prefix_fingerprint(
        second["messages"], second["tools"]
    )
    last = second["messages"][-1]
    assert last.role == "user" and CONTEXT_HEADING in last.content
    assert "b.md" in last.content and "b.md" not in second["messages"][0].content
    assert [t["function"]["name"] for t in first["tools"]] == ["clock", "lookup"]
    assert first["tools"][1]["function"]["parameters"]["additionalProperties"] is False
    assert json.dumps(first["tools"]) == json.dumps(second["tools"])
