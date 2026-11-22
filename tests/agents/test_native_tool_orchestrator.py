"""Native tool loop: scripted fake LLM, fake registry, no Ollama."""

from __future__ import annotations

import json

import pytest

from agents.native_loop_support import MODEL_CALL_FAILED, STEP_LIMIT_ERROR
from agents.native_tool_orchestrator import NativeToolOrchestrator
from core.config import WitsV3Config
from core.llm_interface import LLMMessage, LLMResponse
from core.prompt_prefix import CONTEXT_HEADING, prefix_fingerprint
from core.schemas import ConversationHistory
from core.tool_call_guard import APOLOGY, OUT_OF_STEPS

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
            "boom": _FakeTool("boom", {"type": "object", "properties": {}}),
            "sad": _FakeTool("sad", {"type": "object", "properties": {}}),
        }
        self.executed: list[tuple[str, dict]] = []

    def get_tool(self, name):
        return self.tools.get(name)

    async def execute_tool(self, name, **kwargs):
        self.executed.append((name, kwargs))
        if name == "boom":
            raise RuntimeError("kaboom")
        if name == "sad":
            return {"success": False, "error": "no data"}
        return {"success": True, "result": f"{name} says {kwargs.get('term', 'noon')}"}


def _assert_well_formed(messages) -> None:
    """Every assistant tool_calls message is followed by one tool reply per call, and
    no tool reply appears anywhere else."""
    i = 0
    while i < len(messages):
        msg = messages[i]
        assert msg.role != "tool", f"tool reply at {i} answers no assistant tool call"
        if msg.role == "assistant" and msg.tool_calls:
            count = len(msg.tool_calls)
            replies = messages[i + 1 : i + 1 + count]
            assert len(replies) == count and all(
                r.role == "tool" for r in replies
            ), f"assistant tool_calls at {i} not fully answered"
            i += 1 + count
            continue
        i += 1


class _ScriptedLLM:
    """Returns the scripted replies in order and records what each call was sent."""

    def __init__(self, replies: list[LLMResponse]):
        self.replies = list(replies)
        self.calls: list[dict] = []

    async def generate_chat(self, messages, tools=None, model=None, temperature=None):
        self.calls.append({"messages": list(messages), "tools": tools, "model": model})
        _assert_well_formed(messages)
        if not self.replies:
            return LLMResponse(content="out of script")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _call(name: str, arguments) -> dict:
    return {"name": name, "arguments": arguments}


def _tools(*calls: dict) -> LLMResponse:
    return LLMResponse(content="", tool_calls=list(calls))


def _make(replies: list[LLMResponse], **conversation) -> tuple:
    config = WitsV3Config()
    # Reason: these tests are about the loop, so every fake tool is sent directly.
    config.conversation.tools.always = ["boom", "clock", "lookup", "sad"]
    for key, value in conversation.items():
        setattr(config.conversation, key, value)
    llm = _ScriptedLLM(replies)
    registry = _FakeRegistry()
    orch = NativeToolOrchestrator("native", config, llm, None, registry)
    return orch, llm, registry


async def _run(orch, message: str = "what is the weather word", **kwargs) -> list:
    events = [event async for event in orch.run(message, **kwargs)]
    for call in orch.llm_interface.calls:
        _assert_well_formed(call["messages"])
    return events


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
    sent = llm.calls[1]["messages"]
    assert (sent[-2].role, sent[-2].content, sent[-2].tool_calls) == ("assistant", text, None)
    assert sent[-1].role == "user" and "structured tool-call format" in sent[-1].content
    assert not any(m.role == "tool" for m in sent)
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
    names = [t["function"]["name"] for t in first["tools"]]
    assert names == sorted(names) == ["boom", "clock", "lookup", "sad"]
    # Reason: the sent view drops the closure flag (C4d); the guard schema keeps it.
    assert "additionalProperties" not in first["tools"][2]["function"]["parameters"]
    assert orch._tool_schemas["lookup"]["additionalProperties"] is False
    assert json.dumps(first["tools"]) == json.dumps(second["tools"])


async def test_assistant_tool_calls_precede_replies_including_rejected():
    orch, llm, _ = _make(
        [_tools(_call("lookup", '{"term": "sun"}'), _call("nope", {"x": 1})), LLMResponse("Ok.")]
    )
    await _run(orch)
    sent = llm.calls[1]["messages"]
    assistant = sent[-3]
    assert assistant.role == "assistant"
    assert assistant.tool_calls == [
        {"function": {"name": "lookup", "arguments": {"term": "sun"}}},
        {"function": {"name": "nope", "arguments": {"x": 1}}},
    ]
    assert [m.role for m in sent[-2:]] == ["tool", "tool"]
    assert "Unknown tool" in sent[-1].content


async def test_goal_left_out_when_it_equals_the_message():
    orch, llm, _ = _make([LLMResponse("Hi.")])
    await _run(orch, "  hello there ")
    last = llm.calls[0]["messages"][-1].content
    assert "goal:" not in last and "documents:" not in last


async def _two_turns_with_history() -> tuple:
    orch, llm, _ = _make([LLMResponse("One."), LLMResponse("Two.")])
    await _run(orch, "first question")
    history = ConversationHistory(session_id="s")
    history.add_message("user", "first question")
    history.add_message("assistant", "One.")
    await _run(orch, "second question", conversation_history=history)
    return llm.calls[0], llm.calls[1]


async def test_prefix_stable_span_holds_with_turn_one_in_history():
    first, second = await _two_turns_with_history()
    assert second["messages"][0] == first["messages"][0]
    assert json.dumps(first["tools"]) == json.dumps(second["tools"])
    assert [m.content for m in second["messages"][1:3]] == ["first question", "One."]


async def _turn(orch, history, question: str, memory: str) -> list:
    """One turn as the server runs it: user message added first, answer after."""

    async def _memory(goal):
        return memory

    orch._memory_context = _memory
    history.add_message("user", question)
    await _run(orch, question, conversation_history=history)
    history.add_message("assistant", f"Answer to {question}.")
    return orch.llm_interface.calls[-1]["messages"]


def _assert_shared_span(earlier: list, later: list, reply: str) -> None:
    as_sent = earlier + [LLMMessage("assistant", reply), later[-1]]
    assert prefix_fingerprint(as_sent, []) == prefix_fingerprint(later, [])


async def test_prefix_equal_over_shared_span_across_three_turns():
    replies = [LLMResponse(f"Answer to q{i}.") for i in range(3)]
    orch, _, _ = _make(replies)
    history = ConversationHistory(session_id="s")
    sent = [await _turn(orch, history, f"q{i}", f"memory {i}") for i in range(3)]
    for i in range(2):
        _assert_shared_span(sent[i], sent[i + 1], f"Answer to q{i}.")
    assert history.messages[0].metadata["sent_context"].startswith(f"\n\n{CONTEXT_HEADING}")
    assert [m.metadata for m in history.messages[1::2]] == [None] * 3


async def test_sent_context_survives_session_store(tmp_path):
    from core.session_store import load_sessions, persist_session

    orch, _, _ = _make([LLMResponse("Answer to q0."), LLMResponse("Answer to q1.")])
    history = ConversationHistory(session_id="s")
    first = await _turn(orch, history, "q0", "memory 0")
    persist_session(history, root=str(tmp_path))
    loaded = load_sessions(root=str(tmp_path))["s"]
    assert loaded.messages[0].metadata == history.messages[0].metadata
    _assert_shared_span(first, await _turn(orch, loaded, "q1", "memory 1"), "Answer to q0.")


async def test_older_matching_message_is_never_written():
    orch, llm, _ = _make([LLMResponse("Two.")])
    history = ConversationHistory(session_id="s")
    history.add_message("user", "same words")
    history.add_message("assistant", "One.")
    await _run(orch, "same words", conversation_history=history)
    assert history.messages[0].metadata is None
    assert [m.content for m in llm.calls[0]["messages"][1:3]] == ["same words", "One."]


async def test_tool_that_raises_becomes_reply_and_loop_continues():
    orch, llm, _ = _make([_tools(_call("boom", {})), LLMResponse("Recovered.")])
    events = await _run(orch)
    assert [e.metadata for e in events if e.type == "tool_call"] == [
        {"tool_name": "boom", "ok": False}
    ]
    assert "kaboom" in llm.calls[1]["messages"][-1].content
    assert events[-1].type == "result" and events[-1].content == "Recovered."


async def test_ok_false_from_tool_result_error_flag():
    orch, _, _ = _make([_tools(_call("sad", {})), LLMResponse("Nothing found.")])
    events = await _run(orch)
    assert [e.metadata["ok"] for e in events if e.type == "tool_call"] == [False]


async def test_llm_raises_mid_loop_after_a_tool_ran():
    orch, _, registry = _make([_tools(_call("clock", {})), RuntimeError("ollama down")])
    events = await _run(orch)
    assert registry.executed == [("clock", {})]
    assert _types(events).count("error") == 1 and "result" not in _types(events)
    assert events[-1].type == "error" and events[-1].content.startswith(MODEL_CALL_FAILED)


async def test_preflight_block_is_answered_and_not_run():
    orch, llm, registry = _make([_tools(_call("clock", {})), LLMResponse("Blocked, sorry.")])
    orch._preflight_tool_call = lambda name, args, state: f"Blocked {name}: test rule."
    events = await _run(orch)
    assert registry.executed == []
    assert "tool_call" not in _types(events)
    assert llm.calls[1]["messages"][-1].content == "Blocked clock: test rule."


async def test_stop_mid_round_answers_every_call_before_final_call():
    replies = [
        _tools(_call("lookup", {"term": "a"})),
        _tools(_call("lookup", {"term": "b"}), _call("clock", {})),
        LLMResponse("Final."),
    ]
    orch, llm, registry = _make(replies, max_steps=1)
    events = await _run(orch)
    assert len(registry.executed) == 1
    final = llm.calls[-1]["messages"]
    assert [m.role for m in final[-4:]] == ["assistant", "tool", "tool", "user"]
    assert final[-2].content == OUT_OF_STEPS and final[-1].content == OUT_OF_STEPS
    assert events[-1].content == "Final."


async def test_schema_stop_mid_round_still_answers_all_calls():
    bad = _call("lookup", {"wrong": 1})
    replies = [_tools(bad), _tools(_call("lookup", {"wrong": 2}), _call("clock", {}))]
    orch, llm, registry = _make(replies)
    events = await _run(orch)
    assert registry.executed == []
    assert events[-1].type == "error" and events[-1].content == APOLOGY


async def test_step_limit_without_observations_still_makes_final_call():
    text = '<tool_call>{"name": "lookup", "arguments": {}}</tool_call>'
    orch, llm, _ = _make(
        [LLMResponse(text), LLMResponse(text), LLMResponse("Best effort.")],
        max_steps=1,
        max_schema_retries=5,
    )
    events = await _run(orch)
    assert len(llm.calls) == 3 and llm.calls[-1]["tools"] is None
    assert events[-1].type == "result" and events[-1].content == "Best effort."


async def test_final_call_tool_call_text_is_not_shown():
    text = '<tool_call>{"name": "lookup", "arguments": {}}</tool_call>'
    replies = [_tools(_call("clock", {})), _tools(_call("clock", {"x": 1})), LLMResponse(text)]
    orch, llm, _ = _make(replies, max_steps=1)
    events = await _run(orch)
    assert llm.calls[-1]["tools"] is None
    assert all("<tool_call>" not in (e.content or "") for e in events)
    assert events[-1].type == "error" and events[-1].content == STEP_LIMIT_ERROR


async def test_non_object_arguments_go_to_the_guard_unchanged():
    orch, llm, registry = _make(
        [_tools(_call("lookup", "[1, 2]")), LLMResponse("Gave up on that call.")]
    )
    await _run(orch)
    assert registry.executed == []
    assert llm.calls[1]["messages"][-1].role == "tool"
    assert "Invalid arguments for 'lookup'" in llm.calls[1]["messages"][-1].content


async def test_no_calculator_shortcut_on_native_path():
    orch, llm, registry = _make([LLMResponse("4")])
    events = await _run(orch, "2 + 2", preferred_tool="calculator")
    assert len(llm.calls) == 1 and registry.executed == []
    assert events[-1].content == "4"


class _NoRouter:
    def route(self, *args, **kwargs):
        raise AssertionError("the native loop must not call the model router")


async def test_one_resident_model_for_every_round_and_router_never_called():
    replies = [_tools(_call("clock", {})), LLMResponse(content="done")]
    orch, llm, _ = _make(replies)
    orch.model_router = _NoRouter()
    events = await _run(orch)
    expected = orch.config.ollama_settings.orchestrator_model
    assert events[-1].type == "result"
    assert len(llm.calls) == 2 and {c["model"] for c in llm.calls} == {expected}


async def test_conversation_model_overrides():
    orch, llm, _ = _make([LLMResponse(content="hi")], model="custom:7b")
    orch.model_router = _NoRouter()
    await _run(orch)
    assert llm.calls[0]["model"] == "custom:7b"


async def test_tool_receives_exactly_the_validated_arguments():
    orch, _, registry = _make([_tools(_call("web_search", {"query": "x"})), LLMResponse("ok")])
    # Reason: web_search is the tool the legacy path injected user_role into.
    registry.tools["web_search"] = _FakeTool(
        "web_search",
        {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    )
    await _run(orch, user_role="owner", guest_profile={"a": 1}, preferred_tool="web_search")
    assert registry.executed == [("web_search", {"query": "x"})]


async def test_missing_model_404_yields_one_plain_error_naming_the_model():
    orch, _, _ = _make([RuntimeError("HTTP 404 Model not found")], model="ghost:1b")
    events = await _run(orch)
    errors = [e for e in events if e.type == "error"]
    assert len(errors) == 1
    assert "ghost:1b" in errors[0].content and "not installed" in errors[0].content
    assert "Traceback" not in errors[0].content
