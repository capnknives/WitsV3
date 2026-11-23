"""Native loop and the tool context: tools get the turn's context, never the model's."""

from __future__ import annotations

import asyncio

import pytest

from agents.native_tool_orchestrator import NativeToolOrchestrator
from core.config import WitsV3Config
from core.llm_interface import LLMResponse
from core.schemas import ConversationHistory
from core.tool_context import CONTEXT_PARAMS, current_tool_context
from tools.conversation_history_tool import ReadConversationHistoryTool
from tools.file_tools import FileReadTool

pytestmark = pytest.mark.asyncio


class _WhoAmI:
    """Records the context it sees, after a pause so concurrent turns interleave."""

    name = "whoami"

    def get_schema(self) -> dict:
        return {"type": "object", "properties": {}}

    def get_llm_description(self) -> dict:
        return {"description": "whoami tool"}

    async def execute(self, **kwargs):
        await asyncio.sleep(0.01)
        ctx = current_tool_context()
        return f"session={ctx.session_id} role={ctx.user_role}"


class _Registry:
    """Runs the real tools, as the tool registry does."""

    def __init__(self):
        self.tools = {
            "read_conversation_history": ReadConversationHistoryTool(),
            "read_file": FileReadTool(),
            "whoami": _WhoAmI(),
        }
        self.executed: list[str] = []

    def get_tool(self, name):
        return self.tools.get(name)

    async def execute_tool(self, name, **kwargs):
        self.executed.append(name)
        return await self.tools[name].execute(**kwargs)


class _PerTurnLLM:
    """First reply calls the tool named for the turn; then answers with the tool reply."""

    def __init__(self, calls: dict[str, dict]):
        self.calls = calls
        self.sent: list[list] = []

    async def generate_chat(self, messages, tools=None, model=None, temperature=None):
        self.sent.append({"messages": list(messages), "tools": tools})
        replies = [m.content for m in messages if m.role == "tool"]
        if replies:
            return LLMResponse(content=replies[-1])
        user_text = [m.content for m in messages if m.role == "user"][-1]
        key = next(k for k in self.calls if k in user_text)
        return LLMResponse(content="", tool_calls=[self.calls[key]])


def _make(calls: dict[str, dict]) -> tuple:
    llm = _PerTurnLLM(calls)
    registry = _Registry()
    orch = NativeToolOrchestrator("native", WitsV3Config(), llm, None, registry)
    return orch, llm, registry


async def _result(orch, message: str, **kwargs) -> str:
    events = [e async for e in orch.run(message, **kwargs)]
    return [e for e in events if e.type == "result"][-1].content


async def test_read_conversation_history_gets_the_turn_conversation_with_no_argument():
    conv = ConversationHistory(session_id="s1")
    conv.add_message("user", "remember the blue kettle")
    conv.add_message("assistant", "Noted.")
    call = {"name": "read_conversation_history", "arguments": {}}
    orch, _, registry = _make({"history": call})
    out = await _result(orch, "history please", conversation_history=conv, session_id="s1")
    assert registry.executed == ["read_conversation_history"]
    assert "remember the blue kettle" in out


async def test_model_cannot_pass_user_role_to_read_file():
    call = {"name": "read_file", "arguments": {"file_path": "README.md", "user_role": "owner"}}
    orch, llm, registry = _make({"read": call})
    out = await _result(orch, "read it")
    assert registry.executed == []
    assert "user_role" in out and "not a known parameter" in out
    for tool in llm.sent[0]["tools"]:
        params = tool["function"]["parameters"]
        assert not CONTEXT_PARAMS & set(params["properties"])
        assert not CONTEXT_PARAMS & set(params.get("required", []))


async def test_two_concurrent_turns_keep_their_contexts_apart():
    call = {"name": "whoami", "arguments": {}}
    orch, _, _ = _make({"who": call})
    first, second = await asyncio.gather(
        _result(orch, "who a", session_id="a", user_role="owner"),
        _result(orch, "who b", session_id="b", user_role="family"),
    )
    assert first.endswith("session=a role=owner")
    # Reason: RT-10, "family" is not owner, so its turn is a guest and the owner-only tool
    # is blocked (it used to run with the raw role); the owner turn beside it is unaffected.
    assert "not available for guest" in second
    assert current_tool_context() is None


async def test_role_is_guest_when_the_caller_sends_none():
    # Reason: RT-10 reversed the old owner default; trusted callers now pass "owner".
    call = {"name": "whoami", "arguments": {}}
    orch, _, _ = _make({"who": call})
    out = await _result(orch, "who", session_id="c")
    assert "not available for guest" in out
