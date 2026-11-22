"""Tool context: ContextVar scoping, fallbacks, schema stripping and the tools that read it."""

from __future__ import annotations

import asyncio

import pytest

from core.schemas import ConversationHistory
from core.tool_call_guard import normalise_schema
from core.tool_context import (
    CONTEXT_PARAMS,
    ToolContext,
    current_tool_context,
    strip_context_params,
    tool_conversation,
    tool_user_role,
    use_tool_context,
)
from tools.conversation_history_tool import AnalyzeConversationTool, ReadConversationHistoryTool
from tools.knowledge_log_tool import KnowledgeLogSummaryTool


def _ctx(role: str = "owner", conversation=None, session: str = "s1") -> ToolContext:
    return ToolContext(
        session_id=session, user_role=role, conversation_history=conversation, turn_id="t1"
    )


def _conversation(text: str) -> ConversationHistory:
    conv = ConversationHistory(session_id="s1")
    conv.add_message("user", text)
    return conv


def test_default_is_none_and_context_is_restored_on_exit():
    assert current_tool_context() is None
    outer, inner = _ctx(session="outer"), _ctx(session="inner")
    with use_tool_context(outer):
        with use_tool_context(inner):
            assert current_tool_context() is inner
        assert current_tool_context() is outer
    assert current_tool_context() is None


def test_context_is_restored_when_the_block_raises():
    with pytest.raises(RuntimeError):
        with use_tool_context(_ctx()):
            raise RuntimeError("boom")
    assert current_tool_context() is None


@pytest.mark.asyncio
async def test_context_survives_awaits_and_reaches_gathered_tasks():
    ctx = _ctx()

    async def seen():
        await asyncio.sleep(0)
        return current_tool_context()

    with use_tool_context(ctx):
        await asyncio.sleep(0)
        assert current_tool_context() is ctx
        assert await asyncio.gather(seen(), seen()) == [ctx, ctx]


@pytest.mark.asyncio
async def test_concurrent_tasks_keep_their_contexts_apart():
    async def turn(name: str):
        with use_tool_context(_ctx(session=name)):
            await asyncio.sleep(0.01)
            return current_tool_context().session_id

    assert await asyncio.gather(turn("a"), turn("b")) == ["a", "b"]
    assert current_tool_context() is None


def test_without_context_the_argument_is_used_unchanged():
    assert tool_user_role("guest") == "guest"
    assert tool_user_role("admin") == "admin"
    assert tool_user_role() == "owner"
    conv = _conversation("hi")
    assert tool_conversation(conv) is conv


@pytest.mark.parametrize("role,expected", [("owner", "owner"), ("guest", "guest"), ("", "guest")])
def test_context_role_wins_and_anything_but_owner_is_guest(role, expected):
    with use_tool_context(_ctx(role=role)):
        # Reason: the argument a caller passes must never override the context.
        assert tool_user_role("owner") == expected
        assert tool_user_role("guest") == expected


def test_strip_removes_context_params_from_properties_and_required():
    raw = {
        "type": "object",
        "properties": {"file_path": {"type": "string"}, "user_role": {"type": "string"}},
        "required": ["file_path", "user_role", "session_id"],
    }
    schema = strip_context_params(normalise_schema(raw))
    assert set(schema["properties"]) == {"file_path"}
    assert schema["required"] == ["file_path"]
    assert not CONTEXT_PARAMS & set(schema["properties"])
    assert "user_role" in raw["properties"]


@pytest.mark.asyncio
async def test_read_conversation_history_reads_the_context_conversation():
    tool = ReadConversationHistoryTool()
    with use_tool_context(_ctx(conversation=_conversation("from the turn"))):
        out = await tool.execute(conversation_history=_conversation("from an argument"))
    assert "from the turn" in out and "from an argument" not in out
    assert "from an argument" in await tool.execute(_conversation("from an argument"))
    assert await tool.execute() == "Starting new conversation."


@pytest.mark.asyncio
async def test_analyze_conversation_reads_the_context_conversation():
    with use_tool_context(_ctx(conversation=_conversation("one"))):
        result = await AnalyzeConversationTool().execute()
    assert result["message_count"] == 1


@pytest.mark.asyncio
async def test_owner_only_tool_refuses_a_non_owner_context_even_if_argument_says_owner():
    tool = KnowledgeLogSummaryTool()
    with use_tool_context(_ctx(role="family")):
        out = await tool.execute(user_role="owner")
    assert out == "knowledge_log_summary is only available to the owner."
