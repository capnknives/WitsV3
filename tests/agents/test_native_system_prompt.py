"""Native loop system prompt: built from the template, same bytes every turn."""

from __future__ import annotations

import pytest

from agents.native_loop_support import NATIVE_SYSTEM_TEMPLATE, native_system_prompt
from agents.native_tool_orchestrator import NativeToolOrchestrator
from core.config import WitsV3Config
from core.llm_interface import LLMResponse
from core.schemas import ConversationHistory

MODEL = "resident-test:7b"
ANCHORS = (
    "names a length over about 600 words",
    "document_search before answering",
    "without asking permission first",
    "look before answering",
    "did not come from a tool",
    "A stored note is not a feature",
    "newest user message",
    "find_tools",
)


class _LLM:
    def __init__(self):
        self.sent: list[tuple[list, str | None]] = []

    async def generate_chat(self, messages, tools=None, model=None, temperature=None):
        self.sent.append((list(messages), model))
        return LLMResponse(content="ok")


@pytest.mark.parametrize("anchor", ANCHORS)
def test_template_has_anchor(anchor):
    assert anchor in NATIVE_SYSTEM_TEMPLATE


def test_prompt_is_short_and_names_the_model():
    prompt = native_system_prompt("m" * 20)
    assert len(prompt) < 2400 and "m" * 20 in prompt and "{model}" not in prompt
    assert prompt.startswith("You are WITS, a local assistant")
    assert "\n\n\n" not in prompt and prompt.count("\n\n") == 8


@pytest.mark.asyncio
async def test_system_message_is_the_prompt_and_identical_across_turns():
    config = WitsV3Config()
    config.conversation.model = MODEL
    llm = _LLM()
    orch = NativeToolOrchestrator("native", config, llm, None, None)
    inventories = iter([{}, {"a.md": 1}, {"a.md": 1, "b.md": 4}])
    memories = iter(["", "fact one", "fact two"])

    async def _inventory():
        return next(inventories)

    async def _memory(goal):
        return next(memories)

    orch._get_document_inventory = _inventory
    orch._memory_context = _memory
    history = ConversationHistory(session_id="s")
    for question in ("first", "second question", "third, longer question"):
        history.add_message("user", question)
        [e async for e in orch.run(question, conversation_history=history)]
        history.add_message("assistant", "ok")
    expected = native_system_prompt(MODEL)
    assert MODEL in expected
    for messages, model in llm.sent:
        assert messages[0].role == "system" and messages[0].content == expected
        assert model == MODEL
    assert len({messages[0].content.encode() for messages, _ in llm.sent}) == 1
    assert len(llm.sent[2][0]) > len(llm.sent[0][0])
