"""Native loop memory context: only durable owner-stated facts, framed as background."""

from __future__ import annotations

import pytest

from agents.native_loop_support import (
    MEMORY_FACT_CHARS,
    MEMORY_HEADING,
    native_system_prompt,
    resident_model,
)
from agents.native_tool_orchestrator import NativeToolOrchestrator
from core.config import WitsV3Config
from core.llm_interface import LLMResponse
from core.memory_manager import MemorySegment, MemorySegmentContent

pytestmark = pytest.mark.asyncio

EDGE = "Please open microsoft edge"
BISCUIT = "the owner's dog is called Biscuit"


def _seg(kind: str, text: str, score: float | None = None) -> MemorySegment:
    return MemorySegment(
        type=kind,
        source="test",
        content=MemorySegmentContent(text=text),
        relevance_score=score,
    )


class _FakeMemory:
    """Honours the equality filter like the real backends; ignores everything else."""

    def __init__(self, segments: list[MemorySegment]):
        self.segments = segments

    async def search_memory(self, query_text, limit=5, min_relevance=0.0, filter_dict=None):
        found = [
            s
            for s in self.segments
            if all(getattr(s, k, None) == v for k, v in (filter_dict or {}).items())
        ]
        return found[:limit]


class _LLM:
    def __init__(self):
        self.sent: list[list] = []

    async def generate_chat(self, messages, tools=None, model=None, temperature=None):
        self.sent.append(list(messages))
        return LLMResponse(content="ok")


def _orch(segments, inventory=None, **conversation):
    config = WitsV3Config()
    for key, value in conversation.items():
        setattr(config.conversation, key, value)
    llm = _LLM()
    orch = NativeToolOrchestrator("native", config, llm, _FakeMemory(segments), None)

    async def _no_docs():
        return inventory or {}

    orch._get_document_inventory = _no_docs
    return orch, llm


async def _turn(orch, message="Please restart yourself now."):
    return [e async for e in orch.run(message)]


def _user_text(llm, call=0) -> str:
    return llm.sent[call][-1].content


def _edge_store(*extra):
    return [
        _seg("USER_INPUT", f"User input: {EDGE}"),
        _seg("GOAL", f"Goal: {EDGE}"),
        _seg("FINAL_ANSWER", "Final Answer: I opened Edge"),
        _seg("REASONING", "thinking about edge"),
        _seg("OBSERVATION", "edge launched"),
        _seg("USER_FACT", BISCUIT),
        *extra,
    ]


async def test_only_the_durable_fact_reaches_the_turn():
    orch, llm = _orch(_edge_store())
    await _turn(orch)
    text = _user_text(llm)
    assert f"{MEMORY_HEADING}\n- {BISCUIT}" in text
    for leaked in ("edge", "Edge", "User input", "Goal:", "Final Answer"):
        assert leaked not in text


async def test_no_block_when_only_excluded_types_exist():
    orch, llm = _orch(_edge_store()[:5])
    await _turn(orch)
    text = _user_text(llm)
    assert MEMORY_HEADING not in text and "memory:" not in text and "edge" not in text.lower()


async def test_configured_reasoning_is_still_refused():
    orch, llm = _orch(_edge_store(), memory_types=["USER_FACT", "REASONING", "USER_INPUT"])
    await _turn(orch)
    text = _user_text(llm)
    assert BISCUIT in text and "thinking about edge" not in text and "microsoft" not in text


async def test_long_fact_truncated_and_prefix_stripped():
    long_fact = "x" * 900
    orch, llm = _orch([_seg("USER_FACT", long_fact), _seg("USER_FACT", "User input: I like tea")])
    await _turn(orch)
    text = _user_text(llm)
    assert "x" * MEMORY_FACT_CHARS in text and "x" * (MEMORY_FACT_CHARS + 1) not in text
    assert "- I like tea" in text and "User input" not in text


async def test_min_score_drops_weak_matches():
    segs = [_seg("USER_FACT", "weak fact", 0.1), _seg("USER_FACT", "strong fact", 0.9)]
    orch, llm = _orch(segs, memory_min_score=0.5)
    await _turn(orch)
    assert "strong fact" in _user_text(llm) and "weak fact" not in _user_text(llm)


async def test_system_message_identical_across_turns():
    orch, llm = _orch(_edge_store())
    await _turn(orch, "first")
    await _turn(orch, "second, different")
    prompt = native_system_prompt(resident_model(orch.config))
    assert llm.sent[0][0].content == llm.sent[1][0].content == prompt
    assert "newest user message" in prompt


async def test_documents_line_only_when_something_is_ingested():
    orch, llm = _orch([_seg("USER_FACT", BISCUIT)])
    await _turn(orch, "first")
    assert "documents:" not in _user_text(llm) and "No user documents" not in _user_text(llm)
    orch, llm = _orch([_seg("USER_FACT", BISCUIT)], inventory={"notes.md": 2})
    await _turn(orch, "second")
    text = _user_text(llm)
    assert "\ndocuments: " in text and "notes.md (2 chunks)" in text
    assert f"memory: {MEMORY_HEADING}" in text and "goal:" not in text
