"""Coding-agent routing must fire only for real code-authoring requests.

2026-11 finding: the owner typed "Its richard i created the software you are
wearing" and WITS ran the coding agent, because any message containing a bare
code noun ("software", "app", "class" ...) was routed there. Also covers the
removal of per-turn intent-analysis JSON from long-term memory.
"""

from types import SimpleNamespace

import pytest

from agents.routing_classifier import is_code_authoring_request
from agents.wits_control_center_agent import WitsControlCenterAgent
from core.config import WitsV3Config
from tests.agents.test_wcca_routing import DummyLLM, FakeMemoryManager

OWNER_SENTENCE = "Its richard i created the software you are wearing"

AUTHORING_REQUESTS = [
    "write a python script that renames files",
    "can you build me a small website for my reptile business",
    "fix the bug in parser.py",
    "please add an endpoint to the api",
    "refactor this function",
    "make an app that tracks pool chemistry",
    "could you write a function that parses dates?",
    "I need you to create a bash script for backups",
    "help me debug this javascript",
    "let's build a cli tool for renaming photos",
    "generate unit tests for utils.py",
    "convert this script to typescript",
    "Hey WITS, please implement a class for pool readings",
    "I want to scaffold a new project called poolpal",
    "update index.html so the header is blue",
    "port the module to rust",
    "write me a python script that sorts a list",
]

NOT_AUTHORING = [
    OWNER_SENTENCE,
    "what is an API?",
    "the app is slow today",
    "my class starts at 9",
    "I wrote that program years ago",
    "tell me about the WITS software",
    "how does the coding agent work?",
    "thanks, that script worked",
    "what changed in the workspace today?",
    "hi",
    "I built this app last year",
    "how does this program work?",
    "can you explain what a python decorator is?",
    "make me a sandwich",
    "add milk to the shopping list",
    "update me on the weather",
    "the website looks great",
    "did you write that function?",
    "is the api down?",
    "remind me to fix the pool pump tomorrow",
    "I created a new module yesterday",
    "what is the best programming language?",
    "write a story about a dragon",
    "",
]


@pytest.mark.parametrize("message", AUTHORING_REQUESTS)
def test_authoring_requests_are_detected(message):
    assert is_code_authoring_request(message) is True


@pytest.mark.parametrize("message", NOT_AUTHORING)
def test_non_authoring_messages_are_rejected(message):
    assert is_code_authoring_request(message) is False


class _RecordingMemoryManager(FakeMemoryManager):
    """Captures every segment the agent tries to persist."""

    def __init__(self):
        super().__init__({})
        self.segments = []

    async def add_segment(self, segment):
        self.segments.append(segment)
        return f"seg-{len(self.segments)}"


def _make_wcca(memory_manager=None):
    agent = WitsControlCenterAgent(
        agent_name="TestWCCA",
        config=WitsV3Config(),
        llm_interface=DummyLLM(),
        memory_manager=memory_manager or FakeMemoryManager({}),
        specialized_agents={
            "book_writing": SimpleNamespace(name="book_writing"),
            "coding": SimpleNamespace(name="coding"),
            "self_repair": SimpleNamespace(name="self_repair"),
        },
    )
    agent.has_enhanced_capabilities = False
    agent.meta_reasoning = None
    return agent


@pytest.mark.asyncio
async def test_select_agent_routes_authoring_request_to_coding():
    agent = await _make_wcca()._select_specialized_agent("make an app that tracks pool chemistry")
    assert agent.name == "coding"


@pytest.mark.asyncio
async def test_select_agent_ignores_owner_statement():
    assert await _make_wcca()._select_specialized_agent(OWNER_SENTENCE) is None


@pytest.mark.asyncio
async def test_turn_does_not_store_intent_analysis():
    memory = _RecordingMemoryManager()
    wcca = _make_wcca(memory)
    async for _ in wcca.run("what changed in the workspace today?", session_id="sess-mem"):
        pass
    types = [s.type for s in memory.segments]
    assert "USER_INPUT" in types  # proves the recorder is wired to store_memory
    assert "INTENT_ANALYSIS" not in types
