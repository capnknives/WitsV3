"""Tests for WitsControlCenterAgent intent routing.

Covers the July 7 2026 failures: document questions misrouted to casual
chat ("hi" substring-matched inside "things") or to clarification loops
because the intent analyzer had no knowledge of ingested documents.
"""

import inspect
from collections.abc import AsyncGenerator
from types import SimpleNamespace

import pytest

from agents.advanced_coding_agent import AdvancedCodingAgent
from agents.base_orchestrator_agent import BaseOrchestratorAgent
from agents.wits_control_center_agent import WitsControlCenterAgent
from core.config import WitsV3Config
from core.llm_interface import BaseLLMInterface
from core.schemas import ConversationHistory, StreamData


def _history_with_clarification(question: str, user_reply: str) -> ConversationHistory:
    history = ConversationHistory(session_id="sess-follow-up")
    history.add_message("user", "summarize the audit report you have access to.")
    history.add_message("assistant", question)
    history.add_message("user", user_reply)
    return history


def _history_after_casual_chat(user_reply: str) -> ConversationHistory:
    history = ConversationHistory(session_id="sess-casual")
    history.add_message("user", "hi there, how are you doing today my friend")
    history.add_message("assistant", "I'm doing well! How can I help you today?")
    history.add_message("user", user_reply)
    return history


class DummyLLM(BaseLLMInterface):
    def __init__(self):
        pass

    async def generate_text(self, prompt: str, **kwargs) -> str:
        return "dummy"

    async def stream_text(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        yield "dummy"

    async def get_embedding(self, text, model=None):
        return [0.0] * 8


class ScriptedLLM(BaseLLMInterface):
    """Returns a fixed response string for every call — used to control
    intent-analysis JSON in the clarification-merge tests below."""

    def __init__(self, response: str):
        self.response = response

    async def generate_text(self, prompt: str, **kwargs) -> str:
        return self.response

    async def stream_text(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        yield self.response

    async def get_embedding(self, text, model=None):
        return [0.0] * 8


class FakeMemoryManager:
    """Only what _get_document_inventory needs."""

    def __init__(self, doc_files):
        self._segments = [
            SimpleNamespace(metadata={"file_path": fp, "chunk_index": i})
            for fp, chunks in doc_files.items()
            for i in range(chunks)
        ]

    async def get_recent_memory(self, limit, filter_dict=None):
        return self._segments


@pytest.fixture
def wcca():
    agent = WitsControlCenterAgent(
        agent_name="TestWCCA",
        config=WitsV3Config(),
        llm_interface=DummyLLM(),
        memory_manager=FakeMemoryManager(
            {"Pleistocene_Megafauna_Audit_Report.md": 16, "proof_of_enrollment.v3 (2).pdf": 1}
        ),
    )
    # Force the plain routing path (no meta-reasoning shortcut)
    agent.has_enhanced_capabilities = False
    agent.meta_reasoning = None
    return agent


# ------------------------------------------------------- casual heuristic


def test_casual_word_requires_word_boundary(wcca):
    # "things" must not match casual word "hi" (regression: substring match)
    assert (
        wcca._is_casual_conversation("i've updated things, please check the results once more")
        is False
    )


def test_greetings_still_casual(wcca):
    assert wcca._is_casual_conversation("hi there, how are you doing today my friend") is True
    assert wcca._is_casual_conversation("thanks!") is True


def test_short_follow_up_after_clarification_is_not_casual(wcca):
    history = _history_with_clarification("Which audit report?", "the megafauna one")
    assert wcca._is_conversation_follow_up("the megafauna one", history) is True
    assert wcca._is_casual_conversation("the megafauna one", history) is False


def test_short_affirmative_after_task_context_is_follow_up(wcca):
    history = _history_with_clarification(
        "I can help with that. Which file should I summarize?", "yes"
    )
    assert wcca._is_conversation_follow_up("yes", history) is True


def test_short_reply_after_casual_chat_can_still_be_casual(wcca):
    history = _history_after_casual_chat("thanks!")
    assert wcca._is_conversation_follow_up("thanks!", history) is False
    assert wcca._is_casual_conversation("thanks!", history) is True


@pytest.mark.asyncio
async def test_summarize_it_follow_up_routes_to_orchestrator(wcca):
    history = _history_with_clarification("Which audit report?", "summarize it")
    intent = await wcca._analyze_user_intent("summarize it", history)
    assert intent["suggested_response"] == "orchestrator"
    assert intent["requires_tools"] is True


@pytest.mark.asyncio
async def test_yes_after_audit_question_routes_to_orchestrator(wcca):
    history = _history_with_clarification(
        "Which audit report would you like me to summarize?", "yes"
    )
    intent = await wcca._analyze_user_intent("yes", history)
    assert intent["suggested_response"] == "orchestrator"
    assert intent["requires_tools"] is True


@pytest.mark.asyncio
async def test_conversation_intent_overridden_for_follow_up_web_search(wcca):
    wcca.orchestrator_agent = MockOrchestrator()
    history = ConversationHistory(session_id="sess-web-follow-up")
    history.add_message("user", "What famous musician died on june 14th 2026?")
    history.add_message(
        "assistant",
        "I don't have that in my training data. Would you like me to look it up?",
    )
    history.add_message("user", "yes")

    intent = {
        "type": "conversation",
        "complexity": "simple",
        "requires_tools": False,
        "suggested_response": "direct",
    }
    results = await _collect_handler_results(wcca, intent, "yes", history)
    assert any(r.type == "result" and r.content == "orchestrator handled it" for r in results)


async def _collect_handler_results(wcca, intent, user_input, conversation_history=None):
    results = []
    async for stream_data in wcca._handle_intent_response(
        intent, user_input, conversation_history, "test-session"
    ):
        results.append(stream_data)
    return results


# ---------------------------------------------- specialized-agent selection


@pytest.fixture
def wcca_with_specialists():
    """Regression coverage for the 2026-07-08 live-chat finding: 'find and
    fix bugs in the codebase' routed to the coding agent instead of
    self-repair, because plain substring matching let "codebase" trip the
    "code" keyword before the repair keywords were ever checked."""
    agent = WitsControlCenterAgent(
        agent_name="TestWCCA",
        config=WitsV3Config(),
        llm_interface=DummyLLM(),
        memory_manager=FakeMemoryManager({}),
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
async def test_bug_report_mentioning_codebase_routes_to_self_repair(wcca_with_specialists):
    agent = await wcca_with_specialists._select_specialized_agent(
        "find and fix any bugs in the wits v3 codebase"
    )
    assert agent.name == "self_repair"


@pytest.mark.asyncio
async def test_codebase_word_does_not_false_positive_match_code(wcca_with_specialists):
    # "codebase" must not substring-match the "code" keyword.
    agent = await wcca_with_specialists._select_specialized_agent(
        "tell me about the codebase structure"
    )
    assert agent is None


@pytest.mark.asyncio
async def test_description_word_does_not_false_positive_match_script(wcca_with_specialists):
    # "description" contains "script" as a raw substring (de-SCRIPT-ion).
    agent = await wcca_with_specialists._select_specialized_agent(
        "give me a longer description of this"
    )
    assert agent is None


@pytest.mark.asyncio
async def test_plain_coding_request_still_routes_to_coding(wcca_with_specialists):
    agent = await wcca_with_specialists._select_specialized_agent(
        "write me a python script that sorts a list"
    )
    assert agent.name == "coding"


@pytest.mark.asyncio
async def test_story_request_routes_to_book_writing(wcca_with_specialists):
    agent = await wcca_with_specialists._select_specialized_agent("write a story about a dragon")
    assert agent.name == "book_writing"


@pytest.mark.asyncio
async def test_live_transcript_story_request_routes_to_book_writing(wcca_with_specialists):
    """2026-07-08 finding: this exact live request matched none of the
    original multi-word story phrases (the comma after "story" broke the
    "story about" substring match), fell through to the generic
    orchestrator, and got a fabricated "has been created" reply with
    nothing written to disk."""
    agent = await wcca_with_specialists._select_specialized_agent(
        "Please write the equivalent to a 100 page story, about a knight in a "
        "medieval town that develops powers"
    )
    assert agent.name == "book_writing"


def test_needs_story_writing_matches_live_finding(wcca_with_specialists):
    assert (
        wcca_with_specialists._needs_story_writing(
            "Please write the equivalent to a 100 page story, about a knight"
        )
        is True
    )


def test_needs_story_writing_ignores_unrelated_chat(wcca_with_specialists):
    assert wcca_with_specialists._needs_story_writing("what's the story with this bug?") is False


# -------------------------------------------- unambiguous bug-hunt override


def test_needs_self_repair_matches_the_live_finding(wcca_with_specialists):
    assert wcca_with_specialists._needs_self_repair("find and fix any bugs in your code.") is True


def test_needs_self_repair_ignores_unrelated_chat(wcca_with_specialists):
    assert wcca_with_specialists._needs_self_repair("hey, how's it going?") is False


# -------- transcript chat_export_901b8182: "Run self repair" hallucination


@pytest.mark.parametrize(
    "message",
    [
        "Run self repair",
        "run self-repair",
        "please run a self repair",
        "start self-repair now",
        "self repair",
        "repair yourself",
    ],
)
def test_run_self_repair_command_matches_self_repair(wcca_with_specialists, message):
    """The literal "Run self repair" command matched none of the bug-hunt
    phrases and was flagged casual (3 words), so the model fabricated a
    "self-repair complete" report without the agent ever running."""
    assert wcca_with_specialists._needs_self_repair(message) is True


@pytest.mark.parametrize(
    "message",
    [
        "Run self repair",
        "fix the login bug",
        "search my notes for the invoice",
        "summarize it",
    ],
)
def test_short_imperative_command_is_not_casual(wcca, message):
    """Terse imperative commands must bypass the "short messages are casual"
    length rule so they reach real routing instead of the chat path."""
    assert wcca._is_casual_conversation(message) is False


@pytest.mark.asyncio
async def test_run_self_repair_reaches_self_repair_agent(wcca_with_specialists):
    """End-to-end: "Run self repair" must invoke the self-repair agent rather
    than the casual-chat path that previously confabulated a success report."""
    self_repair_agent = _FakeSpecializedAgent("self_repair")
    wcca_with_specialists.specialized_agents["self_repair"] = self_repair_agent

    streams = [
        item
        async for item in wcca_with_specialists.run("Run self repair", session_id="sess-selfrepair")
    ]
    assert self_repair_agent.received_input == "Run self repair"
    assert any("self_repair ran" in s.content for s in streams)


# --- transcript chat_export_901b8182: action-confabulation guard on chat path


def test_action_confabulation_detects_fabricated_self_repair_report(wcca):
    """The exact fabricated reply from the transcript must be recognized as a
    false claim of having performed a system action."""
    fabricated = (
        "**System Update: Self-Repair Initiated**\n"
        "After running a self-repair cycle, I have re-evaluated my internal "
        "state and updated my knowledge graph.\n"
        "All system checks indicate that my core directives remain intact. "
        "My memory is clean."
    )
    assert wcca._looks_like_action_confabulation(fabricated) is True


@pytest.mark.parametrize(
    "text",
    [
        "I can run a self-repair cycle if you'd like — just say the word.",
        "Self-repair scans logs and failing tests, then applies verified fixes.",
        "Sure! How can I help you today?",
        "I'd be happy to fix that bug once you point me at the file.",
    ],
)
def test_action_confabulation_ignores_capability_descriptions(wcca, text):
    """Naming a capability (without claiming it already ran) must not trip
    the guard."""
    assert wcca._looks_like_action_confabulation(text) is False


class _FakeSpecializedAgent:
    """Records that .run() was actually invoked, for the intent-override test below."""

    def __init__(self, name):
        self.name = name
        self.received_input = None

    async def run(self, user_input, conversation_history=None, session_id=None, **kwargs):
        self.received_input = user_input
        yield StreamData(type="result", content=f"{self.name} ran", source=self.name)


@pytest.mark.asyncio
async def test_bug_hunt_request_reaches_self_repair_despite_clarification_classification(
    wcca_with_specialists,
):
    """2026-07-08 finding: the LLM intent classifier called this a
    clarification_question, which returns before specialized-agent routing
    is ever considered — _needs_self_repair must force delegation anyway."""
    self_repair_agent = _FakeSpecializedAgent("self_repair")
    wcca_with_specialists.specialized_agents["self_repair"] = self_repair_agent

    intent_analysis = {
        "type": "clarification_question",
        "complexity": "simple",
        "suggested_response": "clarification",
        "requires_tools": False,
        "clarification_question": "Could you clarify what you'd like me to check?",
    }
    streams = [
        item
        async for item in wcca_with_specialists._handle_intent_response(
            intent_analysis, "find and fix any bugs in your code.", None, "sess-1"
        )
    ]
    assert self_repair_agent.received_input == "find and fix any bugs in your code."
    assert any("self_repair ran" in s.content for s in streams)


@pytest.mark.asyncio
async def test_story_request_reaches_book_writing_despite_conversation_classification(
    wcca_with_specialists,
):
    """2026-07-08 live-chat finding: the LLM intent step classified this
    story request as ordinary conversation, which returns before
    specialized-agent routing is ever considered — _needs_story_writing must
    force delegation to book_writing anyway, the same way _needs_self_repair
    already does for bug-hunt requests. Before this fix, the generic
    orchestrator handled the turn and fabricated a "has been created" reply
    without writing anything to disk."""
    book_agent = _FakeSpecializedAgent("book_writing")
    wcca_with_specialists.specialized_agents["book_writing"] = book_agent

    intent_analysis = {
        "type": "conversation",
        "complexity": "simple",
        "suggested_response": "direct",
        "requires_tools": False,
    }
    live_request = (
        "Please write the equivalent to a 100 page story, about a knight in a "
        "medieval town that develops powers like dbz characters and takes over "
        "the surrounding area. Save the story as TheBigStory01"
    )
    streams = [
        item
        async for item in wcca_with_specialists._handle_intent_response(
            intent_analysis, live_request, None, "sess-story"
        )
    ]
    assert book_agent.received_input == live_request
    assert any("book_writing ran" in s.content for s in streams)


@pytest.mark.asyncio
async def test_continuation_followup_resumes_previous_specialized_agent(wcca_with_specialists):
    """2026-07-08 finding: "Okay, so make it." after a story request carries
    no keywords of its own, so keyword-based specialized-agent selection
    finds nothing — the session must remember which agent handled the
    previous turn and resume it directly instead of falling through to
    casual chat or an unrelated fresh orchestrator run."""
    book_agent = _FakeSpecializedAgent("book_writing")
    wcca_with_specialists.specialized_agents["book_writing"] = book_agent
    session_id = "sess-book-continuation"

    async for _ in wcca_with_specialists.run("write a story about a dragon", session_id=session_id):
        pass
    assert wcca_with_specialists._active_specialized_agent[session_id] == "book_writing"

    book_agent.received_input = None
    streams = [
        item
        async for item in wcca_with_specialists._handle_intent_response(
            {
                "type": "conversation",
                "complexity": "simple",
                "suggested_response": "direct",
                "requires_tools": False,
            },
            "Okay, so make it.",
            None,
            session_id,
        )
    ]
    assert book_agent.received_input == "Okay, so make it."
    assert any("book_writing ran" in s.content for s in streams)


# --------------------------------------- conversation-history-aware follow-ups


@pytest.mark.asyncio
async def test_clarification_question_records_pending_state(wcca_with_specialists):
    """_handle_intent_response's clarification_question branch must record
    the original request so a later reply can be merged with it.

    Uses a request that does NOT match _SELF_REPAIR_SIGNALS — that override
    runs unconditionally before the clarification_question branch and would
    otherwise reroute this to specialized-agent selection instead (covered
    separately by the _needs_self_repair tests above).
    """
    intent_analysis = {
        "type": "clarification_question",
        "complexity": "simple",
        "suggested_response": "clarification",
        "requires_tools": False,
        "clarification_question": "Which project?",
    }
    original_request = "help me improve something in my project."
    async for _ in wcca_with_specialists._handle_intent_response(
        intent_analysis, original_request, None, "sess-pending"
    ):
        pass
    assert wcca_with_specialists._pending_clarifications["sess-pending"] == original_request


@pytest.mark.asyncio
async def test_followup_reply_merges_with_pending_clarification_and_reaches_self_repair(
    wcca_with_specialists,
):
    """End-to-end regression for the actual July 8 live-chat bug: a bare
    follow-up reply ("Specifically the wits v3 codebase") is meaningless on
    its own and the LLM classifier really did call it "conversation" — the
    merge with the pending original request must rescue it regardless."""
    agent = wcca_with_specialists
    self_repair_agent = _FakeSpecializedAgent("self_repair")
    agent.specialized_agents["self_repair"] = self_repair_agent
    session_id = "sess-followup"
    agent._pending_clarifications[session_id] = "find and fix any bugs in your code."
    agent.llm_interface = ScriptedLLM(
        '{"type": "conversation", "complexity": "simple", "suggested_response": "direct"}'
    )

    streams = [
        item async for item in agent.run("Specifically the wits v3 codebase", session_id=session_id)
    ]

    assert session_id not in agent._pending_clarifications
    assert self_repair_agent.received_input == (
        "find and fix any bugs in your code.\nSpecifically the wits v3 codebase"
    )
    assert any("self_repair ran" in s.content for s in streams)


@pytest.mark.asyncio
async def test_casual_reply_after_clarification_is_not_merged(wcca_with_specialists):
    """A topic change ("thanks") after a clarifying question should just
    clear the stale pending state, not get prefixed onto unrelated chat or
    misrouted to the self-repair agent it was never about."""
    agent = wcca_with_specialists
    self_repair_agent = _FakeSpecializedAgent("self_repair")
    agent.specialized_agents["self_repair"] = self_repair_agent
    session_id = "sess-casual"
    agent._pending_clarifications[session_id] = "find and fix any bugs in your code."

    async for _ in agent.run("thanks!", session_id=session_id):
        pass

    assert session_id not in agent._pending_clarifications
    assert self_repair_agent.received_input is None


@pytest.mark.asyncio
async def test_remember_command_clears_pending_clarification_without_merging(wcca_with_specialists):
    agent = wcca_with_specialists
    session_id = "sess-remember"
    agent._pending_clarifications[session_id] = "find and fix any bugs in your code."

    async for _ in agent.run("remember my favorite color is blue", session_id=session_id):
        pass

    assert session_id not in agent._pending_clarifications


# ------------------------------------------------------- document routing


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        # The three phrasings that failed on July 7 2026
        "i've updated things, please check the audit again",
        "summarize the audit report you have access to.",
        "I shared Pleistocene_Megafauna_Audit_Report.md with you, summarize it.",
    ],
)
async def test_document_mentions_route_to_orchestrator(wcca, message):
    intent = await wcca._analyze_user_intent(message, None)
    assert intent["suggested_response"] == "orchestrator"
    assert intent["requires_tools"] is True


# ------------------------------------------------- web-search routing


@pytest.mark.parametrize(
    "message",
    [
        "What famous musician died of june 14th 2026?",  # the reported failure
        "look it up",  # explicit follow-up command
        "who won the world cup this year?",
        "what's the latest news on Ollama?",
        "search the web for python 3.14 release date",
        "what's the weather in Seattle right now?",
    ],
)
def test_current_info_questions_need_web_search(wcca, message):
    assert wcca._needs_web_search(message) is True


@pytest.mark.parametrize(
    "message",
    [
        "hi there, how are you doing today my friend",  # 'today' must NOT trigger
        "thanks, that was helpful",
        "what can you do?",
        "write me a python function to reverse a list",
    ],
)
def test_ordinary_chat_does_not_need_web_search(wcca, message):
    assert wcca._needs_web_search(message) is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "What famous musician died of june 14th 2026?",
        "look it up",
    ],
)
async def test_current_info_routes_to_orchestrator(wcca, message):
    intent = await wcca._analyze_user_intent(message, None)
    assert intent["suggested_response"] == "orchestrator"
    assert intent["requires_tools"] is True


# ------------------------------------------------------- save-to-file routing


@pytest.mark.parametrize(
    "message",
    [
        "Please save this conversation to a file",
        "Save a log of our conversations as a file",
        "write the story to disk as goku.txt",
    ],
)
def test_save_to_file_needs_orchestrator(wcca, message):
    assert wcca._needs_file_write(message) is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "Please save this conversation to exports/chat.txt",
        "Save a log of our conversations as a file",
    ],
)
async def test_save_to_file_routes_to_orchestrator(wcca, message):
    intent = await wcca._analyze_user_intent(message, None)
    assert intent["suggested_response"] == "orchestrator"
    assert intent["requires_tools"] is True
    assert "write_file" in intent["notes"] or "save" in intent["notes"].lower()


# ------------------------------------------------------- intent parsing


def test_goal_defined_routes_to_orchestrator(wcca):
    parsed = wcca._parse_intent_response(
        '{"type": "goal_defined", "confidence": 0.9, "goal_statement": "do the thing"}'
    )
    assert parsed["suggested_response"] == "orchestrator"
    assert parsed["requires_tools"] is True
    assert parsed["complexity"] == "moderate"


def test_direct_response_intent_metadata(wcca):
    parsed = wcca._parse_intent_response(
        '{"type": "direct_response", "direct_response": "Hey there!"}'
    )
    assert parsed["suggested_response"] == "direct"
    assert parsed["requires_tools"] is False
    assert parsed["direct_response"] == "Hey there!"


def test_clarification_intent_metadata(wcca):
    parsed = wcca._parse_intent_response(
        '{"type": "clarification_question", "clarification_question": "Which file?"}'
    )
    assert parsed["suggested_response"] == "clarification"
    assert parsed["requires_tools"] is False
    assert parsed["clarification_question"] == "Which file?"


# ---------------------------------------- intent handler (no double LLM)


class TrackingLLM(DummyLLM):
    """Records generate_text calls so handler tests can assert zero extra LLM work."""

    def __init__(self):
        self.calls = []

    async def generate_text(self, prompt: str, **kwargs) -> str:
        self.calls.append(prompt)
        return "unexpected llm output"


class MockOrchestrator:
    async def run(self, user_input, conversation_history=None, session_id=None, **kwargs):
        yield SimpleNamespace(type="result", content="orchestrator handled it", source="mock")


@pytest.mark.asyncio
async def test_handle_direct_response_uses_intent_text_without_llm(wcca):
    tracking = TrackingLLM()
    wcca.llm_interface = tracking
    intent = wcca._parse_intent_response(
        '{"type": "direct_response", "direct_response": "Hello from intent JSON"}'
    )
    results = await _collect_handler_results(wcca, intent, "thanks!")
    assert not tracking.calls
    assert any(r.type == "result" and r.content == "Hello from intent JSON" for r in results)


@pytest.mark.asyncio
async def test_handle_clarification_uses_intent_question_without_llm(wcca):
    tracking = TrackingLLM()
    wcca.llm_interface = tracking
    intent = wcca._parse_intent_response(
        '{"type": "clarification_question", "clarification_question": "Which audit report?"}'
    )
    results = await _collect_handler_results(wcca, intent, "summarize it")
    assert not tracking.calls
    assert any(r.type == "result" and r.content == "Which audit report?" for r in results)


@pytest.mark.asyncio
async def test_direct_response_overridden_when_web_search_needed(wcca):
    tracking = TrackingLLM()
    wcca.llm_interface = tracking
    wcca.orchestrator_agent = MockOrchestrator()
    intent = wcca._parse_intent_response(
        '{"type": "direct_response", "direct_response": "From memory: nobody"}'
    )
    results = await _collect_handler_results(
        wcca, intent, "What famous musician died on june 14th 2026?"
    )
    assert not tracking.calls
    assert any(r.type == "result" and r.content == "orchestrator handled it" for r in results)


@pytest.mark.asyncio
async def test_conversation_still_calls_llm_once(wcca):
    tracking = TrackingLLM()
    wcca.llm_interface = tracking
    intent = {
        "type": "conversation",
        "complexity": "simple",
        "requires_tools": False,
        "suggested_response": "direct",
    }
    await _collect_handler_results(wcca, intent, "hi there!")
    assert len(tracking.calls) == 1
    assert "casual conversation" in tracking.calls[0].lower()


# ------------------------------------------------------- documents context


def test_documents_context_lists_files(wcca):
    ctx = wcca._documents_context({"a_report.md": 3})
    assert "a_report.md (3 chunks)" in ctx
    assert "ALREADY ingested" in ctx


def test_documents_context_empty():
    assert "No user documents" in WitsControlCenterAgent._documents_context({})


# ---------------------------------------- agent delegation contract


# --- transcript chat_export_901b8182: "what did guest X chat about" routing


def test_named_guest_chat_history_routes_to_guest_audit(tmp_path, monkeypatch):
    """2026-07-08 finding: "What did the test user Sean chat with you about?"
    matched none of the guest-audit signals ("what did tester" expects the
    literal "tester") nor the profile person-query signals, so it fell to the
    generic orchestrator and looped to max_iterations. Naming a registered
    guest plus a conversation verb must route to guest_audit_summary."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    from core.guest_access import GuestRegistry

    from agents.wcca_routing_mixin import OrchestratorRoutingMixin

    class _Probe(OrchestratorRoutingMixin):
        pass

    reg = GuestRegistry()
    reg.register_or_update(display_name="Sean", device_id="device-sean-0001")

    probe = _Probe()
    msg = "What did the test user Sean chat with you about?"
    assert probe._needs_guest_chat_history(msg) is True
    assert probe._needs_guest_audit_review(msg) is True
    assert probe._extract_guest_name_for_profile_query(msg) == "Sean"


def test_guest_chat_history_requires_a_known_guest_name(tmp_path, monkeypatch):
    """The conversation verb alone (no registered guest named) must not route
    an ordinary question to the guest audit log."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()

    from agents.wcca_routing_mixin import OrchestratorRoutingMixin

    class _Probe(OrchestratorRoutingMixin):
        pass

    probe = _Probe()
    assert probe._needs_guest_chat_history("what did you chat about with the model?") is False


def test_orchestrator_and_coding_agents_use_user_input_param():
    """WCCA delegates with user_input=; all BaseAgent subclasses must accept it."""
    orch_params = inspect.signature(BaseOrchestratorAgent.run).parameters
    coding_params = inspect.signature(AdvancedCodingAgent.run).parameters
    assert "user_input" in orch_params
    assert "goal" not in orch_params
    assert "user_input" in coding_params
    assert "request" not in coding_params
