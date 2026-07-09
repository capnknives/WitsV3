"""Tests for orchestrator tool preflight / circuit-breaker guardrails."""

from agents.base_orchestrator_agent import BaseOrchestratorAgent


class _GuardHarness(BaseOrchestratorAgent):
    def _build_reasoning_prompt(self, state):
        return ""

    def _parse_reasoning_response(self, response):
        return {}


def _harness() -> _GuardHarness:
    return _GuardHarness.__new__(_GuardHarness)


def _state_with_docs():
    return {
        "documents_context": (
            "These user documents are ALREADY ingested:\n" "- report.md (5 chunks)"
        ),
        "tool_repeat_failures": {},
        "tool_total_failures": {},
    }


def test_blocks_read_file_when_documents_ingested():
    h = _harness()
    msg = h._preflight_tool_call("read_file", {}, _state_with_docs())
    assert msg is not None
    assert "document_search" in msg
    assert "Blocked read_file" in msg


def test_blocks_list_directory_when_documents_ingested():
    h = _harness()
    msg = h._preflight_tool_call("list_directory", {"directory_path": "."}, _state_with_docs())
    assert msg is not None
    assert "list_directory" in msg


def test_allows_document_search_with_ingested_docs():
    h = _harness()
    msg = h._preflight_tool_call(
        "document_search",
        {"query": "summary", "file_name": "report.md"},
        _state_with_docs(),
    )
    assert msg is None


def test_blocks_repeat_identical_failed_call():
    h = _harness()
    state = _state_with_docs()
    args = {"query": "x", "file_name": "report.md"}
    state["tool_repeat_failures"][h._tool_call_signature("document_search", args)] = 2
    msg = h._preflight_tool_call("document_search", args, state)
    assert msg is not None
    assert "Skipped repeat" in msg


def test_blocks_after_total_tool_failures():
    h = _harness()
    state = _state_with_docs()
    state["tool_total_failures"]["web_search"] = 3
    msg = h._preflight_tool_call("web_search", {"query": "news"}, state)
    assert msg is not None
    assert "failed 3 times" in msg


def test_records_failure_from_error_dict():
    h = _harness()
    state = _state_with_docs()
    h._record_tool_failure("document_search", {"query": "q"}, state)
    assert state["tool_total_failures"]["document_search"] == 1
    sig = h._tool_call_signature("document_search", {"query": "q"})
    assert state["tool_repeat_failures"][sig] == 1


def test_blocks_intent_analysis_in_orchestrator():
    h = _harness()
    msg = h._preflight_tool_call("intent_analysis", {"query": "game"}, _state_with_docs())
    assert msg is not None
    assert "Blocked intent_analysis" in msg


def test_blocks_repeat_read_history_on_save_goal():
    h = _harness()
    state = {
        "goal": "Save a log as exports/chat.txt",
        "observations": ["Tool read_conversation_history result: USER: hi"],
        "tool_repeat_failures": {},
        "tool_total_failures": {},
    }
    msg = h._preflight_tool_call("read_conversation_history", {}, state)
    assert msg is not None
    assert "Skipped repeat read_conversation_history" in msg
    assert "write_file" in msg


def test_save_file_path_from_goal():
    h = _harness()
    path = h._save_file_path_from_goal(
        "Save a log of our conversations as exports/chat_log_report_failure.txt"
    )
    assert path == "var/exports/chat_log_report_failure.txt"


def test_save_file_path_extensionless_importantissues01():
    h = _harness()
    path = h._save_file_path_from_goal("Save a copy of our conversation as importantissues01")
    assert path == "var/exports/importantissues01.txt"


def test_blocks_repeat_list_mcp_tools():
    h = _harness()
    state = {
        "goal": "Please open microsoft edge",
        "observations": [
            "Tool list_mcp_tools result: []",
            "Tool list_mcp_tools result: []",
        ],
        "tool_repeat_failures": {},
        "tool_total_failures": {},
    }
    msg = h._preflight_tool_call("list_mcp_tools", {}, state)
    assert msg is not None
    assert "Skipped repeat list_mcp_tools" in msg


def test_allows_read_file_for_codebase_intro_with_ingested_docs():
    h = _harness()
    state = _state_with_docs()
    state["goal"] = "What can you tell me about your codebase wits?"
    msg = h._preflight_tool_call("read_file", {"file_path": "README.md"}, state)
    assert msg is None


def test_blocks_web_search_on_pure_math_goal():
    h = _harness()
    state = {
        "goal": "what is the square-root of 75231",
        "observations": [],
        "tool_repeat_failures": {},
        "tool_total_failures": {},
    }
    msg = h._preflight_tool_call("web_search", {"query": "sqrt 75231"}, state)
    assert msg is not None
    assert "Blocked web_search" in msg


def test_blocks_document_search_on_web_lookup_goal():
    h = _harness()
    state = {
        "goal": "Look up Dragon Ball Advent Truth MUD and give a small report",
        "observations": [],
        "tool_repeat_failures": {},
        "tool_total_failures": {},
    }
    msg = h._preflight_tool_call("document_search", {"query": "dragon ball"}, state)
    assert msg is not None
    assert "Blocked document_search" in msg


def test_blocks_tools_after_lookup_search_done():
    h = _harness()
    state = {
        "goal": "Look up a game and report",
        "lookup_search_done": True,
        "observations": ["web_search results (base your answer on the SOURCES below):"],
        "tool_repeat_failures": {},
        "tool_total_failures": {},
    }
    msg = h._preflight_tool_call("document_search", {"query": "x"}, state)
    assert msg is not None
    assert "final_answer" in msg


def test_blocks_write_file_for_guest_role():
    h = _harness()
    state = {
        "user_role": "guest",
        "goal": "write a file",
        "tool_repeat_failures": {},
        "tool_total_failures": {},
    }
    msg = h._preflight_tool_call("write_file", {"file_path": "x.py", "content": "hi"}, state)
    assert msg is not None
    assert "guest" in msg.lower()


def test_allows_web_search_for_guest_role():
    h = _harness()
    state = {
        "user_role": "guest",
        "goal": "search news",
        "tool_repeat_failures": {},
        "tool_total_failures": {},
    }
    assert h._preflight_tool_call("web_search", {"query": "news"}, state) is None
