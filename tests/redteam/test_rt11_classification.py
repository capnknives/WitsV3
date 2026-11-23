"""Red team RT-11: unit tests for the private-read fact and the fail-closed outbound rule.

Every registered tool must be classified in ``TOOL_CLASSES``; after a private read only
exact, known local-only names may run.
"""

from __future__ import annotations

import pytest

from core.tool_registry import ToolRegistry
from core.untrusted_content import (
    LOCAL_ONLY_TOOLS,
    PRIVATE_READ_TOOLS,
    TOOL_CLASSES,
    WRITE_BLOCKED,
    UntrustedWriteGate,
)


def _after(*names: str) -> UntrustedWriteGate:
    gate = UntrustedWriteGate(True)
    for name in names:
        gate.note_dispatch(name)
    return gate


def test_every_registered_tool_is_classified():
    missing = sorted(set(ToolRegistry().tools) - set(TOOL_CLASSES))
    assert missing == [], f"classify these in core/untrusted_content.py: {missing}"


def test_classes_are_known_values():
    assert set(TOOL_CLASSES.values()) <= {"private", "outbound", "both", "neither"}


@pytest.mark.parametrize(
    "name", ["python_execute", "diagnose_log_errors", "list_directory", "analyze_conversation"]
)
def test_more_private_readers_set_the_flag(name):
    assert name in PRIVATE_READ_TOOLS
    assert _after(name).private_read_by == name


def test_mcp_tool_counts_as_a_private_read():
    assert _after("mcp_fs_read").private_read_by == "mcp_fs_read"


@pytest.mark.parametrize(
    "name",
    ["web_search", "search_mcp_tools", "ask_claude", "python_execute", "mcp_x", "unknown_tool"],
)
def test_outbound_mcp_and_unknown_are_refused_after_a_private_read(name):
    reason = _after("read_conversation_history").block_reason(name)
    assert reason and reason.startswith("Blocked:")


@pytest.mark.parametrize("name", ["Web_Search", "READ_FILE", "web_search ", 7, None, ["x"]])
def test_wrong_case_and_non_string_names_are_refused(name):
    assert _after("read_conversation_history").block_reason(name)


@pytest.mark.parametrize("name", sorted(LOCAL_ONLY_TOOLS - {"read_file"}))
def test_local_only_tools_still_run_after_a_conversation_read(name):
    # Reason: the private fact alone never refuses a write or local tool (RT-3 decides
    # writes, and read_conversation_history is no RT-3 source).
    assert _after("read_conversation_history").block_reason(name) is None


def test_rt3_message_wins_when_both_rules_refuse():
    reason = _after("read_file").block_reason("mcp_x")
    assert reason == WRITE_BLOCKED.format(name="mcp_x")


def test_rt11_message_for_what_rt3_allows():
    reason = _after("read_file").block_reason("web_search")
    assert "private local content" in reason


def test_search_then_read_leaves_nothing_blocked_before_the_read():
    gate = _after("web_search", "search_mcp_tools")
    assert gate.private_read_by is None
    assert gate.block_reason("read_file") is None


def test_setting_off_blocks_nothing():
    gate = UntrustedWriteGate(False)
    gate.note_dispatch("read_file")
    assert gate.block_reason("web_search") is None
    assert gate.block_reason("mcp_x") is None
