"""Red team RT-11: unit tests for the private-read fact and the fail-closed outbound rule.

Every registered tool must be classified in ``TOOL_CLASSES``; after a private read only
exact, known local-only names may run.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.tool_registry import ToolRegistry
from core.untrusted_content import (
    LOCAL_ONLY_TOOLS,
    PRIVATE_READ_TOOLS,
    SANDBOX_LOCAL_TOOLS,
    TOOL_CLASSES,
    WRITE_BLOCKED,
    UntrustedWriteGate,
    python_sandboxed,
)


def _after(*names: str) -> UntrustedWriteGate:
    # Reason: not docker, so these tests never depend on the checkout's config.yaml.
    gate = UntrustedWriteGate(True, sandboxed=lambda: False)
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


_CODE_RUNNERS = ("python_execute", "run_test_suite", "apply_code_fix")


@pytest.mark.parametrize("name", sorted(LOCAL_ONLY_TOOLS))
def test_local_only_tools_that_run_no_code_still_run_after_a_conversation_read(name):
    # Reason: the private fact alone never refuses a write or local tool (RT-3 decides
    # writes, and read_conversation_history is no RT-3 source).
    assert name not in _CODE_RUNNERS
    assert _after("read_conversation_history").block_reason(name) is None


@pytest.mark.parametrize("mode", ["docker", "Docker", "subprocess", "off", "Docker "])
@pytest.mark.parametrize("name", _CODE_RUNNERS)
def test_code_running_tools_follow_the_sandbox_in_each_mode(name, mode):
    live = SimpleNamespace(security=SimpleNamespace(sandbox_mode=mode))
    gate = UntrustedWriteGate(True, sandboxed=lambda: python_sandboxed(live))
    gate.note_dispatch("read_conversation_history")
    # Reason: only python_execute has a network-less path, and only in docker mode;
    # run_test_suite runs pytest on the host and run_pytest_sandboxed keeps the network.
    local = name in SANDBOX_LOCAL_TOOLS and mode.lower() == "docker"
    assert (gate.block_reason(name) is None) is local
    runner = UntrustedWriteGate(True, sandboxed=lambda: python_sandboxed(live))
    runner.note_dispatch(name)
    assert (runner.private_read_by is None) is local


def test_gate_reads_the_mode_at_each_decision():
    live = {"mode": "docker"}
    gate = UntrustedWriteGate(True, sandboxed=lambda: live["mode"] == "docker")
    gate.note_dispatch("list_directory")
    assert gate.block_reason("python_execute") is None
    live["mode"] = "subprocess"
    assert gate.block_reason("python_execute").startswith("Blocked:")


def test_python_sandboxed_uses_load_config_like_the_tool(monkeypatch):
    live = SimpleNamespace(security=SimpleNamespace(sandbox_mode="docker"))
    monkeypatch.setattr("core.config.load_config", lambda *a, **k: live)
    assert python_sandboxed() is True
    live.security.sandbox_mode = "subprocess"
    assert python_sandboxed() is False


@pytest.mark.parametrize("name", ["unknown_tool", "Read_File", "mcp_x", 7, None, ["x"], ""])
def test_unclassified_names_count_as_a_private_read(name):
    gate = _after(name)
    assert gate.private_read_by
    assert gate.block_reason("web_search").startswith("Blocked:")


@pytest.mark.parametrize("name", ["unknown_tool", "Web_Search", 7, None])
def test_unknown_names_get_a_truthful_block_message(name):
    reason = _after("read_conversation_history").block_reason(name)
    assert reason.startswith("Blocked:") and "owner" in reason
    assert "send text off" not in reason


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


@pytest.mark.parametrize(
    "mode,expected",
    [("docker", True), ("DOCKER", True), (" DOCKER\n", False), ("Docker ", False)]
    + [("subprocess", False), ("off", False), (None, False), ("dockerish", False), (42, False)],
)
def test_python_sandboxed_reads_the_mode(mode, expected):
    assert (
        python_sandboxed(SimpleNamespace(security=SimpleNamespace(sandbox_mode=mode))) is expected
    )


@pytest.mark.parametrize("config", [object(), SimpleNamespace(security=None)])
def test_python_sandboxed_fails_closed_without_a_mode(config):
    assert python_sandboxed(config) is False


def test_python_sandboxed_gate_never_relaxes_rt3():
    gate = UntrustedWriteGate(True, sandboxed=lambda: True)
    gate.note_dispatch("web_search")
    assert gate.block_reason("python_execute") == WRITE_BLOCKED.format(name="python_execute")


def test_python_sandboxed_needs_a_real_true():
    gate = UntrustedWriteGate(True, sandboxed=lambda: "yes")
    gate.note_dispatch("list_directory")
    assert gate.block_reason("python_execute")
