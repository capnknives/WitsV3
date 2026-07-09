"""Tests for the self-repair tools (log diagnosis, test running, verified fixes, restart)."""

from unittest.mock import AsyncMock, patch

import pytest

from core.knowledge_log import KnowledgeLogStore
from tools.self_repair_tools import (
    ApplyCodeFixTool,
    DiagnoseLogErrorsTool,
    RestartAppTool,
    RunTestSuiteTool,
    parse_traceback_issues,
)

SAMPLE_LOG = """\
2026-07-08 03:00:00,000 - WitsV3.Main - INFO - startup complete
2026-07-08 03:00:01,000 - WitsV3.Something - ERROR - Something failed: division by zero
Traceback (most recent call last):
  File "c:\\Users\\capta\\source\\repos\\capnknives\\WitsV3-claude\\agents\\base_agent.py", line 42, in bar
    return 1/0
ZeroDivisionError: division by zero
2026-07-08 03:00:02,000 - WitsV3.Other - ERROR - Unrelated bare error with no traceback
"""


def test_parse_traceback_issues_extracts_traceback_with_file_and_line():
    # SAMPLE_LOG deliberately uses a sibling worktree path (.../WitsV3-claude/...)
    # — the parser must remap package-relative suffixes onto this checkout.
    issues = parse_traceback_issues(SAMPLE_LOG, max_issues=5)
    actionable = [i for i in issues if i["actionable"]]
    assert actionable, "expected at least one actionable (file-resolvable) issue"
    assert actionable[0]["file"] == "agents/base_agent.py"
    assert actionable[0]["line"] == 42
    assert "ZeroDivisionError" in actionable[0]["message"]


def test_relative_to_project_remaps_sibling_worktree_paths():
    from tools.self_repair_tools import _relative_to_project

    foreign = r"c:\Users\capta\source\repos\capnknives\WitsV3-claude\agents\base_agent.py"
    assert _relative_to_project(foreign) == "agents/base_agent.py"
    assert _relative_to_project(r"C:\Python310\Lib\logging\__init__.py") is None


def test_parse_traceback_issues_includes_bare_error_lines_as_non_actionable():
    issues = parse_traceback_issues(SAMPLE_LOG, max_issues=5)
    non_actionable = [i for i in issues if not i["actionable"]]
    assert any("Unrelated bare error" in i["message"] for i in non_actionable)


def test_parse_traceback_issues_dedupes_identical_signatures():
    dup_log = SAMPLE_LOG + "\n" + SAMPLE_LOG
    issues = parse_traceback_issues(dup_log, max_issues=10)
    tracebacks = [i for i in issues if i.get("kind") == "traceback"]
    assert len(tracebacks) == 1


def test_parse_traceback_issues_respects_max_issues():
    big_log = "\n".join(
        f"2026-07-08 03:00:0{i},000 - WitsV3.X - ERROR - distinct error {i}" for i in range(9)
    )
    issues = parse_traceback_issues(big_log, max_issues=2)
    assert len(issues) == 2


@pytest.mark.asyncio
async def test_diagnose_log_errors_tool_reads_real_log_file(tmp_path, monkeypatch):
    tool = DiagnoseLogErrorsTool()
    log_file = tmp_path / "witsv3.log"
    log_file.write_text(SAMPLE_LOG, encoding="utf-8")
    tool.log_path = log_file
    tool.knowledge_log = KnowledgeLogStore(tmp_path / "knowledge_log.json")

    result = await tool.execute(lines=100, max_issues=5)
    assert result["success"] is True
    assert result["count"] >= 1

    logged = tool.knowledge_log._load()
    assert logged["errors"], "expected the scanned issues to be recorded"


@pytest.mark.asyncio
async def test_diagnose_log_errors_tool_handles_missing_log():
    tool = DiagnoseLogErrorsTool()
    tool.log_path = tool.log_path.parent / "does_not_exist_xyz.log"
    result = await tool.execute()
    assert result["success"] is True
    assert result["issues"] == []


@pytest.mark.asyncio
async def test_run_test_suite_tool_wraps_run_pytest():
    tool = RunTestSuiteTool()
    with patch(
        "tools.self_repair_tools.run_pytest",
        new=AsyncMock(return_value=(True, "5 passed")),
    ) as mock_run:
        result = await tool.execute(test_path="tests/tools/test_math_tool.py")
    mock_run.assert_awaited_once()
    assert result["passed"] is True
    assert "passed" in result["message"]


@pytest.mark.asyncio
async def test_apply_code_fix_tool_wraps_apply_verified_edit():
    tool = ApplyCodeFixTool()
    from core.safe_code_editor import EditResult

    fake_result = EditResult(
        success=True,
        file_path="agents/base_agent.py",
        message="Edit applied and verified.",
        test_output="1 passed",
        committed=True,
        commit_sha="abc1234",
    )
    with patch(
        "tools.self_repair_tools.apply_verified_edit",
        new=AsyncMock(return_value=fake_result),
    ) as mock_apply:
        result = await tool.execute(
            file_path="agents/base_agent.py", new_content="# fixed", reason="fix the bug"
        )
    mock_apply.assert_awaited_once()
    assert result["success"] is True
    assert result["commit_sha"] == "abc1234"


@pytest.mark.asyncio
async def test_restart_app_tool_schedules_relaunch_without_executing_it():
    tool = RestartAppTool()
    with patch("tools.self_repair_tools._relaunch") as mock_relaunch:
        # execute() clamps delay_seconds to a minimum of 0.5s so a restart
        # can never fire before the current response finishes streaming.
        result = await tool.execute(delay_seconds=0.01, reason="test restart")
        assert result["success"] is True
        assert "0s" in result["message"] or "1s" in result["message"]
        import asyncio

        await asyncio.sleep(0.7)
    mock_relaunch.assert_called_once()
