"""Tests for the real SelfRepairAgent (detect -> diagnose -> fix -> verify loop)."""

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest

from agents.self_repair_agent import (
    SelfRepairAgent,
    extract_code_from_response,
    extract_file_mention,
)
from core.config import WitsV3Config
from core.llm_interface import BaseLLMInterface
from core.schemas import StreamData


class ScriptedLLM(BaseLLMInterface):
    """Returns scripted responses and records calls."""

    def __init__(self, response: str = "System healthy.", config: WitsV3Config | None = None):
        super().__init__(config or WitsV3Config())
        self.response = response
        self.calls: list[str] = []

    async def generate_text(self, prompt: str, **kwargs) -> str:
        self.calls.append(prompt)
        return self.response

    async def stream_text(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        yield self.response

    async def get_embedding(self, text: str, model: str | None = None) -> list[float]:
        return [0.0] * 8


def _fake_registry(tools: dict):
    registry = MagicMock()
    registry.get_tool = lambda name: tools.get(name)
    return registry


@pytest.mark.asyncio
async def test_self_repair_agent_initializes():
    agent = SelfRepairAgent("TestSelfRepair", WitsV3Config(), ScriptedLLM())
    assert agent.agent_name == "TestSelfRepair"
    assert agent.llm_interface is not None


@pytest.mark.asyncio
async def test_falls_back_to_plain_llm_response_without_tool_registry():
    """No tool_registry (e.g. constructed standalone) -> graceful LLM passthrough."""
    llm = ScriptedLLM("All systems nominal.")
    agent = SelfRepairAgent("TestSelfRepair", WitsV3Config(), llm, tool_registry=None)

    streams: list[StreamData] = []
    async for item in agent.run("check disk space"):
        streams.append(item)

    assert streams[-1].type == "result"
    assert streams[-1].content == "All systems nominal."
    assert llm.calls == ["check disk space"]


@pytest.mark.asyncio
async def test_disabled_via_config_short_circuits():
    config = WitsV3Config()
    config.self_repair.enabled = False
    agent = SelfRepairAgent(
        "TestSelfRepair", config, ScriptedLLM(), tool_registry=_fake_registry({})
    )

    streams = [item async for item in agent.run("fix the bug")]
    assert len(streams) == 1
    assert "disabled" in streams[0].content.lower()


@pytest.mark.asyncio
async def test_no_issues_found_reports_nothing_to_repair():
    diagnose = AsyncMock(return_value={"issues": [], "message": "No issues."})
    registry = _fake_registry({"diagnose_log_errors": MagicMock(execute=diagnose)})
    agent = SelfRepairAgent("TestSelfRepair", WitsV3Config(), ScriptedLLM(), tool_registry=registry)

    streams = [item async for item in agent.run("please do a health check")]
    assert any("nothing to repair" in s.content.lower() for s in streams)


@pytest.mark.asyncio
async def test_falls_back_to_test_suite_when_no_file_named_and_no_log_issues():
    """2026-07-08 finding: a vague 'find bugs in the codebase' request with no
    resolvable log traceback previously had no path to inspect the codebase
    at all. Falls back to running the real test suite and parsing failures
    the same way as a logged traceback (run_pytest's --tb=native output)."""
    import core.safe_code_editor as sce

    scratch = sce.PROJECT_ROOT / "tests" / "agents" / "_scratch_target3.py"
    scratch.write_text("def broken():\n    return 1 / 0\n", encoding="utf-8")

    diagnose = AsyncMock(return_value={"issues": [], "message": "No issues."})
    native_failure_output = (
        "Traceback (most recent call last):\n"
        f'  File "{scratch}", line 2, in broken\n'
        "    return 1 / 0\n"
        "ZeroDivisionError: division by zero\n"
    )
    test_suite = AsyncMock(return_value={"passed": False, "output": native_failure_output})
    fix_execute = AsyncMock(
        return_value={
            "success": True,
            "committed": True,
            "commit_sha": "def5678",
            "message": "Edit applied and verified.",
            "test_output": "1 passed",
        }
    )
    registry = _fake_registry(
        {
            "diagnose_log_errors": MagicMock(execute=diagnose),
            "run_test_suite": MagicMock(execute=test_suite),
            "apply_code_fix": MagicMock(execute=fix_execute),
        }
    )
    llm = ScriptedLLM("```python\ndef broken():\n    return 0\n```")
    agent = SelfRepairAgent("TestSelfRepair", WitsV3Config(), llm, tool_registry=registry)

    try:
        streams = [item async for item in agent.run("find and fix any bugs in the codebase")]
    finally:
        scratch.unlink(missing_ok=True)

    test_suite.assert_awaited_once()
    fix_execute.assert_awaited_once()
    assert any("Repaired" in s.content for s in streams if s.type == "result")


@pytest.mark.asyncio
async def test_reports_nothing_when_test_suite_passes_and_no_other_issues():
    diagnose = AsyncMock(return_value={"issues": [], "message": "No issues."})
    test_suite = AsyncMock(return_value={"passed": True, "output": "5 passed"})
    registry = _fake_registry(
        {
            "diagnose_log_errors": MagicMock(execute=diagnose),
            "run_test_suite": MagicMock(execute=test_suite),
            "apply_code_fix": MagicMock(execute=AsyncMock()),
        }
    )
    agent = SelfRepairAgent("TestSelfRepair", WitsV3Config(), ScriptedLLM(), tool_registry=registry)

    streams = [item async for item in agent.run("find and fix any bugs in the codebase")]

    test_suite.assert_awaited_once()
    assert any("nothing to repair" in s.content.lower() for s in streams)


@pytest.mark.asyncio
async def test_targets_a_file_named_in_the_request(tmp_path, monkeypatch):
    import core.safe_code_editor as sce

    scratch = sce.PROJECT_ROOT / "tests" / "agents" / "_scratch_target.py"
    scratch.write_text("def broken():\n    return 1 / 0\n", encoding="utf-8")

    fix_execute = AsyncMock(
        return_value={
            "success": True,
            "committed": True,
            "commit_sha": "abc1234",
            "message": "Edit applied and verified.",
            "test_output": "1 passed",
        }
    )
    registry = _fake_registry({"apply_code_fix": MagicMock(execute=fix_execute)})
    llm = ScriptedLLM("```python\ndef broken():\n    return 0\n```")
    agent = SelfRepairAgent("TestSelfRepair", WitsV3Config(), llm, tool_registry=registry)

    try:
        streams = [
            item
            async for item in agent.run(
                "tests/agents/_scratch_target.py has a ZeroDivisionError, please fix it"
            )
        ]
    finally:
        scratch.unlink(missing_ok=True)

    fix_execute.assert_awaited_once()
    call_kwargs = fix_execute.await_args.kwargs
    assert call_kwargs["file_path"] == "tests/agents/_scratch_target.py"
    assert "def broken():\n    return 0" in call_kwargs["new_content"]
    assert any("Repaired" in s.content for s in streams if s.type == "result")


@pytest.mark.asyncio
async def test_reverted_fix_is_reported_as_failure(tmp_path):
    import core.safe_code_editor as sce

    scratch = sce.PROJECT_ROOT / "tests" / "agents" / "_scratch_target2.py"
    scratch.write_text("def broken():\n    return 1 / 0\n", encoding="utf-8")

    fix_execute = AsyncMock(
        return_value={
            "success": False,
            "committed": False,
            "commit_sha": None,
            "message": "Verification failed; change reverted to the original file.",
            "test_output": "1 failed",
        }
    )
    registry = _fake_registry({"apply_code_fix": MagicMock(execute=fix_execute)})
    llm = ScriptedLLM("```python\nstill broken\n```")
    agent = SelfRepairAgent("TestSelfRepair", WitsV3Config(), llm, tool_registry=registry)

    try:
        streams = [
            item async for item in agent.run("tests/agents/_scratch_target2.py is broken, fix it")
        ]
    finally:
        scratch.unlink(missing_ok=True)

    assert any(
        "could not safely repair" in s.content.lower() for s in streams if s.type == "result"
    )


@pytest.mark.asyncio
async def test_scans_logs_when_no_file_named_in_request():
    diagnose = AsyncMock(
        return_value={
            "issues": [
                {
                    "actionable": True,
                    "file": "does/not/exist.py",
                    "line": 1,
                    "message": "boom",
                    "kind": "traceback",
                }
            ],
            "message": "Found 1 issue.",
        }
    )
    registry = _fake_registry(
        {
            "diagnose_log_errors": MagicMock(execute=diagnose),
            "apply_code_fix": MagicMock(execute=AsyncMock()),
        }
    )
    agent = SelfRepairAgent("TestSelfRepair", WitsV3Config(), ScriptedLLM(), tool_registry=registry)

    streams = [item async for item in agent.run("please run a health check and fix any issues")]

    diagnose.assert_awaited_once()
    assert any("does not exist" in s.content.lower() for s in streams)


@pytest.mark.asyncio
async def test_reports_when_fix_tool_unavailable_but_issues_found():
    diagnose = AsyncMock(
        return_value={
            "issues": [
                {
                    "actionable": True,
                    "file": "does/not/exist.py",
                    "line": 1,
                    "message": "boom",
                    "kind": "traceback",
                }
            ],
            "message": "Found 1 issue.",
        }
    )
    registry = _fake_registry({"diagnose_log_errors": MagicMock(execute=diagnose)})
    agent = SelfRepairAgent("TestSelfRepair", WitsV3Config(), ScriptedLLM(), tool_registry=registry)

    streams = [item async for item in agent.run("please run a health check and fix any issues")]
    assert any("apply_code_fix tool is unavailable" in s.content for s in streams)


def test_extract_file_mention_finds_existing_project_file():
    result = extract_file_mention("please look at agents/self_repair_agent.py:42 and fix it")
    assert result is not None
    path, line = result
    assert path == "agents/self_repair_agent.py"
    assert line == 42


def test_extract_file_mention_ignores_nonexistent_files():
    assert extract_file_mention("fix agents/totally_made_up_xyz123.py") is None


def test_extract_code_from_response_strips_fence():
    response = "Here you go:\n```python\nx = 1\n```\n"
    assert extract_code_from_response(response) == "x = 1\n"


def test_extract_code_from_response_falls_back_to_raw_text():
    assert extract_code_from_response("no fences here") == "no fences here"
