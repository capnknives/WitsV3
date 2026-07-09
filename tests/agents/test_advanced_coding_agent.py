"""Tests for AdvancedCodingAgent's real file-writing and existing-file-fix capabilities.

Prior to this, the agent only produced LLM prose/canned scaffold text held
in an in-memory dict — nothing ever touched disk. These tests cover the two
real capabilities added: writing generated project files to workspace/ with
a compile check, and editing an existing named file through the same
verify-before-commit pipeline the self-repair agent uses.
"""

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest

from agents.advanced_coding_agent import AdvancedCodingAgent
from agents.coding_models import CodeProject
from core.config import WitsV3Config
from core.llm_interface import BaseLLMInterface
from core.runtime_paths import workspace_dir
from core.safe_code_editor import PROJECT_ROOT


class ScriptedLLM(BaseLLMInterface):
    def __init__(self, response: str = "ok", config: WitsV3Config | None = None):
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


@pytest.fixture
def agent():
    return AdvancedCodingAgent(
        agent_name="TestCoder",
        config=WitsV3Config(),
        llm_interface=ScriptedLLM(),
    )


@pytest.mark.asyncio
async def test_write_project_files_writes_to_workspace_and_compiles(agent):
    project = CodeProject(
        id="proj1",
        name="_scratch_test_project",
        description="",
        language="python",
        project_type="cli_tool",
        structure={},
        dependencies=[],
        files={},
        tests={},
        documentation="",
    )
    workspace_dir_path = workspace_dir() / project.name
    try:
        results = await agent._write_project_files(
            project, {"main.py": "print('hello')\n", "README.md": "# scratch\n"}
        )
        assert any("✓ main.py" in r for r in results)
        assert any("✓ README.md" in r for r in results)
        assert (workspace_dir_path / "main.py").read_text(encoding="utf-8") == "print('hello')\n"
    finally:
        import shutil

        shutil.rmtree(workspace_dir_path, ignore_errors=True)


@pytest.mark.asyncio
async def test_write_project_files_reports_syntax_errors(agent):
    project = CodeProject(
        id="proj2",
        name="_scratch_bad_project",
        description="",
        language="python",
        project_type="cli_tool",
        structure={},
        dependencies=[],
        files={},
        tests={},
        documentation="",
    )
    workspace_dir_path = workspace_dir() / project.name
    try:
        results = await agent._write_project_files(project, {"broken.py": "def broken(:\n"})
        assert any("syntax error" in r for r in results)
    finally:
        import shutil

        shutil.rmtree(workspace_dir_path, ignore_errors=True)


@pytest.mark.asyncio
async def test_run_routes_to_existing_file_fix_when_file_named():
    scratch_rel = "tests/agents/_scratch_coding_target.py"
    scratch_abs = PROJECT_ROOT / scratch_rel
    scratch_abs.write_text("def broken():\n    return 1 / 0\n", encoding="utf-8")

    fix_execute = AsyncMock(
        return_value={
            "success": True,
            "committed": False,
            "commit_sha": None,
            "message": "Edit applied and verified.",
            "test_output": "1 passed",
        }
    )
    registry = _fake_registry({"apply_code_fix": MagicMock(execute=fix_execute)})
    llm = ScriptedLLM("```python\ndef broken():\n    return 0\n```")
    agent = AdvancedCodingAgent(
        agent_name="TestCoder",
        config=WitsV3Config(),
        llm_interface=llm,
        tool_registry=registry,
    )

    try:
        streams = [
            item
            async for item in agent.run(f"{scratch_rel} raises a ZeroDivisionError, please fix it")
        ]
    finally:
        scratch_abs.unlink(missing_ok=True)

    fix_execute.assert_awaited_once()
    assert fix_execute.await_args.kwargs["file_path"] == scratch_rel
    assert any("Updated" in s.content for s in streams if s.type == "result")


@pytest.mark.asyncio
async def test_run_reports_failed_fix_as_reverted():
    scratch_rel = "tests/agents/_scratch_coding_target2.py"
    scratch_abs = PROJECT_ROOT / scratch_rel
    scratch_abs.write_text("def broken():\n    return 1 / 0\n", encoding="utf-8")

    fix_execute = AsyncMock(
        return_value={
            "success": False,
            "committed": False,
            "commit_sha": None,
            "message": "Verification failed; change reverted.",
            "test_output": "1 failed",
        }
    )
    registry = _fake_registry({"apply_code_fix": MagicMock(execute=fix_execute)})
    llm = ScriptedLLM("```python\nstill broken\n```")
    agent = AdvancedCodingAgent(
        agent_name="TestCoder",
        config=WitsV3Config(),
        llm_interface=llm,
        tool_registry=registry,
    )

    try:
        streams = [item async for item in agent.run(f"fix the bug in {scratch_rel}")]
    finally:
        scratch_abs.unlink(missing_ok=True)

    assert any("could not safely apply" in s.content.lower() for s in streams if s.type == "result")


@pytest.mark.asyncio
async def test_run_without_tool_registry_reports_unavailable():
    scratch_rel = "tests/agents/_scratch_coding_target3.py"
    scratch_abs = PROJECT_ROOT / scratch_rel
    scratch_abs.write_text("x = 1\n", encoding="utf-8")

    agent = AdvancedCodingAgent(
        agent_name="TestCoder", config=WitsV3Config(), llm_interface=ScriptedLLM()
    )
    try:
        streams = [item async for item in agent.run(f"clean up {scratch_rel}")]
    finally:
        scratch_abs.unlink(missing_ok=True)

    assert any("apply_code_fix tool is unavailable" in s.content for s in streams)
