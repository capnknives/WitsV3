"""RT-12: apply_code_fix must not edit the files that implement its own guards,
and must not host-run edited tests when the sandbox is not docker."""

import pytest

import core.safe_code_editor as sce
import core.sandbox_runner as sbr
from tools.self_repair_tools import ApplyCodeFixTool

# Files that implement the guards. Derived from the code: the verified-edit
# pipeline and path rules, untrusted-content marking, the write gate and tool
# tiers, the sandbox runner, the python tool, and the apply_code_fix tool itself.
GUARD_FILES = [
    "core/safe_code_editor.py",
    "core/path_access.py",
    "core/untrusted_content.py",
    "core/sandbox_runner.py",
    "core/docker_sandbox.py",
    "core/injection_guard.py",
    "core/tool_call_guard.py",
    "agents/tool_tiers.py",
    "agents/turn_context.py",
    "tools/python_execution_tool.py",
    "tools/self_repair_tools.py",
    "tests/redteam/test_rt1_apply_code_fix.py",
    "tests/redteam/test_rt12_guard_files.py",
    "tests/redteam/__init__.py",
]
# Windows-style spellings of the same files.
SPELLINGS = [
    "core\\safe_code_editor.py",
    "CORE/Safe_Code_Editor.PY",
    "tests\\redteam\\test_rt1_apply_code_fix.py",
    "Tests/RedTeam/test_rt1_apply_code_fix.py",
    "agents\\tool_tiers.py",
    "core/SAFE_C~1.PY",
]


@pytest.fixture
def fake_project(tmp_path, monkeypatch):
    root = (tmp_path / "proj").resolve()
    root.mkdir()
    monkeypatch.setattr(sce, "PROJECT_ROOT", root)

    async def no_git(*args):
        return 1, "git disabled in test"

    monkeypatch.setattr(sce, "_git", no_git)
    monkeypatch.setattr(sbr, "sandbox_mode", lambda config: "off")
    return root


@pytest.fixture
def pytest_spy(fake_project, monkeypatch):
    calls = []

    async def spy(test_paths=None, timeout=120.0):
        calls.append(test_paths)
        return True, "ok"

    monkeypatch.setattr(sce, "run_pytest", spy)
    return calls


def _make(root, rel, text="ORIGINAL = 1\n"):
    path = root / rel.replace("\\", "/")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


async def _edit(rel, test_path="t.py"):
    return await ApplyCodeFixTool().execute(
        rel, "EVIL = 1\n", "rt", test_path=test_path, commit=False
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("rel", GUARD_FILES)
async def test_guard_files_refused(fake_project, pytest_spy, rel):
    path = _make(fake_project, rel)
    res = await _edit(rel)
    assert not res["success"]
    assert path.read_text() == "ORIGINAL = 1\n"
    assert pytest_spy == []


@pytest.mark.parametrize("spelling", SPELLINGS)
def test_guard_file_spellings_refused_by_pure_decision(fake_project, spelling):
    # Pure decision level: no files created, so 8.3-looking names are judged by text.
    with pytest.raises(PermissionError):
        sce.resolve_writable_within_project(spelling)


def test_windows_alias_spellings_with_real_trailing_chars_already_refused(fake_project):
    # Already refused by the RT-1 alias rules (pinned, not a hole).
    for spelling in (
        "core/safe_code_editor.py.",
        "core/safe_code_editor.py ",
        "core/path_access.py::$DATA",
    ):
        with pytest.raises(PermissionError):
            sce.resolve_writable_within_project(spelling)


@pytest.mark.parametrize("mode", ["off", "subprocess"])
@pytest.mark.parametrize("rel", ["tests/test_evil.py", "tests/helpers/util.py"])
@pytest.mark.asyncio
async def test_edited_test_code_not_host_run_without_docker(
    fake_project, pytest_spy, monkeypatch, mode, rel
):
    monkeypatch.setattr(sbr, "sandbox_mode", lambda config: mode)
    _make(fake_project, rel, "def test_a():\n    assert True\n")
    res = await _edit(rel, test_path=rel)
    ran_edited = any(p is None or rel in p or "tests" in "".join(p) for p in pytest_spy)
    assert (not res["success"]) or not ran_edited


@pytest.mark.asyncio
async def test_ordinary_source_edit_still_goes_through_pipeline(fake_project, pytest_spy):
    path = _make(fake_project, "pkg/mod.py")
    res = await _edit("pkg/mod.py", test_path="tests/test_mod.py")
    assert res["success"]
    assert path.read_text() == "EVIL = 1\n"
    assert pytest_spy == [["tests/test_mod.py"]]
