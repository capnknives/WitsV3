"""RT-1: apply_code_fix must not write secrets/hooks/conftest/data files or run supplied code."""

import pytest

import core.safe_code_editor as sce
import core.sandbox_runner as sbr
from tools.self_repair_tools import ApplyCodeFixTool

BAD_PATHS = [".env", ".git/hooks/pre-commit", "conftest.py", "tests/conftest.py", "var/data/x.db"]


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


@pytest.mark.asyncio
@pytest.mark.xfail(
    strict=True, reason="RT-1: sensitive/executable project paths are written then tested"
)
@pytest.mark.parametrize("rel", BAD_PATHS)
async def test_sensitive_paths_refused_and_never_written(fake_project, monkeypatch, rel):
    seen = []

    async def spy_pytest(test_paths=None, timeout=120.0):
        seen.append((fake_project / rel).exists())
        return True, "ok"

    monkeypatch.setattr(sce, "run_pytest", spy_pytest)
    res = await ApplyCodeFixTool().execute(rel, "X=1\n", "rt", test_path="t.py", commit=False)
    assert not res["success"]
    assert seen in ([], [False])


@pytest.mark.asyncio
@pytest.mark.xfail(strict=True, reason="RT-1: supplied conftest.py executes under host pytest")
async def test_supplied_conftest_is_not_executed(fake_project, tmp_path):
    marker = tmp_path / "pwned.marker"
    (fake_project / "test_ok.py").write_text("def test_ok():\n    assert True\n")
    body = f"import pathlib\npathlib.Path({str(marker)!r}).write_text('x')\n"
    await ApplyCodeFixTool().execute(
        "conftest.py", body, "rt", test_path="test_ok.py", commit=False
    )
    assert not marker.exists()
