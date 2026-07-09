"""Tests for FileWriteTool's project-directory guard.

Regression coverage for a real bug found during the self-repair audit: the
tool had two full try/except blocks back to back, where the first always
returned before the second (which held the only path guard) could ever run
— so write_file had zero enforced path restriction in practice.
"""

import sys

import pytest

from tools.file_tools import FileWriteTool


@pytest.mark.asyncio
async def test_write_file_succeeds_inside_project(tmp_path, monkeypatch):
    import core.safe_code_editor as sce

    scratch_rel = "tests/tools/_scratch_write_file_target.txt"
    scratch_abs = sce.PROJECT_ROOT / scratch_rel
    tool = FileWriteTool()
    try:
        result = await tool.execute(file_path=scratch_rel, content="hello")
        assert "Successfully" in result
        assert scratch_abs.read_text(encoding="utf-8") == "hello"
    finally:
        scratch_abs.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_write_file_refuses_outside_project():
    outside = (
        "C:/Windows/Temp/wits_should_not_write_here.txt"
        if sys.platform == "win32"
        else "/tmp/wits_should_not_write_here.txt"
    )
    tool = FileWriteTool()
    result = await tool.execute(file_path=outside, content="should not land")
    assert result.startswith("Error:")
    assert "project directory" in result


@pytest.mark.asyncio
async def test_read_file_refuses_outside_allowed_roots(monkeypatch, tmp_path):
    import core.filesystem_policy as fp
    import core.safe_code_editor as sce

    monkeypatch.setattr(sce, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(fp, "PROJECT_ROOT", tmp_path)
    inside = tmp_path / "allowed.txt"
    inside.write_text("ok", encoding="utf-8")

    tool = __import__("tools.file_tools", fromlist=["FileReadTool"]).FileReadTool()
    result = await tool.execute(file_path=str(inside), user_role="owner")
    assert result == "ok"

    outside = (
        "C:/Windows/win.ini" if sys.platform == "win32" else "/etc/hosts"
    )
    result = await tool.execute(file_path=outside, user_role="guest")
    assert result.startswith("Error:")
    assert "allowed" in result.lower()
