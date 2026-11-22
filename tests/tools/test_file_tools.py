"""Tests for the file tools' path guard (core.path_access).

Regression coverage for a real bug found during the self-repair audit: the
tool had two full try/except blocks back to back, where the first always
returned before the second (which held the only path guard) could ever run
— so write_file had zero enforced path restriction in practice.
"""

import sys
from types import SimpleNamespace

import pytest

from tools.file_tools import FileWriteTool


@pytest.fixture
def temp_base(tmp_path, monkeypatch):
    """Point folders.base at a temp folder so write_file never lands in the repo."""
    config = SimpleNamespace(folders=SimpleNamespace(base=str(tmp_path / "base")), security=None)
    monkeypatch.setattr("core.config.load_config", lambda *a, **k: config)
    return tmp_path / "base"


@pytest.mark.asyncio
async def test_write_file_relative_path_lands_in_projects(temp_base):
    tool = FileWriteTool()
    result = await tool.execute(file_path="scratch/target.txt", content="hello")
    assert "Successfully" in result
    assert (temp_base / "projects" / "scratch" / "target.txt").read_text() == "hello"


@pytest.mark.asyncio
async def test_write_file_refuses_outside_base_and_repo(temp_base):
    import core.safe_code_editor as sce

    outside = (
        "C:/Windows/Temp/wits_should_not_write_here.txt"
        if sys.platform == "win32"
        else "/tmp/wits_should_not_write_here.txt"
    )
    tool = FileWriteTool()
    for target in (outside, str(sce.PROJECT_ROOT / "_wits_should_not_write_here.txt")):
        result = await tool.execute(file_path=target, content="should not land")
        assert result.startswith("Error:")
        assert "Refusing to write" in result
    assert not (sce.PROJECT_ROOT / "_wits_should_not_write_here.txt").exists()


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

    outside = "C:/Windows/win.ini" if sys.platform == "win32" else "/etc/hosts"
    result = await tool.execute(file_path=outside, user_role="guest")
    assert result.startswith("Error:")
    assert "allowed" in result.lower()
