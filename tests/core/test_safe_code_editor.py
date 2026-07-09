"""Tests for the verified-edit pipeline shared by the coding and self-repair agents."""

import sys

import pytest

from core import safe_code_editor as sce


def test_resolve_within_project_accepts_relative_path():
    resolved = sce.resolve_within_project("tests/core/test_safe_code_editor.py")
    assert resolved.exists()
    assert resolved.is_relative_to(sce.PROJECT_ROOT)


def test_resolve_within_project_rejects_escape():
    with pytest.raises(PermissionError):
        sce.resolve_within_project("../outside_the_project.py")


def test_resolve_within_project_rejects_absolute_outside():
    outside = "C:/Windows/System32/whatever.py" if sys.platform == "win32" else "/etc/passwd"
    with pytest.raises(PermissionError):
        sce.resolve_within_project(outside)


@pytest.mark.asyncio
async def test_apply_verified_edit_reverts_on_test_failure(tmp_path):
    target = tmp_path / "scratch_module.py"
    # Not under PROJECT_ROOT, so exercise resolve failure path directly instead.
    with pytest.raises(PermissionError):
        sce.resolve_within_project(str(target))


def _force_subprocess_pytest_path(monkeypatch):
    """apply_verified_edit uses Docker when config.security.sandbox_mode is docker.

    Stub load_config so unit tests exercise the local run_pytest path only.
    """
    from types import SimpleNamespace

    monkeypatch.setattr(
        "core.config.load_config",
        lambda *a, **k: SimpleNamespace(security=SimpleNamespace(sandbox_mode="off")),
    )


@pytest.mark.asyncio
async def test_apply_verified_edit_writes_and_reverts_in_project(monkeypatch):
    """Use a real scratch file inside the project so resolve_within_project succeeds,
    but stub run_pytest so the test doesn't depend on the real suite's runtime."""
    scratch_rel = "tests/core/_scratch_safe_edit_target.txt"
    scratch_abs = sce.PROJECT_ROOT / scratch_rel
    assert not scratch_abs.exists(), "scratch fixture file should not pre-exist"

    async def fake_fail(*args, **kwargs):
        return False, "1 failed, 0 passed"

    _force_subprocess_pytest_path(monkeypatch)
    monkeypatch.setattr(sce, "run_pytest", fake_fail)
    try:
        result = await sce.apply_verified_edit(
            scratch_rel, "new content", reason="test revert path", commit=False
        )
        assert result.success is False
        assert not scratch_abs.exists(), "failed edit to a new file should remove it"
    finally:
        scratch_abs.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_apply_verified_edit_restores_original_bytes_on_failure(monkeypatch):
    scratch_rel = "tests/core/_scratch_safe_edit_existing.txt"
    scratch_abs = sce.PROJECT_ROOT / scratch_rel
    scratch_abs.write_text("original content", encoding="utf-8")

    async def fake_fail(*args, **kwargs):
        return False, "boom"

    _force_subprocess_pytest_path(monkeypatch)
    monkeypatch.setattr(sce, "run_pytest", fake_fail)
    try:
        result = await sce.apply_verified_edit(
            scratch_rel, "corrupted content", reason="test revert existing", commit=False
        )
        assert result.success is False
        assert scratch_abs.read_text(encoding="utf-8") == "original content"
    finally:
        scratch_abs.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_apply_verified_edit_writes_without_commit_on_success(monkeypatch):
    scratch_rel = "tests/core/_scratch_safe_edit_success.txt"
    scratch_abs = sce.PROJECT_ROOT / scratch_rel

    async def fake_pass(*args, **kwargs):
        return True, "1 passed"

    _force_subprocess_pytest_path(monkeypatch)
    monkeypatch.setattr(sce, "run_pytest", fake_pass)
    try:
        result = await sce.apply_verified_edit(
            scratch_rel, "verified content", reason="test success path", commit=False
        )
        assert result.success is True
        assert result.committed is False
        assert scratch_abs.read_text(encoding="utf-8") == "verified content"
    finally:
        scratch_abs.unlink(missing_ok=True)


def test_guess_related_tests_returns_empty_when_no_match():
    assert sce.guess_related_tests("agents/some_totally_made_up_module_xyz.py") == []


def test_guess_related_tests_finds_matching_test_file():
    matches = sce.guess_related_tests("core/safe_code_editor.py")
    assert "tests/core/test_safe_code_editor.py" in matches
