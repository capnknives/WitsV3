"""Tests for filesystem read allowlist."""

import sys

import pytest

from core.filesystem_policy import (
    project_read_root,
    read_roots_for_role,
    resolve_allowed_read_path,
)


def test_owner_can_read_inside_project(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "core.filesystem_policy.PROJECT_ROOT",
        tmp_path,
        raising=False,
    )
    f = tmp_path / "README.md"
    f.write_text("hello", encoding="utf-8")
    resolved = resolve_allowed_read_path("README.md", role="owner", config=None)
    assert resolved == f.resolve()


def test_guest_denied_outside_project(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "core.filesystem_policy.PROJECT_ROOT",
        tmp_path,
        raising=False,
    )
    outside = (
        "C:/Windows/win.ini" if sys.platform == "win32" else "/etc/hosts"
    )
    with pytest.raises(PermissionError):
        resolve_allowed_read_path(outside, role="guest", config=None)


def test_guest_project_only_roots(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "core.filesystem_policy.PROJECT_ROOT",
        tmp_path,
        raising=False,
    )
    roots = read_roots_for_role("guest", config=None)
    assert roots == [tmp_path.resolve()]


def test_traversal_blocked(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "core.filesystem_policy.PROJECT_ROOT",
        tmp_path,
        raising=False,
    )
    with pytest.raises(PermissionError):
        resolve_allowed_read_path("../outside.txt", role="owner", config=None)


def test_project_read_root_matches_safe_editor():
    from core.safe_code_editor import PROJECT_ROOT

    assert project_read_root() == PROJECT_ROOT.resolve()
