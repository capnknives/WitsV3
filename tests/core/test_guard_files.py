"""Unit tests for core/guard_files.py (RT-12 guard list and host test-edit rule)."""

from pathlib import Path

import pytest

from core import guard_files as gf


def _guard(text: str) -> str | None:
    return gf.guard_reason(text, Path(text))


@pytest.mark.parametrize("rel", sorted(gf.GUARD_FILES))
def test_every_listed_file_is_refused(rel):
    assert _guard(rel) == gf.GUARD_REFUSAL


def test_list_names_the_required_files():
    required = {
        "core/safe_code_editor.py",
        "core/path_access.py",
        "core/sandbox_runner.py",
        "core/safe_calc.py",
        "core/guard_files.py",
        "agents/tool_tiers.py",
        "tools/self_repair_tools.py",
    }
    assert required <= gf.GUARD_FILES
    assert "tests/redteam" in gf.GUARD_DIRS


@pytest.mark.parametrize(
    "text",
    [
        "CORE\\Path_Access.PY",
        "./core/./guard_files.py",
        "pkg/../core/safe_code_editor.py",
        "tests/redteam/test_new.py",
        "Tests\\RedTeam\\deep\\x.py",
    ],
)
def test_normalised_spellings_refused(text):
    assert _guard(text) == gf.GUARD_REFUSAL


@pytest.mark.parametrize(
    "text",
    ["core/SAFE_C~1.PY", "agents/TOOL_T~2.PY", "tests/REDTEA~1/x.py", "TESTS~1/redteam/x.py"],
)
def test_short_names_refused(text):
    assert _guard(text) == gf.GUARD_REFUSAL


@pytest.mark.parametrize(
    "text",
    [
        "core/config.py",
        "core/safe_code_editor_notes.py",
        "core/OTHER~1.PY",
        "tests/test_redteam.py",
        "tests/redteam",
        "redteam/x.py",
        "agents/self_repair_agent.py",
    ],
)
def test_non_guard_files_allowed(text):
    assert _guard(text) is None


def test_resolved_path_is_judged_too():
    # A harmless-looking request that resolved onto a guard file (e.g. via a symlink).
    assert gf.guard_reason("pkg/link.py", Path("core/path_access.py")) == gf.GUARD_REFUSAL


@pytest.mark.parametrize("mode", ["off", "subprocess", "", "OFF"])
@pytest.mark.parametrize("rel", ["tests/test_x.py", "tests/helpers/util.py", "TESTS\\a.py"])
def test_test_files_refused_outside_docker(mode, rel):
    assert gf.host_test_reason(rel, Path(rel), mode) == gf.TEST_REFUSAL


@pytest.mark.parametrize("mode", ["docker", "Docker"])
def test_test_files_allowed_in_docker(mode):
    assert gf.host_test_reason("tests/test_x.py", Path("tests/test_x.py"), mode) is None


@pytest.mark.parametrize("rel", ["core/config.py", "testsuite/x.py", "pkg/tests_util.py"])
def test_non_test_files_not_refused_by_test_rule(rel):
    assert gf.host_test_reason(rel, Path(rel), "off") is None
