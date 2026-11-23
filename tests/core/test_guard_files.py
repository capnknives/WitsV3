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
        "core/untrusted_content.py",
        "core/sandbox_runner.py",
        "core/docker_sandbox.py",
        "core/injection_guard.py",
        "core/tool_call_guard.py",
        "core/safe_calc.py",
        "core/guard_files.py",
        "agents/tool_tiers.py",
        "agents/turn_context.py",
        "tools/python_execution_tool.py",
        "tools/self_repair_tools.py",
        "core/config.py",
        "core/roles.py",
        "core/__init__.py",
        "core/workspace/__init__.py",
        "agents/__init__.py",
        "tools/__init__.py",
        "web/__init__.py",
    }
    for name in sorted(required):
        assert name in gf.GUARD_FILES, name
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
        "core/safe_code_editor_notes.py",
        "OTHER~1.PY",
        "docs/OTHER~1.MD",
        "scripts/a~b.py",
        "core/approx~.py",
        "tests/test_redteam.py",
        "tests/redteam",
        "redteam/x.py",
        "agents/self_repair_agent.py",
    ],
)
def test_non_guard_files_allowed(text):
    assert _guard(text) is None


@pytest.mark.parametrize(
    "text",
    [
        "core/SA1B2C~1.PY",
        "core/OTHER~1.PY",
        "agents/AB12CD~3.PY",
        "tools/X~9.PY",
        "web/FILE_R~1.PY",
        "core/WORKSP~1/__init__.py",
        "core/workspace/PE1A2B~1.PY",
        "tests/REDTEA~1/x.py",
        "tests/AB12~1/x.py",
    ],
)
def test_any_short_name_under_a_guard_folder_refused(text):
    # Reason: hashed short names (after four collisions) carry no usable prefix.
    assert _guard(text) == gf.GUARD_REFUSAL


@pytest.mark.parametrize("text", ["core/../core/path_access.py", "./core/config.py"])
def test_relative_text_is_anchored_at_project_root(text):
    assert _guard(text) == gf.GUARD_REFUSAL


@pytest.mark.parametrize("text", ["pkg/core/path_access.py", "x/tests/redteam/a.py"])
def test_guard_names_deeper_in_the_tree_are_not_guards(text):
    assert _guard(text) is None


@pytest.mark.parametrize("mode", [" docker ", "DOCKER", "Docker\n"])
def test_mode_is_normalised_once(mode):
    assert gf.normalise_mode(mode) == "docker"
    assert gf.host_test_reason("tests/t.py", Path("tests/t.py"), mode) is None


@pytest.mark.parametrize("mode", [None, "", "dockerx", "docker-ish", "dock er", 0])
def test_anything_not_exactly_docker_counts_as_host(mode):
    assert gf.host_test_reason("tests/t.py", Path("tests/t.py"), mode) == gf.TEST_REFUSAL


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
