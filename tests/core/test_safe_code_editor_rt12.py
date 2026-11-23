"""RT-12 rules at the verified-edit choke point: both entry points, one mode value."""

import pytest

import core.config as config_mod
import core.safe_code_editor as sce
import core.sandbox_runner as sbr
from core import guard_files as gf

PATCH = "<<<<<<< SEARCH\nORIGINAL = 1\n=======\nEVIL = 1\n>>>>>>> REPLACE\n"


@pytest.fixture
def proj(tmp_path, monkeypatch):
    root = (tmp_path / "proj").resolve()
    root.mkdir()
    monkeypatch.setattr(sce, "PROJECT_ROOT", root)

    async def no_git(*args):
        return 1, "git disabled in test"

    monkeypatch.setattr(sce, "_git", no_git)
    return root


@pytest.fixture
def runs(proj, monkeypatch):
    calls = {"host": [], "docker": []}

    async def host(test_paths=None, timeout=120.0):
        calls["host"].append(test_paths)
        return True, "ok"

    async def docker(test_paths=None, **kwargs):
        calls["docker"].append(test_paths)
        return True, "ok"

    monkeypatch.setattr(sce, "run_pytest", host)
    monkeypatch.setattr(sbr, "run_pytest_sandboxed", docker)
    return calls


def _make(root, rel, text="ORIGINAL = 1\n"):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _mode(monkeypatch, mode):
    monkeypatch.setattr(sbr, "sandbox_mode", lambda config: mode)


@pytest.mark.asyncio
async def test_patch_path_refuses_guard_file(proj, runs, monkeypatch):
    _mode(monkeypatch, "docker")
    path = _make(proj, "core/path_access.py")
    res = await sce.apply_verified_patch("core/path_access.py", PATCH, reason="t", commit=False)
    assert not res.success and "guard file" in res.message
    assert path.read_text() == "ORIGINAL = 1\n"
    assert runs == {"host": [], "docker": []}


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["off", "subprocess", " dockerx", ""])
async def test_patch_path_refuses_test_edit_outside_docker(proj, runs, monkeypatch, mode):
    _mode(monkeypatch, mode)
    path = _make(proj, "tests/test_x.py")
    res = await sce.apply_verified_patch("tests/test_x.py", PATCH, reason="t", commit=False)
    assert not res.success and "docker sandbox" in res.message
    assert path.read_text() == "ORIGINAL = 1\n"
    assert runs == {"host": [], "docker": []}


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["Docker", " DOCKER "])
async def test_mixed_case_docker_runs_in_docker_not_host(proj, runs, monkeypatch, mode):
    _mode(monkeypatch, mode)
    _make(proj, "tests/test_x.py")
    res = await sce.apply_verified_edit("tests/test_x.py", "X = 2\n", reason="t", commit=False)
    assert res.success
    assert runs["host"] == [] and len(runs["docker"]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["docker", "off"])
async def test_ordinary_edit_works_in_both_modes(proj, runs, monkeypatch, mode):
    _mode(monkeypatch, mode)
    path = _make(proj, "pkg/mod.py")
    res = await sce.apply_verified_edit("pkg/mod.py", "X = 2\n", reason="t", commit=False)
    assert res.success and path.read_text() == "X = 2\n"
    assert len(runs["docker" if mode == "docker" else "host"]) == 1


@pytest.mark.asyncio
async def test_unreadable_config_fails_closed_with_edit_result(proj, runs, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("config broken")

    monkeypatch.setattr(config_mod, "load_config", boom)
    path = _make(proj, "pkg/mod.py")
    res = await sce.apply_verified_edit("pkg/mod.py", "X = 2\n", reason="t", commit=False)
    assert isinstance(res, sce.EditResult) and not res.success
    assert "sandbox mode" in res.message and "config broken" in res.message
    assert path.read_text() == "ORIGINAL = 1\n"
    with pytest.raises(PermissionError):
        sce.resolve_writable_within_project("tests/test_x.py")


def test_broken_config_reads_as_off():
    assert sbr.sandbox_mode(None) == "off"
    assert sbr.sandbox_mode(object()) == "off"


@pytest.mark.asyncio
async def test_sandboxed_runner_without_host_fallback_fails_closed(monkeypatch):
    async def host(*args, **kwargs):
        raise AssertionError("must not run on host")

    monkeypatch.setattr(sce, "run_pytest", host)
    passed, out = await sbr.run_pytest_sandboxed(None, config=None, host_fallback=False)
    assert not passed and "not docker" in out


def test_symlink_to_guard_file_refused(proj):
    _make(proj, "core/path_access.py")
    link = proj / "pkg" / "innocent.py"
    link.parent.mkdir()
    try:
        link.symlink_to(proj / "core" / "path_access.py")
    except (OSError, NotImplementedError):
        pytest.skip("no symlink rights on this platform")
    with pytest.raises(PermissionError, match="guard file"):
        sce.resolve_writable_within_project("pkg/innocent.py")


@pytest.mark.parametrize(
    "text", ["tests./x.py", "tests /x.py", "core /safe_code_editor.py", "core./config.py"]
)
def test_trailing_dot_or_space_directory_parts_refused(proj, text):
    with pytest.raises(PermissionError):
        sce.resolve_writable_within_project(text, mode="docker")


def test_absolute_path_inside_project_is_judged_relative_to_root(proj):
    with pytest.raises(PermissionError, match="guard file"):
        sce.resolve_writable_within_project(str(proj / "core" / "config.py"))


def test_guard_rule_independent_of_mode(proj):
    for mode in ("docker", "off"):
        with pytest.raises(PermissionError, match="guard file"):
            sce.resolve_writable_within_project("tests/redteam/x.py", mode=mode)
    assert gf.normalise_mode(None) == ""
