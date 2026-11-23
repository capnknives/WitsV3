"""Red-team RT-13: the docker pytest container has network and sees the whole project.

``run_pytest_sandboxed`` mounts PROJECT_ROOT read-only at /workspace but passes no
``--network=none`` and none of the hardening flags ``_run_docker_sandbox`` uses, so test
code run after an edit can read ``.env`` / ``var/`` and reach the network. Fixed: shared
hardening flags plus masking mounts over secrets; the pins guard what already held.
"""

import asyncio
import posixpath
import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest

import core.docker_sandbox as docker_sandbox
import core.safe_code_editor as sce
import core.sandbox_runner as sbr

ROOT = str(sbr.PROJECT_ROOT)
SENSITIVE = [".env", ".env.local", "config.local.yaml", "var", ".git"]


# --- helpers -------------------------------------------------------------------------


def _mounts(cmd):
    """Return ordered (src, dst) pairs; src is None for tmpfs. Covers -v/--volume/--tmpfs."""
    out = []
    i = 0
    while i < len(cmd):
        tok = cmd[i]
        val = None
        if tok in ("-v", "--volume", "--tmpfs", "--mount") and i + 1 < len(cmd):
            val, key = cmd[i + 1], tok
            i += 1
        elif tok.startswith(("--volume=", "--tmpfs=", "--mount=")):
            key, val = tok.split("=", 1)
        if val is not None:
            if key in ("-v", "--volume"):
                parts = val.split(":")
                # Reason: Windows host paths contain "C:"; destination is the /-prefixed part.
                dst_idx = next((n for n, p in enumerate(parts) if n > 0 and p.startswith("/")), 1)
                out.append((":".join(parts[:dst_idx]), parts[dst_idx]))
            elif key == "--tmpfs":
                out.append((None, val.split(":")[0]))
            else:
                kv = dict(p.split("=", 1) for p in val.split(",") if "=" in p)
                out.append((kv.get("src") or kv.get("source"), kv.get("dst") or kv.get("target")))
        i += 1
    return out


def sensitive_path_hidden(cmd, rel, project_root=ROOT):
    """True when /workspace/<rel> cannot expose the host project's file in this command.

    Hidden means: the project root is never mounted under /workspace, or a later
    mount/tmpfs sits on that path (or a parent below /workspace) and masks it.
    """
    target = posixpath.join("/workspace", rel)
    exposed = False
    for src, dst in _mounts(cmd):
        dst = posixpath.normpath(dst)
        covers = target == dst or target.startswith(dst.rstrip("/") + "/")
        if src is not None and src.rstrip("/\\") == project_root.rstrip("/\\") and covers:
            exposed = True  # project (or a parent of the target) mounted here
        elif exposed and covers and dst != "/workspace":
            exposed = False  # later mount masks it
        elif exposed and covers and dst == "/workspace":
            exposed = False  # replaced by something that is not the project
    return not exposed


def _flag(cmd, name):
    return any(t == name or t.startswith(name + "=") for t in cmd)


def _flag_value(cmd, name):
    for n, t in enumerate(cmd):
        if t.startswith(name + "="):
            return t.split("=", 1)[1]
        if t == name and n + 1 < len(cmd):
            return cmd[n + 1]
    return None


def _capture_pytest_cmd(monkeypatch, communicate=None, timeout=120.0):
    """Run run_pytest_sandboxed in docker mode; return (ok, output, argv)."""
    captured = []

    class _Proc:
        returncode = 0

        async def communicate(self):
            return await communicate() if communicate else (b"ok", b"")

        def kill(self):
            pass

        async def wait(self):
            return -9

    async def _fake_exec(*cmd, **kwargs):
        captured.append(list(cmd))
        return _Proc()

    monkeypatch.setattr(docker_sandbox, "_docker_exe", lambda: "/usr/bin/docker")
    monkeypatch.setattr(sbr.asyncio, "create_subprocess_exec", _fake_exec)
    config = SimpleNamespace(security=SimpleNamespace(sandbox_mode="docker"))
    run = sbr.run_pytest_sandboxed(["tests/x.py"], config=config, timeout=timeout)
    ok, out = asyncio.run(run)
    assert len(captured) == 1
    return ok, out, captured[0]


@pytest.fixture
def docker_cmd(monkeypatch):
    """Run run_pytest_sandboxed in docker mode and return the captured argv."""
    ok, _, cmd = _capture_pytest_cmd(monkeypatch)
    assert ok
    return cmd


@pytest.fixture
def fake_root(tmp_path, monkeypatch):
    """A project root where every sensitive path exists (the owner's PC has them)."""
    root = tmp_path / "proj"
    root.mkdir()
    for name in (".env", ".env.local", ".env.production", "config.local.yaml", "keep.py"):
        (root / name).write_text("SECRET=1")
    (root / "var" / "data").mkdir(parents=True)
    (root / ".git").mkdir()
    monkeypatch.setattr(sbr, "PROJECT_ROOT", root)
    return root


# --- helper self-tests ---------------------------------------------------------------

_BASE = ["docker", "run", "-v", f"{ROOT}:/workspace:ro", "img", "pytest"]


def test_helper_flags_plain_project_mount_as_exposed():
    assert not any(sensitive_path_hidden(_BASE, p) for p in SENSITIVE)


def test_helper_accepts_masking_mounts():
    cmd = ["docker", "run", "-v", f"{ROOT}:/workspace:ro"]
    cmd += ["--tmpfs=/workspace/var", "-v", "/dev/null:/workspace/.env:ro"]
    cmd += ["--tmpfs", "/workspace/.git", "img"]
    assert sensitive_path_hidden(cmd, "var") and sensitive_path_hidden(cmd, ".env")
    assert sensitive_path_hidden(cmd, ".git")
    assert not sensitive_path_hidden(cmd, "config.local.yaml")
    assert not sensitive_path_hidden(cmd, ".env.local")


def test_helper_accepts_filtered_copy_and_ignores_mask_before_project():
    copy = ["docker", "run", "-v", "/tmp/filtered:/workspace:ro", "img"]
    assert all(sensitive_path_hidden(copy, p) for p in SENSITIVE)
    early = ["docker", "run", "--tmpfs=/workspace/var", "-v", f"{ROOT}:/workspace:ro", "img"]
    assert not sensitive_path_hidden(early, "var")


# --- pins that hold today ------------------------------------------------------------


def test_project_mount_is_read_only(docker_cmd):
    assert f"{ROOT}:/workspace:ro" in docker_cmd


def test_pytest_runs_inside_the_image_not_on_host(docker_cmd):
    assert docker_cmd[:3] == ["/usr/bin/docker", "run", "--rm"]
    assert docker_cmd.index("pytest") > docker_cmd.index("witsv3-sandbox")


def test_missing_docker_returns_failure_without_running_pytest(monkeypatch):
    monkeypatch.setattr(docker_sandbox, "_docker_exe", lambda: None)

    async def _boom(*a, **k):
        raise AssertionError("must not spawn anything")

    async def _host(*a, **k):
        raise AssertionError("must not run on host")

    monkeypatch.setattr(sbr.asyncio, "create_subprocess_exec", _boom)
    monkeypatch.setattr(sce, "run_pytest", _host)
    config = SimpleNamespace(security=SimpleNamespace(sandbox_mode="docker"))
    ok, out = asyncio.run(sbr.run_pytest_sandboxed(None, config=config))
    assert not ok and "docker CLI not found" in out


def test_non_docker_mode_without_host_fallback_refuses(monkeypatch):
    async def _host(*a, **k):
        raise AssertionError("must not run on host")

    monkeypatch.setattr(sce, "run_pytest", _host)
    config = SimpleNamespace(security=SimpleNamespace(sandbox_mode="subprocess"))
    ok, out = asyncio.run(sbr.run_pytest_sandboxed(None, config=config, host_fallback=False))
    assert not ok and out.startswith("refused")


# --- RT-13 fixed: container is network-less, hardened and blind to secrets ----------


def test_pytest_container_has_no_network(docker_cmd):
    assert "--network=none" in docker_cmd or _flag_value(docker_cmd, "--network") == "none"


@pytest.mark.parametrize("rel", SENSITIVE)
def test_secrets_not_visible_in_pytest_container(monkeypatch, fake_root, rel):
    _, _, cmd = _capture_pytest_cmd(monkeypatch)
    assert f"{fake_root}:/workspace:ro" in cmd
    assert sensitive_path_hidden(
        cmd, rel, project_root=str(fake_root)
    ), f"/workspace/{rel} readable: {shlex.join(cmd)}"


def test_pytest_container_drops_capabilities(docker_cmd):
    assert _flag_value(docker_cmd, "--cap-drop") == "ALL"


def test_pytest_container_no_new_privileges(docker_cmd):
    assert "--security-opt=no-new-privileges" in docker_cmd


def test_pytest_container_has_pids_limit(docker_cmd):
    assert _flag(docker_cmd, "--pids-limit")


def test_pytest_container_has_memory_limit(docker_cmd):
    assert _flag(docker_cmd, "--memory")


def test_pytest_container_runs_non_root(docker_cmd):
    user = _flag_value(docker_cmd, "--user") or ""
    assert user and user.split(":")[0] not in ("0", "root")


def test_pytest_container_read_only_rootfs_with_tmpfs(docker_cmd):
    assert "--read-only" in docker_cmd
    assert any(dst == "/tmp" and src is None for src, dst in _mounts(docker_cmd))


# --- RT-13b: masking, cleanup, shared hardening, constants ---------------------------


def test_mask_list_only_contains_existing_paths(tmp_path, monkeypatch):
    root = tmp_path / "proj"
    (root / "var").mkdir(parents=True)
    (root / ".env.local").write_text("x")
    (root / "main.py").write_text("x")
    assert sbr.masked_paths(root) == [root / ".env.local", root / "var"]
    monkeypatch.setattr(sbr, "PROJECT_ROOT", root)
    _, _, cmd = _capture_pytest_cmd(monkeypatch)
    dsts = [dst for _, dst in _mounts(cmd)]
    for missing in (".env", "config.local.yaml", ".git"):
        assert f"/workspace/{missing}" not in dsts
    assert {"/workspace/.env.local", "/workspace/var"} <= set(dsts)


def test_env_variants_found_by_enumeration(fake_root, monkeypatch):
    (fake_root / ".ENV.staging").write_text("x")
    names = [p.name for p in sbr.masked_paths(fake_root)]
    for name in (".env", ".env.local", ".env.production", ".ENV.staging"):
        assert name in names
    _, _, cmd = _capture_pytest_cmd(monkeypatch)
    for rel in (".env.production", ".ENV.staging"):
        assert sensitive_path_hidden(cmd, rel, project_root=str(fake_root))
    assert not sensitive_path_hidden(cmd, "keep.py", project_root=str(fake_root))


def test_git_file_of_a_worktree_is_masked_with_a_file(tmp_path, monkeypatch):
    root = tmp_path / "proj"
    root.mkdir()
    (root / ".git").write_text("gitdir: /elsewhere")
    monkeypatch.setattr(sbr, "PROJECT_ROOT", root)
    _, _, cmd = _capture_pytest_cmd(monkeypatch)
    src, dst = [m for m in _mounts(cmd) if m[1] == "/workspace/.git"][0]
    assert src is not None and src != str(root)
    assert sensitive_path_hidden(cmd, ".git", project_root=str(root))


def _mask_files(cmd, root):
    return {src for src, dst in _mounts(cmd) if src and dst.startswith("/workspace/")} - {root}


def test_mask_files_live_outside_project_and_are_removed(fake_root, monkeypatch):
    tmps, during = [], []
    real = sbr.mask_mount_args

    def _spy(root, tmp):
        tmps.append(tmp)
        return real(root, tmp)

    async def _communicate():
        # Reason: the empty mask file must exist while the container runs.
        during.append(any(Path(tmps[0]).iterdir()))
        return b"ok", b""

    monkeypatch.setattr(sbr, "mask_mount_args", _spy)
    ok, _, cmd = _capture_pytest_cmd(monkeypatch, communicate=_communicate)
    files = _mask_files(cmd, str(fake_root))
    assert ok and during == [True] and files
    assert all(Path(f).is_relative_to(tmps[0]) for f in files)
    assert not any(Path(f).is_relative_to(fake_root) for f in files)
    assert not Path(tmps[0]).exists()


def test_mask_files_removed_on_timeout(fake_root, monkeypatch):
    present = []

    async def _hang():
        present.append(True)
        await asyncio.sleep(5)
        return b"", b""

    ok, out, cmd = _capture_pytest_cmd(monkeypatch, communicate=_hang, timeout=0.05)
    assert not ok and "timed out" in out and present
    files = _mask_files(cmd, str(fake_root))
    assert files and not any(Path(f).exists() for f in files)


def test_project_tree_untouched(fake_root, monkeypatch):
    before = sorted(p.relative_to(fake_root) for p in fake_root.rglob("*"))
    _capture_pytest_cmd(monkeypatch)
    assert sorted(p.relative_to(fake_root) for p in fake_root.rglob("*")) == before


def _python_sandbox_cmd(monkeypatch):
    captured = []

    class _Proc:
        returncode = 0

        async def communicate(self):
            return b"1", b""

    async def _fake_exec(*cmd, **kwargs):
        captured.append(list(cmd))
        return _Proc()

    monkeypatch.setattr(docker_sandbox, "_docker_exe", lambda: "/usr/bin/docker")
    monkeypatch.setattr(sbr.asyncio, "create_subprocess_exec", _fake_exec)
    config = SimpleNamespace(security=SimpleNamespace(sandbox_mode="docker"))
    asyncio.run(sbr.run_python_sandboxed("print(1)", config=config))
    return captured[0]


def test_python_sandbox_argv_unchanged(monkeypatch):
    cmd = _python_sandbox_cmd(monkeypatch)
    mount = next(t for t in cmd if t.endswith(":/sandbox:ro"))
    assert cmd == [
        "/usr/bin/docker",
        "run",
        "--rm",
        "--network=none",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit=128",
        "--memory=512m",
        "--memory-swap=512m",
        "--user=65534:65534",
        "--read-only",
        "--tmpfs=/tmp:rw,noexec,nosuid,size=64m",
        "-e",
        "HOME=/tmp",
        "-v",
        mount,
        "witsv3-sandbox",
        "python",
        "/sandbox/user_code.py",
    ]


def test_both_containers_share_the_hardening_helper(monkeypatch):
    calls = []

    def _spy(**kw):
        calls.append(kw)
        return [f"--sentinel-{len(calls)}"]

    monkeypatch.setattr(sbr, "hardening_flags", _spy)
    assert "--sentinel-1" in _python_sandbox_cmd(monkeypatch)
    assert "--sentinel-2" in _capture_pytest_cmd(monkeypatch)[2]
    assert [c["pids"] for c in calls] == [sbr.PYTHON_PIDS_LIMIT, sbr.PYTEST_PIDS_LIMIT]


def test_pytest_limits_come_from_constants(docker_cmd):
    flags = sbr.hardening_flags(
        pids=sbr.PYTEST_PIDS_LIMIT, memory=sbr.PYTEST_MEMORY, tmpfs=sbr.PYTEST_TMPFS
    )
    start = docker_cmd.index(flags[0])
    assert docker_cmd[start : start + len(flags)] == flags
    assert f"--pids-limit={sbr.PYTEST_PIDS_LIMIT}" in docker_cmd
    assert f"--memory={sbr.PYTEST_MEMORY}" in docker_cmd
    assert f"--memory-swap={sbr.PYTEST_MEMORY}" in docker_cmd
    assert f"--tmpfs={sbr.PYTEST_TMPFS}" in docker_cmd
    assert _flag_value(docker_cmd, "--user") == "65534:65534"
    assert "HOME=/tmp" in docker_cmd
    # Reason: hardening precedes the image; anything after it is pytest's own argv.
    assert start < docker_cmd.index("witsv3-sandbox")
