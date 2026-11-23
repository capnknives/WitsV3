"""Red-team RT-13: the docker pytest container has network and sees the whole project.

``run_pytest_sandboxed`` mounts PROJECT_ROOT read-only at /workspace but passes no
``--network=none`` and none of the hardening flags ``_run_docker_sandbox`` uses, so test
code run after an edit can read ``.env`` / ``var/`` and reach the network. The xfail tests
assert the correct behaviour; the plain tests pin what already holds.
"""

import asyncio
import posixpath
import shlex
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


@pytest.fixture
def docker_cmd(monkeypatch):
    """Run run_pytest_sandboxed in docker mode and return the captured argv."""
    captured = []

    class _Proc:
        returncode = 0

        async def communicate(self):
            return b"ok", b""

    async def _fake_exec(*cmd, **kwargs):
        captured.append(list(cmd))
        return _Proc()

    monkeypatch.setattr(docker_sandbox, "_docker_exe", lambda: "/usr/bin/docker")
    monkeypatch.setattr(sbr.asyncio, "create_subprocess_exec", _fake_exec)
    config = SimpleNamespace(security=SimpleNamespace(sandbox_mode="docker"))
    ok, _ = asyncio.run(sbr.run_pytest_sandboxed(["tests/x.py"], config=config))
    assert ok and len(captured) == 1
    return captured[0]


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


# --- RT-13 holes (xfail: assert correct behaviour) -----------------------------------


@pytest.mark.xfail(strict=True, reason="RT-13: pytest container is started without --network=none")
def test_pytest_container_has_no_network(docker_cmd):
    assert "--network=none" in docker_cmd or _flag_value(docker_cmd, "--network") == "none"


@pytest.mark.xfail(
    strict=True, reason="RT-13: whole project incl. .env, var/ and .git is readable in container"
)
@pytest.mark.parametrize("rel", SENSITIVE)
def test_secrets_not_visible_in_pytest_container(docker_cmd, rel):
    assert sensitive_path_hidden(
        docker_cmd, rel
    ), f"/workspace/{rel} readable: {shlex.join(docker_cmd)}"


@pytest.mark.xfail(strict=True, reason="RT-13: no --cap-drop=ALL on the pytest container")
def test_pytest_container_drops_capabilities(docker_cmd):
    assert _flag_value(docker_cmd, "--cap-drop") == "ALL"


@pytest.mark.xfail(strict=True, reason="RT-13: no no-new-privileges on the pytest container")
def test_pytest_container_no_new_privileges(docker_cmd):
    assert "--security-opt=no-new-privileges" in docker_cmd


@pytest.mark.xfail(strict=True, reason="RT-13: no --pids-limit on the pytest container")
def test_pytest_container_has_pids_limit(docker_cmd):
    assert _flag(docker_cmd, "--pids-limit")


@pytest.mark.xfail(strict=True, reason="RT-13: no memory limit on the pytest container")
def test_pytest_container_has_memory_limit(docker_cmd):
    assert _flag(docker_cmd, "--memory")


@pytest.mark.xfail(strict=True, reason="RT-13: pytest container runs as the image default (root)")
def test_pytest_container_runs_non_root(docker_cmd):
    user = _flag_value(docker_cmd, "--user") or ""
    assert user and user.split(":")[0] not in ("0", "root")


@pytest.mark.xfail(strict=True, reason="RT-13: container root fs is writable and has no tmpfs /tmp")
def test_pytest_container_read_only_rootfs_with_tmpfs(docker_cmd):
    assert "--read-only" in docker_cmd
    assert any(dst == "/tmp" and src is None for src, dst in _mounts(docker_cmd))
