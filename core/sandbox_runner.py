"""Optional sandbox boundary for python_execute and verified-edit pytest runs."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.guard_files import normalise_mode
from core.safe_code_editor import PROJECT_ROOT

logger = logging.getLogger(__name__)


@dataclass
class SandboxResult:
    success: bool
    output: str
    error: str = ""
    return_code: int = 0


UNSANDBOXED_REFUSAL = (
    "python_execute is disabled: set security.sandbox_mode to docker, or set "
    "security.python_execution_allow_unsandboxed: true to run code directly on this computer"
)

# Reason: only what the interpreter needs to start on Windows and POSIX; never secrets.
_CHILD_ENV_KEYS = ("PATH", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR")
_CHILD_ENV_LOCALE = ("LANG", "LC_ALL")

# Reason: python_execute runs one short snippet, so small caps are enough.
PYTHON_PIDS_LIMIT = 128
PYTHON_MEMORY = "512m"
PYTHON_TMPFS = "/tmp:rw,noexec,nosuid,size=64m"
# Reason: the gate may run the whole suite (~2,500 tests). asyncio's default executor
# alone can hold 32+ threads and tests spawn subprocesses (each thread counts as a pid),
# so 512 leaves headroom while still stopping a fork bomb. tmpfs pages are charged to the
# container's memory cgroup, so 2g covers the interpreter and imports plus a full /tmp.
# /tmp holds cache_dir and basetemp (every tmp_path is kept for the run): 512m. noexec
# stays: tests run scripts through the interpreter, never exec files from /tmp directly.
PYTEST_PIDS_LIMIT = 512
PYTEST_MEMORY = "2g"
PYTEST_TMPFS = "/tmp:rw,noexec,nosuid,size=512m"
# Reason: secrets and private data that test code must not read; ".env*" is enumerated.
PYTEST_MASKED_NAMES = ("config.local.yaml", "var", ".git")


def unsandboxed_allowed(config: Any | None) -> bool:
    """True only when the owner opted in to running code directly on the host."""
    security = getattr(config, "security", None) if config else None
    return getattr(security, "python_execution_allow_unsandboxed", False) is True


def minimal_child_env(workdir: str, extra: dict[str, str] | None = None) -> dict[str, str]:
    """Build a small environment for a host child process (not os.environ)."""
    env = {k: os.environ[k] for k in _CHILD_ENV_KEYS + _CHILD_ENV_LOCALE if k in os.environ}
    env["HOME"] = workdir
    env["USERPROFILE"] = workdir
    # Reason: temp files land in the throwaway dir, not the host's shared temp.
    for key in ("TEMP", "TMP", "TMPDIR"):
        env[key] = workdir
    env.update(extra or {})
    return env


def hardening_flags(*, pids: int, memory: str, tmpfs: str) -> list[str]:
    """Docker flags shared by every sandbox container so the launches cannot drift."""
    return [
        "--network=none",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        f"--pids-limit={pids}",
        f"--memory={memory}",
        f"--memory-swap={memory}",
        "--user=65534:65534",
        "--read-only",
        f"--tmpfs={tmpfs}",
        "-e",
        "HOME=/tmp",
    ]


def masked_paths(root: Path) -> list[Path]:
    """Existing secret paths under root: every ``.env*`` entry plus PYTEST_MASKED_NAMES.

    Reason: a mask over a missing path makes docker create it on some setups, so only
    paths that exist now are returned; ``.env*`` is listed at launch, never hardcoded.
    """
    found = sorted(p for p in root.iterdir() if p.name.lower().startswith(".env"))
    return found + [root / n for n in PYTEST_MASKED_NAMES if (root / n).exists()]


def mask_mount_args(root: Path, tmp: str) -> list[str]:
    """Mounts that hide masked_paths() under /workspace; must follow the project mount.

    Directories get an empty read-only tmpfs; files get an empty read-only file created
    in ``tmp`` (never in the project tree).
    """
    empty = Path(tmp) / "empty"
    empty.write_bytes(b"")
    os.chmod(empty, 0o444)
    args: list[str] = []
    for path in masked_paths(root):
        dst = f"/workspace/{path.name}"
        if path.is_dir():
            args.append(f"--tmpfs={dst}:ro,noexec,nosuid")
        else:
            args += ["-v", f"{empty}:{dst}:ro"]
    return args


def sandbox_mode(config: Any | None) -> str:
    security = getattr(config, "security", None) if config else None
    mode = getattr(security, "sandbox_mode", "off") if security else "off"
    return normalise_mode(mode) or "off"


async def run_python_sandboxed(
    code: str,
    *,
    config: Any | None = None,
    timeout: float = 30.0,
) -> SandboxResult:
    """Execute Python code in subprocess or docker sandbox when configured."""
    mode = sandbox_mode(config)
    if mode == "off":
        return SandboxResult(success=False, output="", error="sandbox not enabled", return_code=-1)
    if mode == "subprocess":
        if not unsandboxed_allowed(config):
            return SandboxResult(
                success=False, output="", error=UNSANDBOXED_REFUSAL, return_code=-1
            )
        return await _run_subprocess_sandbox(code, config=config, timeout=timeout)
    if mode == "docker":
        return await _run_docker_sandbox(code, config=config, timeout=timeout)
    return SandboxResult(
        success=False, output="", error=f"unknown sandbox_mode: {mode}", return_code=-1
    )


async def _run_subprocess_sandbox(
    code: str, *, config: Any | None, timeout: float
) -> SandboxResult:
    security = getattr(config, "security", None) if config else None
    allow_net = bool(getattr(security, "python_execution_network_access", False))
    with tempfile.TemporaryDirectory(prefix="wits_sandbox_") as tmp:
        script = Path(tmp) / "user_code.py"
        script.write_text(code, encoding="utf-8")
        env = {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
        }
        if not allow_net:
            env["WITS_SANDBOX_NO_NETWORK"] = "1"
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            str(script),
            cwd=tmp,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=minimal_child_env(tmp, env),
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return SandboxResult(
                success=False,
                output="",
                error=f"Execution timed out after {timeout:.0f}s",
                return_code=-1,
            )
        output = stdout.decode(errors="replace")
        rc = proc.returncode or 0
        return SandboxResult(success=rc == 0, output=output, return_code=rc)


async def _run_docker_sandbox(code: str, *, config: Any | None, timeout: float) -> SandboxResult:
    # Reason: reuse docker_sandbox PATH resolution so PATH-less shells work on Windows.
    from core.docker_sandbox import _docker_env, _docker_exe

    security = getattr(config, "security", None) if config else None
    image = getattr(security, "sandbox_image", "witsv3-sandbox") if security else "witsv3-sandbox"
    exe = _docker_exe()
    if not exe:
        return SandboxResult(
            success=False, output="", error="docker CLI not found on PATH", return_code=-1
        )
    with tempfile.TemporaryDirectory(prefix="wits_docker_sandbox_") as tmp:
        script = Path(tmp) / "user_code.py"
        script.write_text(code, encoding="utf-8")
        # Reason: --cap-drop=ALL drops DAC_OVERRIDE and the run is uid 65534, so the
        # 0700 temp dir would be unreadable on native Linux. Contents are only the snippet.
        os.chmod(tmp, 0o755)
        os.chmod(script, 0o644)
        cmd = [
            exe,
            "run",
            "--rm",
            *hardening_flags(pids=PYTHON_PIDS_LIMIT, memory=PYTHON_MEMORY, tmpfs=PYTHON_TMPFS),
            "-v",
            f"{tmp}:/sandbox:ro",
            image,
            "python",
            "/sandbox/user_code.py",
        ]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=_docker_env(),
            )
        except OSError as exc:
            # Never fall back to host execution when docker cannot start.
            return SandboxResult(
                success=False, output="", error=f"docker failed to start: {exc}", return_code=-1
            )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return SandboxResult(
                success=False,
                output="",
                error=f"Docker sandbox timed out after {timeout:.0f}s",
                return_code=-1,
            )
        output = stdout.decode(errors="replace")
        rc = proc.returncode or 0
        if rc != 0 and "Cannot connect to the Docker daemon" in output:
            return SandboxResult(
                success=False,
                output=output,
                error="Docker daemon not available",
                return_code=rc,
            )
        return SandboxResult(success=rc == 0, output=output, return_code=rc)


async def run_pytest_sandboxed(
    test_paths: list[str] | None = None,
    *,
    config: Any | None = None,
    timeout: float = 120.0,
    host_fallback: bool = True,
) -> tuple[bool, str]:
    """Run pytest inside docker sandbox when sandbox_mode=docker; else delegate locally.

    With ``host_fallback=False`` a non-docker mode fails instead of running on the host.
    """
    from core.safe_code_editor import run_pytest

    mode = normalise_mode(sandbox_mode(config))
    if mode != "docker":
        if not host_fallback:
            return False, f"refused: sandbox_mode is {mode!r}, not docker; pytest not run"
        return await run_pytest(test_paths, timeout=timeout)

    from core.docker_sandbox import _docker_env, _docker_exe

    security = getattr(config, "security", None) if config else None
    image = getattr(security, "sandbox_image", "witsv3-sandbox") if security else "witsv3-sandbox"
    exe = _docker_exe()
    if not exe:
        return False, "docker CLI not found on PATH"
    # Project mount is read-only; keep pytest cache/temp off /workspace.
    args = [
        "pytest",
        "-q",
        "-o",
        "addopts=",
        "-o",
        "cache_dir=/tmp/pytest_cache",
        "--basetemp=/tmp/pytest_basetemp",
        "--tb=native",
        *(test_paths or []),
    ]
    # Reason: masking mounts beat a filtered copy: no per-run copy of thousands of files,
    # and docker orders mounts by depth, so the masks always land on top of /workspace.
    with tempfile.TemporaryDirectory(
        prefix="wits_pytest_masks_", ignore_cleanup_errors=True
    ) as tmp:
        cmd = [
            exe,
            "run",
            "--rm",
            *hardening_flags(pids=PYTEST_PIDS_LIMIT, memory=PYTEST_MEMORY, tmpfs=PYTEST_TMPFS),
            "-v",
            f"{PROJECT_ROOT}:/workspace:ro",
            *mask_mount_args(Path(PROJECT_ROOT), tmp),
            "-w",
            "/workspace",
            image,
            *args,
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=_docker_env(),
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return False, f"Docker pytest timed out after {timeout:.0f}s"
    output = stdout.decode(errors="replace")
    return (proc.returncode or 0) == 0, output[-4000:]
