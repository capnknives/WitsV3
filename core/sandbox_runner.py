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

from core.safe_code_editor import PROJECT_ROOT

logger = logging.getLogger(__name__)


@dataclass
class SandboxResult:
    success: bool
    output: str
    error: str = ""
    return_code: int = 0


def sandbox_mode(config: Any | None) -> str:
    security = getattr(config, "security", None) if config else None
    mode = getattr(security, "sandbox_mode", "off") if security else "off"
    return str(mode or "off").lower()


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
        return await _run_subprocess_sandbox(code, config=config, timeout=timeout)
    if mode == "docker":
        return await _run_docker_sandbox(code, config=config, timeout=timeout)
    return SandboxResult(success=False, output="", error=f"unknown sandbox_mode: {mode}", return_code=-1)


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
            env={**os.environ, **env},
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


async def _run_docker_sandbox(
    code: str, *, config: Any | None, timeout: float
) -> SandboxResult:
    security = getattr(config, "security", None) if config else None
    image = getattr(security, "sandbox_image", "witsv3-sandbox") if security else "witsv3-sandbox"
    with tempfile.TemporaryDirectory(prefix="wits_docker_sandbox_") as tmp:
        script = Path(tmp) / "user_code.py"
        script.write_text(code, encoding="utf-8")
        cmd = [
            "docker",
            "run",
            "--rm",
            "--network=none",
            "-v",
            f"{tmp}:/sandbox:ro",
            image,
            "python",
            "/sandbox/user_code.py",
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
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
) -> tuple[bool, str]:
    """Run pytest inside docker sandbox when sandbox_mode=docker; else delegate locally."""
    from core.safe_code_editor import run_pytest

    mode = sandbox_mode(config)
    if mode != "docker":
        return await run_pytest(test_paths, timeout=timeout)

    security = getattr(config, "security", None) if config else None
    image = getattr(security, "sandbox_image", "witsv3-sandbox") if security else "witsv3-sandbox"
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
    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{PROJECT_ROOT}:/workspace:ro",
        "-w",
        "/workspace",
        image,
        *args,
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return False, f"Docker pytest timed out after {timeout:.0f}s"
    output = stdout.decode(errors="replace")
    return (proc.returncode or 0) == 0, output[-4000:]
