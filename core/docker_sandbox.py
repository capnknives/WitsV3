"""Ensure Docker Desktop and the witsv3-sandbox image are ready."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from core.safe_code_editor import PROJECT_ROOT
from core.sandbox_runner import sandbox_mode

logger = logging.getLogger(__name__)

_DOCKER_DESKTOP_PATHS = (
    Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
    / "Docker"
    / "Docker"
    / "Docker Desktop.exe",
    Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
    / "Docker"
    / "Docker"
    / "Docker Desktop.exe",
)

_WAIT_TIMEOUT_SEC = 180
_POLL_INTERVAL_SEC = 3


def _docker_exe() -> str | None:
    found = shutil.which("docker")
    if found:
        return found
    if sys.platform == "win32":
        win_cli = (
            Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
            / "Docker"
            / "Docker"
            / "resources"
            / "bin"
            / "docker.exe"
        )
        if win_cli.is_file():
            return str(win_cli)
    return None


def _docker_env() -> dict[str, str]:
    """Ensure Docker credential helpers resolve on Windows shells with minimal PATH."""
    env = dict(os.environ)
    if sys.platform == "win32":
        docker_bin = (
            Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
            / "Docker"
            / "Docker"
            / "resources"
            / "bin"
        )
        if docker_bin.is_dir():
            prefix = str(docker_bin)
            if prefix not in env.get("PATH", ""):
                env["PATH"] = f"{prefix};{env.get('PATH', '')}"
    return env


def _docker_desktop_exe() -> Path | None:
    for candidate in _DOCKER_DESKTOP_PATHS:
        if candidate.is_file():
            return candidate
    return None


async def _run(cmd: list[str], *, timeout: float = 60.0) -> tuple[int, str]:
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
        return -1, f"Command timed out: {' '.join(cmd)}"
    return proc.returncode or 0, stdout.decode(errors="replace")


async def docker_daemon_ready() -> bool:
    if not _docker_exe():
        return False
    code, _ = await _run(["docker", "info"], timeout=15.0)
    return code == 0


async def start_docker_desktop_if_needed() -> bool:
    """Launch Docker Desktop on Windows when the daemon is not up."""
    if await docker_daemon_ready():
        return True
    if sys.platform != "win32":
        logger.warning("Docker daemon not ready and auto-start is Windows-only")
        return False
    desktop = _docker_desktop_exe()
    if desktop is None:
        logger.error("Docker Desktop executable not found")
        return False
    logger.info("Starting Docker Desktop: %s", desktop)
    subprocess.Popen([str(desktop)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=_docker_env())
    elapsed = 0.0
    while elapsed < _WAIT_TIMEOUT_SEC:
        if await docker_daemon_ready():
            logger.info("Docker daemon is ready (%.0fs)", elapsed)
            return True
        await asyncio.sleep(_POLL_INTERVAL_SEC)
        elapsed += _POLL_INTERVAL_SEC
    logger.error("Docker daemon did not become ready within %.0fs", _WAIT_TIMEOUT_SEC)
    return False


async def ensure_sandbox_image(config: Any | None = None) -> tuple[bool, str]:
    """Build witsv3-sandbox when missing."""
    security = getattr(config, "security", None) if config else None
    image = getattr(security, "sandbox_image", "witsv3-sandbox") if security else "witsv3-sandbox"
    if not _docker_exe():
        return False, "docker CLI not found on PATH"
    code, out = await _run(["docker", "image", "inspect", image], timeout=30.0)
    if code == 0:
        health_code, _ = await _run(
            ["docker", "run", "--rm", image, "python", "-c", "import pydantic, pytest"],
            timeout=60.0,
        )
        if health_code == 0:
            return True, f"Image {image} present"
        logger.warning("Sandbox image %s failed dependency health check; rebuilding", image)
    dockerfile = PROJECT_ROOT / "Dockerfile.sandbox"
    if not dockerfile.is_file():
        return False, f"Missing {dockerfile}"
    logger.info("Building sandbox image %s from %s", image, dockerfile)
    code, out = await _run(
        ["docker", "build", "-f", str(dockerfile), "-t", image, str(PROJECT_ROOT)],
        timeout=600.0,
    )
    if code != 0:
        return False, out[-2000:]
    return True, f"Built image {image}"


async def ensure_docker_sandbox_ready(config: Any | None = None) -> tuple[bool, str]:
    """Full preflight when sandbox_mode=docker."""
    if sandbox_mode(config) != "docker":
        return True, "sandbox_mode is not docker"
    if not await start_docker_desktop_if_needed():
        return False, "Docker Desktop is not running and could not be started"
    ok, detail = await ensure_sandbox_image(config)
    return ok, detail


def ensure_docker_sandbox_ready_sync(config: Any | None = None) -> tuple[bool, str]:
    """Blocking wrapper for startup paths that are not yet async."""
    return asyncio.run(ensure_docker_sandbox_ready(config))
