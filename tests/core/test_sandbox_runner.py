"""Sandbox runner unit tests (no Docker required)."""

import asyncio

import pytest

from core.config import WitsV3Config
from core.sandbox_runner import run_python_sandboxed, run_pytest_sandboxed, sandbox_mode


def test_sandbox_mode_default_off():
    config = WitsV3Config()
    assert sandbox_mode(config) == "off"


@pytest.mark.asyncio
async def test_sandbox_off_returns_error():
    config = WitsV3Config()
    result = await run_python_sandboxed("print(1)", config=config)
    assert not result.success
    assert "not enabled" in result.error


def test_docker_pytest_args_use_tmp_cache(monkeypatch):
    """Docker pytest must not write cache under read-only /workspace."""
    captured: list[list[str]] = []

    class _FakeProc:
        returncode = 0

        async def communicate(self):
            return b"ok", b""

    async def _fake_run(*cmd, **kwargs):
        captured.append(list(cmd))
        return _FakeProc()

    monkeypatch.setattr("core.sandbox_runner.asyncio.create_subprocess_exec", _fake_run)
    config = WitsV3Config()
    config.security.sandbox_mode = "docker"
    asyncio.run(run_pytest_sandboxed(["tests/core/test_sandbox_runner.py"], config=config))
    assert captured
    joined = " ".join(captured[0])
    assert "cache_dir=/tmp/pytest_cache" in joined
    assert "--basetemp=/tmp/pytest_basetemp" in joined
