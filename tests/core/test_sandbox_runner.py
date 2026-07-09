"""Sandbox runner unit tests (no Docker required)."""

import pytest

from core.config import WitsV3Config
from core.sandbox_runner import run_python_sandboxed, sandbox_mode


def test_sandbox_mode_default_off():
    config = WitsV3Config()
    assert sandbox_mode(config) == "off"


@pytest.mark.asyncio
async def test_sandbox_off_returns_error():
    config = WitsV3Config()
    result = await run_python_sandboxed("print(1)", config=config)
    assert not result.success
    assert "not enabled" in result.error
