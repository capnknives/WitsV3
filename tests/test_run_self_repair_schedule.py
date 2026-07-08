"""Tests for the daily self-repair scheduling wired into WitsV3System (run.py)."""

from collections.abc import AsyncGenerator

import pytest

from core.config import WitsV3Config
from core.schemas import StreamData
from run import WitsV3System


def _system(cron: str = "0 3 * * *", enabled: bool = True) -> WitsV3System:
    config = WitsV3Config()
    config.self_repair.daily_schedule_enabled = enabled
    config.self_repair.daily_schedule_cron = cron
    return WitsV3System(config)


@pytest.mark.asyncio
async def test_initialize_self_repair_schedule_registers_cron_job():
    # AsyncIOScheduler.start() requires a running event loop.
    system = _system()
    system._initialize_self_repair_schedule()
    try:
        job = system.self_repair_scheduler.get_job("daily_self_repair")
        assert job is not None
    finally:
        system.self_repair_scheduler.shutdown(wait=False)


@pytest.mark.asyncio
async def test_initialize_self_repair_schedule_bad_cron_does_not_raise():
    system = _system(cron="not a cron expression")
    system._initialize_self_repair_schedule()  # should log a warning, not raise
    assert system.self_repair_scheduler is None or not system.self_repair_scheduler.get_jobs()


@pytest.mark.asyncio
async def test_run_scheduled_self_repair_noop_without_agent():
    system = _system()
    system.self_repair_agent = None
    await system._run_scheduled_self_repair()  # should not raise


@pytest.mark.asyncio
async def test_run_scheduled_self_repair_consumes_agent_stream():
    class FakeAgent:
        async def run(
            self, user_input, session_id=None, **kwargs
        ) -> AsyncGenerator[StreamData, None]:
            yield StreamData(type="thinking", content="scanning", source="SystemDoctor")
            yield StreamData(type="result", content="Repaired agents/foo.py", source="SystemDoctor")

    system = _system()
    system.self_repair_agent = FakeAgent()
    await system._run_scheduled_self_repair()  # should not raise, and should log the result


@pytest.mark.asyncio
async def test_run_scheduled_self_repair_never_raises_on_agent_error():
    class FailingAgent:
        async def run(
            self, user_input, session_id=None, **kwargs
        ) -> AsyncGenerator[StreamData, None]:
            raise RuntimeError("boom")
            yield  # pragma: no cover — makes this an async generator

    system = _system()
    system.self_repair_agent = FailingAgent()
    await system._run_scheduled_self_repair()  # must swallow the error
