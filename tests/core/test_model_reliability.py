"""Tests for the Ollama health probe in ModelReliabilityManager.

`_check_model_health` used to only ever look at recorded failure history
("TODO: Implement actual health check by sending a small test request"), so
a model with zero recorded failures stayed HEALTHY/UNKNOWN even while Ollama
itself was completely unreachable. These tests cover the real `/api/tags`
probe added to close that gap.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.config import WitsV3Config
from core.model_reliability import ModelReliabilityManager, ModelStatus


@pytest.fixture
def manager() -> ModelReliabilityManager:
    config = WitsV3Config()
    return ModelReliabilityManager(config)


def _client_cm(response=None, raise_exc=None):
    """Build a mock usable as `async with httpx.AsyncClient(...) as client`."""
    client = MagicMock()
    if raise_exc is not None:
        client.get = AsyncMock(side_effect=raise_exc)
    else:
        client.get = AsyncMock(return_value=response)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=client)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


def _response(models):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value={"models": [{"name": m} for m in models]})
    return resp


@pytest.mark.asyncio
async def test_probe_ollama_reports_unreachable_on_connection_error(manager):
    with patch(
        "core.model_reliability.httpx.AsyncClient",
        return_value=_client_cm(raise_exc=ConnectionError("down")),
    ):
        reachable, models = await manager._probe_ollama()
    assert reachable is False
    assert models is None


@pytest.mark.asyncio
async def test_probe_ollama_returns_tag_names_on_success(manager):
    resp = _response(["qwen3:8b", "llama3.2:3b"])
    with patch("core.model_reliability.httpx.AsyncClient", return_value=_client_cm(response=resp)):
        reachable, models = await manager._probe_ollama()
    assert reachable is True
    assert "qwen3:8b" in models
    assert "qwen3" in models  # base name (without tag) also included


@pytest.mark.asyncio
async def test_check_model_health_marks_degraded_when_ollama_unreachable(manager):
    manager.model_health["qwen3:8b"].status = ModelStatus.HEALTHY
    await manager._check_model_health("qwen3:8b", ollama_reachable=False, available_models=None)
    assert manager.model_health["qwen3:8b"].status == ModelStatus.DEGRADED


@pytest.mark.asyncio
async def test_check_model_health_marks_degraded_when_model_not_pulled(manager):
    manager.model_health["qwen3:8b"].status = ModelStatus.HEALTHY
    await manager._check_model_health(
        "qwen3:8b", ollama_reachable=True, available_models={"llama3.2:3b", "llama3.2"}
    )
    assert manager.model_health["qwen3:8b"].status == ModelStatus.DEGRADED


@pytest.mark.asyncio
async def test_check_model_health_recovers_when_model_present_and_no_failures(manager):
    manager.model_health["qwen3:8b"].status = ModelStatus.DEGRADED
    await manager._check_model_health(
        "qwen3:8b", ollama_reachable=True, available_models={"qwen3:8b", "qwen3"}
    )
    assert manager.model_health["qwen3:8b"].status == ModelStatus.HEALTHY


@pytest.mark.asyncio
async def test_check_model_health_tolerates_untagged_configured_name(manager):
    """Configured model name has no tag (e.g. "nomic-embed-text"); Ollama's
    tag list reports it with an implicit ":latest" tag — should still match."""
    manager.model_health["nomic-embed-text"] = manager.model_health.get(
        "nomic-embed-text"
    ) or manager._ensure_model_health("nomic-embed-text")
    manager.model_health["nomic-embed-text"].status = ModelStatus.UNKNOWN

    await manager._check_model_health(
        "nomic-embed-text",
        ollama_reachable=True,
        available_models={"nomic-embed-text:latest", "nomic-embed-text"},
    )
    assert manager.model_health["nomic-embed-text"].status == ModelStatus.HEALTHY


@pytest.mark.asyncio
async def test_check_all_models_health_probes_once_for_all_models(manager):
    resp = _response(["qwen3:8b", "llama3.2:3b", "qwen2.5-coder:7b", "nomic-embed-text"])
    with patch(
        "core.model_reliability.httpx.AsyncClient", return_value=_client_cm(response=resp)
    ) as mock_client_cls:
        await manager._check_all_models_health()

    assert mock_client_cls.call_count == 1  # one probe, not one per tracked model
    for health in manager.model_health.values():
        assert health.status == ModelStatus.HEALTHY
