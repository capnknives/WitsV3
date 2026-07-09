"""Unit tests for web.ollama_admin."""

from types import SimpleNamespace

import pytest

from web.ollama_admin import (
    build_ollama_status,
    collect_configured_models,
    model_is_available,
    pull_ollama_model,
    validate_model_name,
)


def _config(**kwargs):
    ollama = SimpleNamespace(
        url="http://127.0.0.1:11434",
        default_model="qwen3:8b",
        orchestrator_model="qwen3:8b",
        embedding_model="nomic-embed-text",
    )
    mr = SimpleNamespace(
        enabled=False,
        trivial_model="llama3.2:3b",
        code_model="qwen2.5-coder:7b",
        complex_model="qwen3:8b",
    )
    return SimpleNamespace(ollama_settings=ollama, model_routing=mr, **kwargs)


def test_validate_model_name_rejects_empty():
    with pytest.raises(ValueError):
        validate_model_name("  ")


def test_model_is_available_matches_base_name():
    available = {"nomic-embed-text:latest", "qwen3"}
    assert model_is_available("nomic-embed-text", available)
    assert model_is_available("qwen3:8b", available)
    assert not model_is_available("missing:7b", available)


def test_collect_configured_models_dedupes():
    cfg = _config()
    names = [m["name"] for m in collect_configured_models(cfg)]
    assert names == ["qwen3:8b", "nomic-embed-text"]


def test_collect_configured_models_includes_routing_when_enabled():
    cfg = _config()
    cfg.model_routing.enabled = True
    names = [m["name"] for m in collect_configured_models(cfg)]
    assert "qwen2.5-coder:7b" in names
    assert "llama3.2:3b" in names


@pytest.mark.asyncio
async def test_build_ollama_status_marks_missing(monkeypatch):
    async def fake_probe(url, timeout=5.0):
        return True, {"qwen3:8b"}

    monkeypatch.setattr("web.ollama_admin.probe_ollama", fake_probe)
    status = await build_ollama_status(_config())
    assert status["reachable"] is True
    by_name = {m["name"]: m for m in status["configured_models"]}
    assert by_name["qwen3:8b"]["installed"] is True
    assert by_name["nomic-embed-text"]["installed"] is False
    assert status["all_installed"] is False


@pytest.mark.asyncio
async def test_build_ollama_status_unreachable(monkeypatch):
    async def fake_probe(url, timeout=5.0):
        return False, set()

    monkeypatch.setattr("web.ollama_admin.probe_ollama", fake_probe)
    status = await build_ollama_status(_config())
    assert status["reachable"] is False
    assert status["available_models"] == []
    assert all(not m["installed"] for m in status["configured_models"])


@pytest.mark.asyncio
async def test_pull_ollama_model_success(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"status": "success"}

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, json):
            assert json["name"] == "qwen3:8b"
            assert json["stream"] is False
            return FakeResponse()

    monkeypatch.setattr("web.ollama_admin.httpx.AsyncClient", lambda **kwargs: FakeClient())
    result = await pull_ollama_model("http://127.0.0.1:11434", "qwen3:8b")
    assert result["model"] == "qwen3:8b"
    assert result["status"] == "success"
