"""Serving settings for OllamaInterface (keep_alive, num_ctx, think, CPU embeddings)."""

import json
from unittest.mock import patch

import httpx
import pytest

from core.config import WitsV3Config
from core.llm_interface import LLMMessage, OllamaInterface

THINKERS = {"think-model"}


def _make(embedding_on_cpu: bool = True):
    """Build an interface whose HTTP transport is mocked and records requests."""
    config = WitsV3Config()
    config.ollama_settings.url = "http://ollama.test"
    config.ollama_settings.default_model = "plain-model"
    config.ollama_settings.embedding_on_cpu = embedding_on_cpu
    seen: list[tuple[str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append((request.url.path, body))
        path = request.url.path
        if path == "/api/show":
            caps = ["completion", "thinking"] if body["model"] in THINKERS else ["completion"]
            return httpx.Response(200, json={"capabilities": caps})
        if path == "/api/chat":
            return httpx.Response(200, json={"message": {"content": "hi"}})
        if path == "/api/generate":
            if body.get("stream"):
                lines = [{"response": "a", "done": False}, {"response": "", "done": True}]
                return httpx.Response(200, text="\n".join(json.dumps(x) for x in lines))
            return httpx.Response(200, json={"response": "text"})
        return httpx.Response(200, json={"embedding": [0.1, 0.2]})

    iface = OllamaInterface(config)
    iface.http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return iface, seen


def _bodies(seen, path):
    return [b for p, b in seen if p == path]


@pytest.mark.asyncio
async def test_chat_payload_uses_settings_and_arguments():
    iface, seen = _make()
    iface.ollama_settings.num_ctx = 4096
    iface.ollama_settings.keep_alive = "10m"
    result = await iface.generate_chat([LLMMessage(role="user", content="x")])
    body = _bodies(seen, "/api/chat")[-1]
    assert body["options"]["num_ctx"] == 4096
    assert body["keep_alive"] == "10m"
    assert result.content == "hi" and result.tool_calls is None

    await iface.generate_chat([LLMMessage(role="user", content="x")], num_ctx=2048, keep_alive="1h")
    body = _bodies(seen, "/api/chat")[-1]
    assert body["options"]["num_ctx"] == 2048
    assert body["keep_alive"] == "1h"


@pytest.mark.asyncio
async def test_think_only_for_thinking_models_and_show_cached():
    iface, seen = _make()
    msgs = [LLMMessage(role="user", content="x")]
    await iface.generate_chat(msgs)
    await iface.generate_chat(msgs)
    assert all("think" not in b for b in _bodies(seen, "/api/chat"))

    await iface.generate_chat(msgs, model="think-model")
    await iface.generate_chat(msgs, model="think-model", think=True)
    first, second = _bodies(seen, "/api/chat")[-2:]
    assert first["think"] is False and second["think"] is True

    shows = [b["model"] for b in _bodies(seen, "/api/show")]
    assert sorted(shows) == ["plain-model", "think-model"]


@pytest.mark.asyncio
async def test_show_failure_means_no_thinking():
    iface, seen = _make()
    with patch.object(iface.http_client, "post", side_effect=httpx.ConnectError("down")):
        assert await iface._supports_thinking("think-model") is False
    assert await iface._supports_thinking("think-model") is False  # cached


@pytest.mark.asyncio
async def test_embedding_cpu_option_follows_setting():
    iface, seen = _make()
    assert await iface.get_embedding("hello") == [0.1, 0.2]
    body = _bodies(seen, "/api/embeddings")[-1]
    assert body["options"] == {"num_gpu": 0}
    assert body["keep_alive"] == "30m"

    iface, seen = _make(embedding_on_cpu=False)
    await iface.get_embedding("hello")
    assert "options" not in _bodies(seen, "/api/embeddings")[-1]


@pytest.mark.asyncio
async def test_generate_and_stream_send_keep_alive_and_keep_returns():
    iface, seen = _make()
    assert await iface.generate_text("p") == "text"
    chunks = [c async for c in iface.stream_text("p")]
    assert chunks == ["a"]
    for body in _bodies(seen, "/api/generate"):
        assert body["keep_alive"] == "30m"
        assert body["options"]["num_ctx"] == iface.ollama_settings.num_ctx
        assert "think" not in body


@pytest.mark.asyncio
async def test_generate_and_stream_num_ctx_override_and_think():
    iface, seen = _make()
    await iface.generate_text("p", model="think-model", num_ctx=1234)
    _ = [c async for c in iface.stream_text("p", model="think-model")]
    first, second = _bodies(seen, "/api/generate")
    assert first["options"]["num_ctx"] == 1234
    assert second["options"]["num_ctx"] == iface.ollama_settings.num_ctx
    assert first["think"] is False and second["think"] is False


@pytest.mark.asyncio
async def test_chat_payload_carries_tool_calls_only_when_set():
    iface, seen = _make()
    calls = [{"function": {"name": "lookup", "arguments": {"term": "sun"}}}]
    await iface.generate_chat(
        [
            LLMMessage(role="user", content="x"),
            LLMMessage(role="assistant", content="", tool_calls=calls),
            LLMMessage(role="tool", content="r", name="lookup"),
        ]
    )
    sent = _bodies(seen, "/api/chat")[-1]["messages"]
    assert "tool_calls" not in sent[0] and "tool_calls" not in sent[2]
    assert sent[1]["tool_calls"] == calls
