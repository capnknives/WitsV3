"""Tests for scripts/bench_models.py using a mocked httpx transport."""

import importlib.util
import json
import sys
from pathlib import Path

import httpx
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "bench_models.py"


@pytest.fixture(scope="module")
def bench():
    """Load the script by path; scripts/ is not a package."""
    spec = importlib.util.spec_from_file_location("bench_models", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["bench_models"] = module
    spec.loader.exec_module(module)
    return module


def _stream_body() -> bytes:
    chunks = [
        {"response": "Hel", "done": False},
        {"response": "lo", "done": False},
        {
            "response": "",
            "done": True,
            "eval_count": 100,
            "eval_duration": 2_000_000_000,
            "prompt_eval_count": 600,
            "prompt_eval_duration": 500_000_000,
            "load_duration": 1_500_000_000,
        },
    ]
    return "\n".join(json.dumps(c) for c in chunks).encode()


UNLOADS: list[str] = []


def _handler(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/api/generate":
        body = json.loads(request.content)
        if body.get("keep_alive") == 0:
            UNLOADS.append(body["model"])
            return httpx.Response(200, json={"done": True})
        if body["model"] == "broken":
            return httpx.Response(500, text="boom")
        assert body["stream"] is True
        assert body["options"]["num_ctx"] in (4096, 8192)
        return httpx.Response(200, content=_stream_body())
    if request.url.path == "/api/ps":
        return httpx.Response(
            200, json={"models": [{"name": "good:latest", "size": 100, "size_vram": 75}]}
        )
    return httpx.Response(404)


async def test_metrics_and_failure_isolation(bench, monkeypatch):
    async def no_gpu():
        return None

    monkeypatch.setattr(bench, "gpu_mem_mib", no_gpu)
    UNLOADS.clear()
    transport = httpx.MockTransport(_handler)
    async with httpx.AsyncClient(transport=transport, base_url="http://x") as client:
        rows = await bench.benchmark(client, ["broken", "good"], [4096, 8192], 2, "hi")

    assert len(rows) == 6  # broken fails fast: 1 row per ctx, not runs
    broken = [r for r in rows if r["model"] == "broken"]
    assert len(broken) == 2 and all("error" in r for r in broken)
    assert UNLOADS == ["broken", "good"]  # once per model, failures included
    good = [r for r in rows if r["model"] == "good"]
    assert len(good) == 4
    assert good[0]["tps"] == pytest.approx(50.0)
    assert good[0]["prompt_tps"] == pytest.approx(1200.0)
    assert good[0]["load_s"] == pytest.approx(1.5)
    assert good[0]["ttft_s"] is not None
    assert good[0]["split"] == "25%/75%"
    assert good[0]["vram"] is None


async def test_table_is_appended(bench, tmp_path):
    rows = [
        {"model": "m", "ctx": 8192, "run": 1, "ttft_s": 0.5, "tps": 50.0, "prompt_tps": 1200.0,
         "load_s": 1.5, "split": "0%/100%", "vram": 5000, "ram": None},
        {"model": "bad", "ctx": 8192, "run": 1, "error": "timeout after 5s"},
    ]  # fmt: skip
    out = tmp_path / "sub" / "results.md"
    bench.append_results(out, bench.format_table(rows, "http://x", 1))
    bench.append_results(out, bench.format_table(rows, "http://x", 1))
    text = out.read_text(encoding="utf-8")
    assert text.count("## Benchmark") == 2
    assert "| m | ollama | 8192 | 1 | 0.50 | 50.0 | 1200.0 | 1.50 | 0%/100% | 5000 | - |" in text
    assert "FAILED: timeout after 5s" in text
    assert "Host:" in text


def test_default_url_falls_back(bench, monkeypatch, tmp_path):
    monkeypatch.setattr(bench, "CONFIG_PATH", tmp_path / "missing.yaml")
    assert bench.default_url() == "http://localhost:11434"


def _sse(chunks: list[dict]) -> bytes:
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    return body.encode()


def _openai_client(content: bytes, seen: list[dict]) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        seen.append(json.loads(request.content))
        return httpx.Response(200, content=content)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://x")


async def test_openai_backend_uses_server_timings(bench, monkeypatch):
    async def no_gpu():
        return None

    monkeypatch.setattr(bench, "gpu_mem_mib", no_gpu)
    body = _sse(
        [
            {"choices": [{"delta": {"content": "Hel"}}]},
            {"choices": [{"delta": {"content": "lo"}}]},
            {
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "timings": {
                    "prompt_n": 600,
                    "prompt_per_second": 1100.5,
                    "predicted_n": 100,
                    "predicted_per_second": 42.5,
                },
            },
        ]
    )
    seen: list[dict] = []
    async with _openai_client(body, seen) as client:
        rows = await bench.benchmark(client, ["m"], [8192], 1, "hi", backend="openai")
    row = rows[0]
    assert row["tps"] == pytest.approx(42.5)
    assert row["prompt_tps"] == pytest.approx(1100.5)
    assert row["ttft_s"] is not None and row["load_s"] is None and row["split"] is None
    assert seen[0]["stream"] is True and seen[0]["temperature"] == 0.0
    assert seen[0]["max_tokens"] == bench.NUM_PREDICT
    assert seen[0]["model"] == "m"
    assert "num_ctx" not in json.dumps(seen[0])


async def test_openai_backend_wall_clock_fallback(bench, monkeypatch):
    async def no_gpu():
        return None

    monkeypatch.setattr(bench, "gpu_mem_mib", no_gpu)
    body = _sse(
        [
            {"choices": [{"delta": {"content": "Hel"}}]},
            {"choices": [{"delta": {"content": "lo"}}]},
            {"choices": [], "usage": {"prompt_tokens": 600, "completion_tokens": 100}},
        ]
    )
    async with _openai_client(body, []) as client:
        rows = await bench.benchmark(client, ["m"], [8192], 1, "hi", backend="openai")
    row = rows[0]
    assert "error" not in row
    assert row["tps"] is not None and row["tps"] > 0
    assert row["prompt_tps"] is not None and row["prompt_tps"] > 0


def test_table_has_backend_and_label(bench):
    rows = [
        {"model": "m", "backend": "openai", "ctx": 8192, "run": 1, "ttft_s": 0.5, "tps": 42.5,
         "prompt_tps": 1100.0, "load_s": None, "split": None, "vram": None, "ram": None},
    ]  # fmt: skip
    text = bench.format_table(rows, "http://x", 2, label="cache=1728 share=0.90")
    assert "| Model | Backend | Ctx |" in text
    assert "| m | openai | 8192 | 1 | 0.50 | 42.5 | 1100.0 | - | - | - | - |" in text
    assert "Label: cache=1728 share=0.90" in text
    assert "Server: http://x" in text
