"""Benchmark Ollama models across context sizes.

Standalone on purpose: depends only on httpx (plus optional PyYAML/psutil), so the owner can run
it on a bare Windows box without installing WitsV3. Each (model, ctx) pair is repeated --runs
times and every run gets its own table row, because medians hide the cold-load first run that is
usually the interesting one. Results are appended to a markdown file, never overwritten.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

DEFAULT_URL = "http://localhost:11434"
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
NUM_PREDICT = 256  # Cap output so a rambling model cannot stall the whole sweep.

# Roughly 600 tokens: a system prompt with tool schemas, then a request that needs a tool call.
DEFAULT_PROMPT = (
    "You are a tool-calling assistant. Respond ONLY with a JSON object of the form "
    '{"tool": <name>, "arguments": {...}}. Available tools:\n'
    + "\n".join(
        f"- {name}: {desc}. Arguments: {args}."
        for name, desc, args in [
            ("read_file", "Read a UTF-8 text file from the workspace", "path (string)"),
            ("write_file", "Write text to a file", "path (string), content (string)"),
            ("list_dir", "List entries in a directory", "path (string), recursive (boolean)"),
            ("web_search", "Search the web", "query (string), max_results (integer)"),
            ("run_python", "Execute a Python snippet", "code (string), timeout (integer)"),
            ("calculator", "Evaluate an arithmetic expression", "expression (string)"),
            ("memory_store", "Save a fact to long-term memory", "text (string), tags (list)"),
            ("memory_search", "Search long-term memory", "query (string), top_k (integer)"),
        ]
    )
    + "\n\nRules: pick exactly one tool, never invent tool names, never add commentary, and "
    "quote string arguments with double quotes. If a request needs several steps, perform the "
    'first step only. If no tool fits, answer with {"tool": "none", "arguments": {}}.\n\n'
    "Example request: What is 17 times 23?\n"
    'Example answer: {"tool": "calculator", "arguments": {"expression": "17*23"}}\n\n'
    "Example request: Find notes about the pool inspection form.\n"
    'Example answer: {"tool": "memory_search", "arguments": {"query": "pool inspection form", '
    '"top_k": 5}}\n\n'
    "Request: Look through the docs folder, find the file that describes the benchmark plan, "
    "and show me its contents. Start by listing the docs folder recursively."
)


def default_url() -> str:
    """Read the Ollama URL from config.yaml, falling back quietly if anything is missing."""
    try:
        import yaml

        data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
        return str(data["ollama_settings"]["url"])
    except Exception:
        return DEFAULT_URL


def _rate(count: int | None, duration_ns: int | None) -> float | None:
    """Tokens per second from Ollama's count and nanosecond duration, or None if unusable."""
    if not count or not duration_ns:
        return None
    return count / (duration_ns / 1e9)


async def run_once(
    client: httpx.AsyncClient, model: str, ctx: int, prompt: str, timeout: float
) -> dict[str, Any]:
    """Stream one generation and return its metrics (raises on transport/HTTP errors)."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {"num_ctx": ctx, "num_predict": NUM_PREDICT, "temperature": 0.0},
    }
    start = time.perf_counter()
    ttft: float | None = None
    final: dict[str, Any] = {}
    async with client.stream("POST", "/api/generate", json=payload, timeout=timeout) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.strip():
                continue
            chunk = json.loads(line)
            if chunk.get("error"):
                raise RuntimeError(chunk["error"])
            # Thinking models emit "thinking" tokens before "response"; both count as first token.
            if ttft is None and (chunk.get("response") or chunk.get("thinking")):
                ttft = time.perf_counter() - start
            if chunk.get("done"):
                final = chunk
    if not final:
        raise RuntimeError("stream ended without a final 'done' chunk")
    return {
        "ttft_s": ttft,
        "tps": _rate(final.get("eval_count"), final.get("eval_duration")),
        "prompt_tps": _rate(final.get("prompt_eval_count"), final.get("prompt_eval_duration")),
        "load_s": (final.get("load_duration") or 0) / 1e9,
    }


def _same_model(a: str, b: str) -> bool:
    """Treat 'x' and 'x:latest' as the same model, as Ollama reports the tagged name."""

    def norm(name: str) -> str:
        return name[: -len(":latest")] if name.endswith(":latest") else name

    return norm(a) == norm(b)


async def unload(client: httpx.AsyncClient, model: str) -> None:
    """Evict the model (keep_alive=0) so the next model's CPU/GPU split and VRAM are clean."""
    try:
        await client.post("/api/generate", json={"model": model, "keep_alive": 0}, timeout=30)
    except Exception:
        pass


async def gpu_split(client: httpx.AsyncClient, model: str) -> str | None:
    """Return a 'CPU%/GPU%' string for the loaded model from /api/ps, or None."""
    try:
        resp = await client.get("/api/ps", timeout=10)
        resp.raise_for_status()
        for entry in resp.json().get("models", []):
            if model in (entry.get("name"), entry.get("model")) or _same_model(
                model, entry.get("name", "")
            ):
                size, vram = entry.get("size") or 0, entry.get("size_vram") or 0
                if size:
                    gpu = round(100 * vram / size)
                    return f"{100 - gpu}%/{gpu}%"
    except Exception:
        pass
    return None


async def gpu_mem_mib() -> int | None:
    """Sample GPU memory used via nvidia-smi; None when it is absent or fails."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
        # Multi-GPU boxes print one line per card; report the sum.
        return sum(int(v) for v in out.decode().split())
    except Exception:
        return None


def ram_used_mib() -> int | None:
    """System RAM in use (MiB) if psutil is installed, else None."""
    try:
        import psutil

        return int(psutil.virtual_memory().used / 2**20)
    except Exception:
        return None


async def benchmark(
    client: httpx.AsyncClient,
    models: list[str],
    ctxs: list[int],
    runs: int,
    prompt: str,
    timeout: float = 300,
) -> list[dict[str, Any]]:
    """Run every (model, ctx, run) combination; failures become rows and the sweep continues."""
    rows: list[dict[str, Any]] = []
    for model in models:
        for ctx in ctxs:
            for run in range(1, runs + 1):
                row: dict[str, Any] = {"model": model, "ctx": ctx, "run": run}
                try:
                    row.update(await run_once(client, model, ctx, prompt, timeout))
                    row["split"] = await gpu_split(client, model)
                    row["vram"] = await gpu_mem_mib()
                    row["ram"] = ram_used_mib()
                except httpx.TimeoutException:
                    row["error"] = f"timeout after {timeout:g}s"
                except Exception as exc:
                    row["error"] = " ".join(f"{type(exc).__name__}: {exc}".split())[:120]
                rows.append(row)
                print(f"{model} ctx={ctx} run={run}: {row.get('error', 'ok')}", flush=True)
                if "error" in row:
                    # Fail fast: a missing model should not cost runs x timeout.
                    break
        await unload(client, model)
    return rows


def _fmt(value: float | int | None, spec: str = ".1f") -> str:
    return "-" if value is None else format(value, spec)


def format_table(rows: list[dict[str, Any]], url: str, runs: int) -> str:
    """Render one timestamped markdown section; one row per run (no aggregation)."""
    host = f"{platform.node()} ({platform.system()} {platform.release()})"
    lines = [
        f"## Benchmark {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Host: {host} | Ollama: {url} | {runs} run(s) per model/ctx, one row per run "
        "(run 1 includes cold model load)",
        "",
        "| Model | Ctx | Run | TTFT (s) | Gen tok/s | Prompt tok/s | Load (s) | CPU/GPU "
        "| VRAM used (MiB) | RAM used (MiB) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        head = f"| {r['model']} | {r['ctx']} | {r['run']} |"
        if "error" in r:
            lines.append(f"{head} FAILED: {r['error'].replace('|', '/')} | | | | | | |")
            continue
        lines.append(
            f"{head} {_fmt(r['ttft_s'], '.2f')} | {_fmt(r['tps'])} | {_fmt(r['prompt_tps'])} "
            f"| {_fmt(r['load_s'], '.2f')} | {r['split'] or '-'} | {_fmt(r['vram'], 'd')} "
            f"| {_fmt(r['ram'], 'd')} |"
        )
    return "\n".join(lines) + "\n\n"


def append_results(path: Path, text: str) -> None:
    """Append to the results file, creating parent directories; never truncates."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(text)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser (kept free of I/O so --help works anywhere)."""
    p = argparse.ArgumentParser(description="Benchmark Ollama models across context sizes.")
    p.add_argument("--models", required=True, help="Comma-separated model names")
    p.add_argument("--ctx", default="8192,32768", help="Comma-separated num_ctx values")
    p.add_argument("--runs", type=int, default=3, help="Runs per model/ctx (one row each)")
    p.add_argument("--url", default=None, help="Ollama URL (default: config.yaml, then localhost)")
    p.add_argument("--out", default="docs/benchmarks/results.md", help="Markdown file to append to")
    p.add_argument("--prompt-file", default=None, help="Text file to use instead of the built-in")
    p.add_argument("--timeout", type=float, default=300, help="Per-run timeout in seconds")
    return p


async def main_async(args: argparse.Namespace) -> int:
    """Run the sweep and append the table."""
    url = args.url or default_url()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    ctxs = [int(c) for c in args.ctx.split(",") if c.strip()]
    prompt = (
        Path(args.prompt_file).read_text(encoding="utf-8") if args.prompt_file else DEFAULT_PROMPT
    )
    async with httpx.AsyncClient(base_url=url) as client:
        rows = await benchmark(client, models, ctxs, args.runs, prompt, args.timeout)
    append_results(Path(args.out), format_table(rows, url, args.runs))
    print(f"Appended {len(rows)} row(s) to {args.out}")
    return 0


def main() -> int:
    """CLI entry point."""
    return asyncio.run(main_async(build_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
