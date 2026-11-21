---
title: "Stage 0 Briefs — Measure the Box"
created: "2026-11-21"
last_updated: "2026-11-21"
status: "active"
---

# Stage 0 briefs

Rules and the review ladder are in [README.md](README.md). Nothing in Stage 0
changes WITS behaviour: it adds two scripts and one results document.

## WP 0.1: Benchmark script

```
Tier: Sonnet, medium
Branch: claude/wp-0.1-bench-script, from origin/main
Read: scripts/ollama_probe.py (reuse its client setup), core/llm_interface.py
      lines 85-115 (payload shape), config.yaml lines 107-128
Do not read: docs/implementation/, docs/archive/, docs/roadmap/revival-2026-07.md
Build:
  - scripts/bench_models.py, standalone, httpx only, async
  - args: --models (comma list), --ctx (default "8192,32768"), --runs (default 3),
    --url (default from config.yaml), --out (default docs/benchmarks/results.md),
    --prompt-file (default: a built-in tool-calling style prompt, about 600 tokens)
  - per model and context size: call /api/generate with stream=true, record
    time to first token, eval_count / eval_duration as tokens per second,
    prompt_eval rate, load_duration; then read /api/ps for the CPU/GPU split
  - sample `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits`
    and system RAM (psutil if present, else skip) at the end of each run
  - write a markdown table (append, never overwrite) with a timestamp and host line
  - a model that fails or times out gets a row saying so; the run continues
Acceptance:
  - tests/tools/test_bench_models.py with a mocked httpx transport: parses a
    fake stream, computes tokens per second, writes a table, survives one failing model
  - `python scripts/bench_models.py --help` exits 0 without Ollama running
Run: pytest tests/tools/test_bench_models.py -q --no-cov; ruff check scripts/bench_models.py tests/tools/test_bench_models.py; black --check scripts/bench_models.py tests/tools/test_bench_models.py
Out of scope: llama.cpp, changing config.yaml, changing any module under core/
Report: files changed, test tail, open questions. 20 lines at most.
```

## WP 0.2: Compute-mode script

```
Tier: Haiku, medium
Branch: claude/wp-0.2-compute-mode, from origin/main
Read: scripts/ensure_docker_desktop.ps1 (house style for PowerShell scripts)
Build:
  - scripts/compute_mode.ps1
  - prints free and total VRAM (nvidia-smi) and free and total RAM
  - lists the ten processes using the most RAM, and every process holding the GPU
  - prints a one-line verdict against two thresholds passed as parameters
    (-MinFreeVramMb default 6500, -MinFreeRamGb default 24)
  - closes nothing. It reports only
Acceptance: the script parses (`pwsh -NoProfile -Command "& { . ./scripts/compute_mode.ps1 -WhatIf }"` or a
  Parser::ParseFile check in the PR description) and has a comment header with usage
Out of scope: killing processes, changing power or GPU settings
Report: files changed, parse-check output, open questions. 20 lines at most.
```

## 0.3: Benchmark run (owner)

After 0.1 and 0.2 merge:

```powershell
git fetch origin; git switch --detach origin/main   # in a scratch clone, not the runtime folder
.\scripts\compute_mode.ps1
python scripts\bench_models.py --models "qwen3:8b,qwen3.5:9b,gemma4:12b,gemma4:e4b,qwen3-8b-fast,gemma4:26b,qwen3.5:35b-a3b"
```

Close browsers first if the verdict line says memory is short. Commit
`docs/benchmarks/results.md` on a branch and open a PR, or paste the table to
the lead.

## WP 0.4: llama.cpp MoE sweep

Starts only after the owner approves downloading a CUDA build of llama.cpp.

```
Tier: Sonnet, medium
Branch: claude/wp-0.4-moe-sweep, from origin/main
Read: scripts/bench_models.py (from 0.1)
Build:
  - extend bench_models.py with --backend openai and --base-url, so the same
    measurements run against an OpenAI-compatible server (/v1/chat/completions,
    stream=true)
  - scripts/moe_sweep.ps1: starts llama-server for a given GGUF with
    -ngl 999 --n-cpu-moe N for each N in a list, waits for /health, runs
    bench_models.py against it, stops the server, moves to the next N
  - parameters: -ModelPath, -LlamaServerPath, -Values (default 16,20,24,28,32), -Ctx
Acceptance: mocked test for the openai backend path; moe_sweep.ps1 parses
Run: pytest tests/tools/test_bench_models.py -q --no-cov; ruff and black on changed Python files
Out of scope: installing llama.cpp, downloading models, wiring llama-server into WITS
Report: files changed, test tail, open questions. 20 lines at most.
```

## 0.5: Lineup decision (lead)

From the results table, name one model each for fast, heavy and embedding, and
say whether `llama-server` beat Ollama's own placement on the two large MoE
models by enough to justify a second server. Record it in
[`../unified-plan-2026-11.md`](../unified-plan-2026-11.md) under "Still open".
The owner confirms.
