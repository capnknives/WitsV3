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

## 0.4: llama.cpp MoE-cache test (lead, on the owner's PC)

Replaces the `--n-cpu-moe` sweep. Starts only after the owner approves
downloading a CUDA build of llama.cpp (v1.0.0, build b12359, or newer). It goes
in `D:\DevData\llama.cpp`, never on `C:`. Ollama stays installed and untouched.

Step one is a measurement, not a package. No repository code is needed.

1. Stop any loaded Ollama model. Find the `qwen3.5:35b-a3b` model file under
   `D:\OllamaModels\blobs` (no new model download).
2. Baseline: `llama-server -m <blob> --no-mmproj -ngl 99 --cpu-moe --fit off
   -fa on -c 8192 -np 1 -t 6 --host 127.0.0.1 --port 8080`. Send the benchmark
   prompt twice; record the second run's generation rate from the server's
   own timings.
3. Cache runs: add `--moe-cache-mib N --moe-cache-cpu-share 0.90` for N in
   1728, 2560, 3072. Record the rate and the cache hit rate the server logs on
   exit. Watch free VRAM; stop a run that spills.
4. Do not combine the cache with `--no-kv-offload`, and do not run
   `scripts/moe-cache-tune.py` on Windows (both reported broken upstream).
5. Append the rows to `docs/benchmarks/results.md` with the exact command lines.

Decision rule, against Ollama's 36 tokens per second on the same model:

- **Under 15% better:** stay on Ollama alone. Close 0.4.
- **15% or more:** write package 0.4b (below) and plan the heavy model on
  `llama-server` in Stage 2.

### WP 0.4b: benchmark backend for llama-server (only if 0.4 passes)

```
Tier: Sonnet, medium
Branch: claude/wp-0.4b-bench-openai, from origin/main
Read: scripts/bench_models.py, tests/tools/test_bench_models.py
Build:
  - bench_models.py: --backend openai and --base-url, so the same measurements
    run against /v1/chat/completions with stream=true (rates from the server's
    timings object when present, else from wall-clock)
  - scripts/moe_cache_sweep.ps1: starts llama-server for a model file with each
    cache size in a list, waits for /health, runs bench_models.py, stops the server
  - parameters: -ModelPath, -LlamaServerPath, -CacheMib (default 0,1728,2560,3072),
    -CpuShare (default 0.90), -Ctx
Acceptance: mocked test for the openai backend path; moe_cache_sweep.ps1 parses
Run: pytest tests/tools/test_bench_models.py -q --no-cov; ruff and black on changed Python files
Out of scope: installing llama.cpp, downloading models, wiring llama-server into WITS
Report: files changed, test tail, open questions. 20 lines at most.
```

## WP 0.6: Tool-calling test

The benchmark measured speed only. This measures whether a model calls tools
correctly, on both servers, because both research passes report tool-call
parsing bugs in Ollama for the Qwen 3.5 and Gemma 4 families.

```
Tier: Sonnet, medium
Branch: claude/wp-0.6-toolcall-eval, from origin/main
Read: scripts/bench_models.py (house style, CLI, table writer),
      tests/tools/test_bench_models.py (how the script is loaded and mocked)
Do not read: docs/implementation/, docs/archive/, docs/roadmap/revival-2026-07.md
Build:
  - scripts/toolcall_eval.py, standalone, async, httpx only (PyYAML for the cases file)
  - scripts/toolcall_cases.yaml: about 36 cases over one fixed set of 8 tools:
    12 single-call; 6 argument-hard (nested arrays, a parameter named `type` and
    one named `description`, a string containing <T=any> and one containing
    </tool_call>); 6 where no tool fits and the right answer is no call;
    4 parallel (two calls expected); 4 two-step chains (a canned tool result is
    fed back and a second call is expected); 4 error-recovery (a canned
    validation error is fed back and a corrected call is expected)
  - backends: --backend ollama (/api/chat with tools, non-streaming) and
    --backend openai (/v1/chat/completions with tools); --url; --models;
    --runs (default 3); --think on|off|both (default both, where the backend
    supports it); --timeout; --out (default docs/benchmarks/toolcall-results.md)
  - scoring is deterministic, no model judge: per case, valid native tool call
    returned (not JSON in text), right tool name(s), arguments equal to the
    expected values after JSON normalisation, no call on no-tool cases, no
    thinking text leaked into content, no empty or truncated arguments.
    Server errors and timeouts are counted separately from wrong answers
  - output: one markdown table per run appended to the out file: per model,
    backend and think mode, the pass rate per category, the overall pass rate,
    median latency and the error count; plus a short list of failed case ids
  - a model that is missing or fails to load gets a row saying so; the run continues
Acceptance:
  - tests/tools/test_toolcall_eval.py with a mocked transport covering both
    backends: a correct call passes; wrong tool, wrong argument, JSON-in-text,
    a call on a no-tool case and an empty-arguments call each fail for the
    right reason; a two-step chain feeds the canned result back; a server
    error is counted as an error, not a wrong answer
  - the cases file loads and every case validates against the tool schemas
  - `python scripts/toolcall_eval.py --help` exits 0 with no server running
Run: pytest tests/tools/test_toolcall_eval.py -q --no-cov; ruff check scripts/toolcall_eval.py tests/tools/test_toolcall_eval.py; black --check scripts/toolcall_eval.py tests/tools/test_toolcall_eval.py
Size: under about 450 changed lines of Python; the cases file does not count
Out of scope: changing anything under core/, agents/ or tools/; WITS's own
  tool registry; pulling models
Report: files changed, test tail, open questions. 20 lines at most.
```

The owner (or the lead, on the owner's PC) then runs it against
`qwen3:8b`, `qwen3.5:9b`, `gemma4:e4b`, `gemma4:26b` and `qwen3.5:35b-a3b` on
Ollama, plus `qwen3.5:4b` if the owner agrees to pull it (3.4 GB).

## 0.5: Lineup decision (lead)

After 0.4 and 0.6. From the speed table and the tool-calling table, name one
model each for fast, tool-calling, heavy and embedding, say which server runs
the heavy model, and record it in
[`../unified-plan-2026-11.md`](../unified-plan-2026-11.md) under "Still open".
The owner confirms.
