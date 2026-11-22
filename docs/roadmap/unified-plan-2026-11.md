---
title: "WitsV3 Unified Execution Plan"
created: "2026-11-21"
last_updated: "2026-11-21"
status: "active"
---

# WitsV3 Unified Execution Plan

**Created:** November 21, 2026
**Status:** Stage 0 in progress (0.1 to 0.4 done; 0.6 built and run on the small models; long runs on hold until the PC is cleaned)
**Reviewed against:** `main` @ `86a0f80` (November 17, 2026)

This plan sets the **order of work**. It folds three plans into one build:

1. The WITS modernization roadmap ([`modernization-2026-11.md`](modernization-2026-11.md)).
2. A cross-device, agent-agnostic workspace (shared tasks, handoffs, a phone
   Decision Inbox, a PC compute node).
3. Local model optimization on the owner's 8 GB GPU.

[`modernization-2026-11.md`](modernization-2026-11.md) stays the **item list and
tracker**: its numbered items and checkboxes are still where status is recorded.
This file says which items ship in which stage, adds the items the other two
plans need, and defines the gates. Package briefs live in
[`packages/`](packages/README.md).

## Why one plan

The workspace is the WITS roadmap seen from the phone. Its pieces already have
homes there:

| Workspace piece | Roadmap item |
|-----------------|--------------|
| One MCP endpoint with coordination tools | 2.1 MCP server (tools join the 2.2 set) |
| Decision Inbox with one-tap approval cards | 1.8 approval queue + 3.3 phone notifications |
| Task lock manager | 1.4 SQLite |
| Append-only handoff log | **New** — a table under 1.4 |
| PC compute node (Whisper, benchmarks, local models) | 4.2 harness runner + 5.1 local Whisper |
| Agent-to-agent task router | 4.7 A2A endpoint |
| Private mesh | "Keep remote access on Tailscale" |
| Any agent can use the models | 1.5 providers, 2.4 OpenAI-compatible API |

Local optimization is the model layer under both: a benchmarked model lineup,
a heavy class in the model router, and one-GPU-job-at-a-time scheduling.

## Decisions made (owner, November 21, 2026)

- The workspace runs **on the PC, behind Tailscale**. Hosted agents that cannot
  join the tailnet wait for the OAuth tunnel (item 5.4).
- A **thin workspace slice ships before the foundations work** (Stage 1 before
  Stage 2).
- The slice is piloted on one active project of the owner's choosing.
- No GPU purchase. RAM is revisited only if Stage 0 shows the heavy model is
  RAM-bound.

## Host facts the plan rests on

Measured on the machine, November 21, 2026. Benchmarks are in
[`docs/benchmarks/results.md`](../benchmarks/results.md).

- GPU: RTX 3070, 8 GB. About 7 GB is usable with a normal desktop session open.
- CPU: i5-12400F (six cores, no integrated graphics, so the display stays on the GPU).
- Board: Gigabyte B660 DS3H AC DDR4. Slot `DDR4_A1` is damaged; A2, B1 and B2 work.
- RAM: 32 GB DDR4 at 2400, XMP off. One Team 16 GB DDR4-2666 stick in A2, two
  Kingston 8 GB DDR4-3200 sticks in B1 and B2, so 16 GB per channel. Until the
  afternoon of November 21 it was 8 GB on channel A and 24 GB on channel B.
- Ollama **0.40.17** (not 0.32.3, as first written here) with `q8_0` KV cache
  and flash attention. It runs the large models through `llama-server`.
  Models are stored on `D:\OllamaModels`. A separate llama.cpp, Tailscale and a
  Linux WSL distro are not installed.
- The owner keeps project and scratch files on `D:`, not `C:`.
- `config.yaml` uses `qwen3:8b` for every role, `model_routing.enabled: false`,
  and names `qwen2.5-coder:7b` and `hermes3:8b`, which are not pulled. Only the
  book agent sets `num_ctx`.

### Stage 0 findings so far

Warm runs, tokens per second, Ollama:

| Model | 8k context | 32k context | Note |
|-------|------------|-------------|------|
| `gemma4:e4b` | 208 | 208 | Fully on the GPU |
| `qwen3:8b` | 71 | 9 | Spills to the CPU at 32k; keep it at 8k |
| `qwen3.5:9b` | 42 | 17 | |
| `gemma4:12b` | 27 | 21 | Slower than `gemma4:26b`; a candidate to drop |
| `gemma4:26b` | 37 to 38 | not re-run | 3.8 GB of weights on the GPU, 13.2 GB in RAM |
| `qwen3.5:35b-a3b` | 36 | not re-run | About three quarters in RAM |

- **Balancing the RAM channels was worth about a quarter.** The two large models
  ran at 30 and 28 before the sticks were moved. The gain held on a second run
  30 minutes after boot.
- **`ollama ps` misreports the split for `gemma4:26b`** ("82% GPU"). The server
  log is the reliable source.
- **Both large models are limited by memory bandwidth**, so they respond to RAM
  speed and to anything that moves expert reads onto the GPU.
- **The benchmark measures speed only.** Nothing here says which model calls
  tools reliably. Package 0.6 measures that.

Two research passes (our own, and a Gemini deep-research report the owner ran)
agree on most points. The one that changes this plan: llama.cpp gained a GPU
cache for mixture-of-experts weights in November 2026 (`--moe-cache-mib` with
`--moe-cache-cpu-share`), Ollama 0.40.17 does not expose it, and a published
test on an RTX 3070 8 GB took a 35B model of the same family from about 35 to
about 49 tokens per second. That is one tester with unknown RAM speed, so
package 0.4 measures it here before anything is built on it.

## Stages

Each stage ends at a gate the owner tests by hand. Sizes are rough, part-time.

### Stage 0 — Measure the box (one weekend)

- [x] 0.1 Benchmark script (tokens/s, time to first token, VRAM, RAM, two context sizes)
- [x] 0.2 Compute-mode script (reports free VRAM and RAM, lists what to close)
- [x] 0.3 Benchmark run across the installed chat models
- [x] 0.4 llama.cpp MoE-cache test on the heavy model: **passed**, 45 to 47 tok/s against Ollama's 36 (about 25 to 30% faster) with a 2.5 to 3 GB cache
- [ ] 0.4b Benchmark backend for `llama-server` and a cache sweep script
- [ ] 0.6 Tool-calling test across the candidate models, on Ollama and on `llama-server`
- [ ] 0.5 Lineup decision, after 0.4 and 0.6: fast, tool-calling, heavy, embedding; and which server runs the heavy model

**Gate:** a results table in `docs/`, and one named model per role.

### Stage 1 — Workspace slice (1–2 weeks)

Pulls forward parts of 1.4, 1.8, 2.1 and 3.3.

- [ ] 1.0 Contracts: SQLite schema, five coordination tool signatures, token scopes
- [ ] 1.1 SQLite store for tasks, locks, approvals, handoffs, clients (1.4, part)
- [ ] 1.2 Per-client tokens with scopes and revocation (1.8, part)
- [ ] 1.3 Approval queue on the store, reusing the `core/escalation.py` pattern (1.8, part)
- [ ] 1.4 MCP server over HTTP at `/mcp` with `claim_task`, `append_handoff`,
      `request_decision`, `link_artifact`, `get_project_state` (2.1)
- [ ] 1.5 Decision Inbox page in the PWA
- [ ] 1.6 Phone push on a new approval, via ntfy or Web Push (3.3)
- [ ] 1.7 Setup guide: Tailscale on PC, phone and laptop; registering the MCP server with an agent
- [ ] 1.8 Stage gate review and pilot wiring

**Gate:** an outside agent claims a task and posts a decision; the owner
approves it from the phone away from home Wi-Fi; revoking the token cuts access.

### Stage 2 — Foundations (2–3 weeks)

The rest of Phase 1, plus the model lineup from Stage 0.

- [ ] 2.1 Remove unused dependencies (1.1)
- [ ] 2.2 Move research code into `research/` (1.2)
- [ ] 2.3 Single `wits` command (1.3)
- [ ] 2.4 Provider interface with OpenAI-compatible and Anthropic providers (1.5)
- [ ] 2.5 Model router heavy class, lineup, explicit `num_ctx` per role, one model on the GPU at a time with the heavy model swapped in deliberately (**new**)
- [ ] 2.6 Native tool calling as default, `json_react` as fallback (1.6)
- [ ] 2.7 Official MCP SDK client (1.7)
- [ ] 2.8 Sessions and memory records into SQLite (rest of 1.4)
- [ ] 2.9 Data tiers (1.9)
- [ ] 2.10 Stage gate review

**Gate:** full suite green on native tool calls; the log shows a greeting going
to the fast model and a multi-step request going to the heavy one.

### Stage 3 — Harness in, completed (1 week)

The rest of Phase 2: memory and document tools, resources and prompts,
`/v1/chat/completions`, the Connected agents page, the injection guard on
inbound text.

**Gate:** the Phase 2 gate, unchanged.

### Stage 4 — Daily affairs (3–4 weeks)

Phase 3 as written, minus 3.3 (shipped in Stage 1).

**Gate:** the Phase 3 gate, unchanged.

### Stage 5 — Harness out and the compute node (2–3 weeks)

Phase 4, plus:

- [ ] GPU lease in the job runner: one GPU job at a time across the LLM, Whisper and benchmark sweeps (**new**)
- [ ] Batch worker: transcribe and score audio dropped into a watched folder (**new**). Speech-to-text and embeddings run on the CPU so they never evict the loaded model; both research passes recommend this

**Gate:** an audio clip dropped from the phone comes back as a scored
transcript, and a delegated coding job ends in a tested diff approved from the
phone.

### Stage 6 — Reach (ongoing)

Phase 5 as written. Item 5.4 (OAuth tunnel) is what lets hosted agents into the
workspace.

## Still open (owner)

- **Thermals.** The PC shut itself off once on November 21 (hard power loss, no crash dump) during a long GPU run with the case closed and the coolers clogged with dust. The same load ran for 32 minutes afterwards with the side panel off and temperatures logged, peaking at 153 F on the CPU and 156 F on the GPU. Cause not established. Until the case is cleaned: no long unattended runs, and heavy runs use the temperature guard in `D:\DevData\wits-scratch`.
- **Model backend.** 0.4 passed, so the working assumption is now `llama-server` for the heavy model. Likely end state: Ollama for the small models and
  `llama-server` for the heavy one, reached through the OpenAI-compatible
  provider in 2.4. Decided by the result of 0.4. If the cache does not deliver
  here, stay on Ollama alone.
- **Provisional lineup, not final:** `gemma4:26b` heavy; `qwen3:8b` at 8k for
  tool calls until 0.6 reports; `nomic-embed-text` for embeddings.
- **RAM, parked by the owner on November 21.** No BIOS change and no purchase
  for now. Options on file: a manual DDR4-2666 setting (estimated 6 to 10% on
  the large models; needs a backup and MemTest86 first) and a matched 2x16 GB
  DDR4-3200 kit (estimated 12 to 33%, about $200). A single extra stick is not
  recommended. The owner is asked again about 2666 on November 22.
- The four open decisions in [`modernization-2026-11.md`](modernization-2026-11.md) still stand.

## Left out on purpose

- A new GPU.
- Fine-tuning and cloud GPU bursts (research tooling, outside WITS scope).
- The July orchestration blueprint (planner DAG, execution rails, loop circuit
  breakers). Loop breakers are worth another look in Stage 5.
- Multi-provider breadth and enterprise SSO.
