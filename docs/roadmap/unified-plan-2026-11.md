---
title: "WitsV3 Unified Execution Plan"
created: "2026-11-21"
last_updated: "2026-11-21"
status: "active"
---

# WitsV3 Unified Execution Plan

**Created:** November 21, 2026
**Status:** Stage 0 in progress (0.1 to 0.3 done)
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

Measured November 21, 2026. No benchmark has been run yet.

- GPU: RTX 3070, 8 GB. About 7 GB is usable with a normal desktop session open.
- CPU: i5-12400F (no integrated graphics, so the display stays on the GPU).
- RAM: 32 GB DDR4 in three sticks (8 + 16 + 8) running at 2400 with uneven
  channels. Matters for MoE models whose experts sit in system RAM.
- Ollama 0.32.3 with KV-cache quantization (`q8_0`) and flash attention already
  set. llama.cpp, Tailscale and a Linux WSL distro are not installed.
- `config.yaml` uses `qwen3:8b` for every role, `model_routing.enabled: false`,
  and names `qwen2.5-coder:7b` and `hermes3:8b`, which are not pulled. Only the
  book agent sets `num_ctx`.

## Stages

Each stage ends at a gate the owner tests by hand. Sizes are rough, part-time.

### Stage 0 — Measure the box (one weekend)

- [x] 0.1 Benchmark script (tokens/s, time to first token, VRAM, RAM, two context sizes)
- [x] 0.2 Compute-mode script (reports free VRAM and RAM, lists what to close)
- [x] 0.3 Benchmark run across the installed chat models
- [ ] 0.4 llama.cpp `--n-cpu-moe` sweep on the two large MoE models
- [ ] 0.5 Lineup decision: fast, heavy, embedding; whether `llama-server` earns a place

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
- [ ] 2.5 Model router heavy class, lineup, explicit `num_ctx` per role (**new**)
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
- [ ] Batch worker: transcribe and score audio dropped into a watched folder (**new**)

**Gate:** an audio clip dropped from the phone comes back as a scored
transcript, and a delegated coding job ends in a tested diff approved from the
phone.

### Stage 6 — Reach (ongoing)

Phase 5 as written. Item 5.4 (OAuth tunnel) is what lets hosted agents into the
workspace.

## Still open (owner)

- Model backend: stay on Ollama unless Stage 0 shows hand-tuned `llama-server`
  clearly faster on the large models.
- RAM: decide after Stage 0.
- The four open decisions in [`modernization-2026-11.md`](modernization-2026-11.md) still stand.

## Left out on purpose

- A new GPU.
- Fine-tuning and cloud GPU bursts (research tooling, outside WITS scope).
- The July orchestration blueprint (planner DAG, execution rails, loop circuit
  breakers). Loop breakers are worth another look in Stage 5.
- Multi-provider breadth and enterprise SSO.
