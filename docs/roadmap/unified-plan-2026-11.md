---
title: "WitsV3 Unified Execution Plan"
created: "2026-11-21"
last_updated: "2026-11-22"
status: "active"
---

# WitsV3 Unified Execution Plan

**Created:** November 21, 2026
**Status:** Stage 1 gate passed on November 22, 2026. Stage 2 conversation core: C1 to C4 and folder access F1 to F5 are merged (through PR #80). The new loop is behind `conversation.front_door` and the old path is still the default. The red-team pass is done for targets 1 to 7 (PRs #83 to #91, see [`red-team-2026-11.md`](red-team-2026-11.md)); target 8, the owner's held-out messages, is open. Stage 0 still has open items (0.1 to 0.4 done; 0.6 built and run on the small models; long runs on hold until the PC is cleaned)
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

Added November 22, 2026:

- Stage 2 rebuilds the **conversation core** as one tool-calling loop, replacing
  the intent call and keyword routing. Design and evidence:
  [`conversation-core-2026-11.md`](conversation-core-2026-11.md).
- **Free cloud models are allowed as optional providers** for turns that carry
  nothing private. Private WITS data stays on local models.

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

- [x] 1.0 Contracts: SQLite schema, five coordination tool signatures, token scopes
- [x] 1.1 SQLite store for tasks, locks, approvals, handoffs, clients (1.4, part)
- [x] 1.2 Per-client tokens with scopes and revocation (1.8, part)
- [x] 1.3 Approval queue on the store, reusing the `core/escalation.py` pattern (1.8, part)
- [x] 1.4 MCP server over HTTP at `/mcp` with `claim_task`, `append_handoff`,
      `request_decision`, `link_artifact`, `get_project_state` (2.1)
- [x] 1.5 Decision Inbox page in the PWA
- [x] 1.6 Phone push on a new approval, via ntfy or Web Push (3.3)
- [ ] 1.7 Setup guide: Tailscale on PC, phone and laptop; registering the MCP server with an agent (written; PR #34 open)
- [ ] 1.8 Stage gate review and pilot wiring (gate review and hand-test done; pilot wiring still to do)

**Gate:** an outside agent claims a task and posts a decision; the owner
approves it from the phone away from home Wi-Fi; revoking the token cuts access.

**Gate result (November 22, 2026): passed.** Run on the owner's PC against `main` at `86c04d4`.

- An independent review built the real app, walked the whole scenario with the official MCP
  client, and passed it with six fixes, which were merged before the hand-test (PR #35).
- A test agent was created through the owner route, connected to `/mcp` with its own token,
  claimed a task, left a handoff, linked a pull request and asked for a decision.
- Two refusals were checked on the way: a misleading link (`invalid_argument`) and a project
  outside the agent's list (`forbidden_project`).
- The owner approved the decision in the Inbox from his phone, on mobile data over the private
  network. The agent then read the approval back.
- The agent was revoked. Its next call got 401, no lock or pending decision was left, and the
  audit log held a row for each step.

Found during the hand-test and tracked separately:

- Nothing on the main page links to the Inbox.
- Chat routing sent a plain statement to the coding agent, and intent analysis was being saved
  to memory (PR #38). The slash menu was empty right after sign-in (PR #39).
- The configured code and trivial models are not installed, so those calls fall back. This waits
  on package 0.5 (final model lineup).
- The first reply of a session took about 45 seconds for intent analysis alone.
- The pilot wiring half of 1.8 (connecting Aquainspection's agents) is not done yet.

### Stage 2 — Foundations (2–3 weeks)

The rest of Phase 1, the model lineup from Stage 0, and the conversation core
(revised November 22, see
[`conversation-core-2026-11.md`](conversation-core-2026-11.md)).

**Conversation core, in this order.** Each is one or more small packages; C1
ships before anything changes behavior, and the old path stays selectable until
the gate.

- [x] C1 Turn metrics and a golden test set (C1a to C1d shipped; 61 cases). Per-turn timing (first token,
  prompt time, model, tool calls) and a labelled set of messages with expected
  tool use, including the "software" sentence as a must-not-route case.
  Assertions in code, no model as judge. Records the baseline for the old path.
- [ ] C2 Serving settings and the resident model (C2a serving settings, C2b
  stable prefix and C2c purge script shipped; still open: C2m, the resident
  model by measurement, on hold with the heavy PC runs). `/api/chat` with a message
  list, thinking off by default, model kept loaded, fixed `num_ctx`, embeddings
  on the CPU, stable prompt prefix. Stop storing intent output and remove the
  stored rows. The resident model is picked by measurement on the PC (absorbs
  2.5).
- [x] C3 The native loop as the front door (C3a to C3f shipped; deleting the
  old path still waits for the gate). Finish and test
  `NativeToolOrchestrator`, put it behind a setting at the single
  `control_center.run` seam, add schema checks, one retry, repeat and step
  caps. Old intent call, keyword lists and JSON-repair loop are deleted only
  after the gate (absorbs 2.6).
- [x] C4 Short tool list and tool lookup (C4a tool context, C4b tiers: 16
  always-on tools with full schemas, 12 found through a lookup tool, 4 refused
  until the approval cards in C7 exist). Not done under C4: the design's
  "writes pass a gate". Four tools are refused; every other write is
  unchecked beyond the path rules. See the red-team file.
- [ ] C5 Specialists as skills plus tools. Coding, self-repair and writing
  instructions become skill files loaded on demand. The verified-edit pipeline
  stays as code and becomes a tool. One specialist at a time, golden set run
  three times per fold.
- [ ] C6 Escalation. Thinking on for the same model first; then the heavy
  provider on an explicit command or two failed tool rounds, given a brief and
  file paths. Heavy provider is local or a free cloud model (needs 2.4).
- [ ] C7 Streaming and control in the UI. Token streaming, one collapsed
  activity line, a stop button, write-gate cards from the approval queue. Built
  with the UI overhaul packages.
- [ ] C8 PC test: cache save and restore on `llama-server` across a model swap.
  Waits for the hold on PC runs to lift.

**Folder access** ([contract](../architecture/folder-access.md)).

- [x] F1 Safe defaults and the base folders
- [x] F2 One path check for the file tools, the indexer and memory export
- [x] F3 Folder requests and owner-approved, expiring grants
- [x] F4 Inbox page: requests in plain words, live grants, revoke, short guide
- [x] F2b Export, upload, delete and memory import/export behind the path
  check; one anchor for `folders.base`
- [x] F5 A `documents` folder inside the base folder for uploads (owner
  decision November 22); re-indexing now replaces a changed file's chunks

**Foundations, independent of the above.**

- [ ] 2.1 Remove unused dependencies (1.1)
- [ ] 2.2 Move research code into `research/` (1.2)
- [ ] 2.3 Single `wits` command (1.3)
- [ ] 2.11 Self-update and restart (**new**, owner request November 22). The
  runtime folder follows `main` without a manual pull. `wits update` (part of
  2.3) and a scheduled check: fetch, fast-forward only, never with local
  changes in tracked files it would touch, install changed requirements,
  restart through the existing restart path, then a health check on `/api`
  and a one-line push to the phone. On a failed health check it returns to
  the previous commit and restarts again. Idle only: never during a turn or a
  GPU job. Off by default; the owner turns it on in `config.local.yaml`.
  Briefed after 2.3
- [ ] 2.4 Provider interface with OpenAI-compatible and Anthropic providers (1.5). The OpenAI-compatible provider also covers `llama-server` and the free cloud tiers; a provider is marked private-safe or not
- [ ] 2.7 Official MCP SDK client (1.7)
- [ ] 2.8 Sessions and memory records into SQLite (rest of 1.4). Memory writes follow the rules in the conversation core file: stated facts with their source, nothing from an untrusted page without the owner's say-so
- [ ] 2.9 Data tiers (1.9)
- [ ] 2.10 Stage gate review

**Gate:** the red-team targets in [`red-team-2026-11.md`](red-team-2026-11.md)
each have a test and none is open; a held-out set of messages the lead did
not write is run once and reported; full suite and the golden set green on the new loop; the "software"
sentence gets a direct reply with no tool call; a simple turn shows its first
token in under 2 seconds on the PC; a multi-step request reaches the heavy
provider by rule and the log shows it.

**Measured on November 22, 2026 (old path, `qwen3:8b`, turn records).**

- Until PR #51, WITS sent a context size only for the book agent. Ollama then
  loaded `qwen3:8b` at the model's own default of 40,960 tokens, and 29% of it
  ran on the CPU. Every chat turn paid for that. With `num_ctx` fixed at 8,192
  the model is fully on the GPU and embeddings run on the CPU.
- The owner "software" sentence: 42.8 s (cold) and 25.6 s (warm) before; 7.8 s
  (cold, 5.7 s of it model load) and 1.2 s (warm) after. A greeting: 11.1 s
  before, 0.5 to 1.1 s after. PR #51 also turned thinking off on this path, so
  the two changes are not separated in these numbers.
- Nine plain chat cases from the golden set passed on the old path after the
  fix, 0.5 to 4.3 s each. The other cases have not been run: golden runs wait
  for the sandbox in package C1c.
- First full golden runs, sandboxed: 20 of 30 on the old path and 25 of 30 on
  the new loop, with tool turns at about one second against 5 to 35. Table in
  [`docs/benchmarks/golden.md`](../benchmarks/golden.md).
- A high-effort independent review is used for C3a and the stage gate only.
  C3b and C3c get an independent review at medium.
- Later the same day, new loop with tool tiers (C4b branch, before merge): 30
  of 32 cases passed all three runs. The two that failed all three are
  `tool-document-search` and `followup-yes-do-it`.
- The 29 owner cases (the owner's July messages) were run once on the new
  loop: 28 ran and 19 passed. One run is a first reading, not a pass rate.
- Not measured yet: all 61 cases three times on `main`; first-token time
  (nothing streams until C7); any model other than `qwen3:8b`; guests on the
  new loop; the heavy provider.

**Where things stand at the end of November 22, 2026.**

Done from the earlier follow-up list: C1e and C1f (runner records failures;
sandbox covers `folders.base`), the 61-case runs, F4, F2b, F5, the loop's
system prompt (C4c), the tool-description trim (C4d) and the checker and case
fixes (C1g). Latest table: 56 of 61 cases pass all three runs on `main`. What
that number does and does not show is in the red-team file.

Built, measured once, **not merged**: C5a, long writing as a tool
(`write_long_text`: an outline, then one fresh model call per section, saved
as it goes). Owner decision November 22: long writing is a tool, not a skill
file, because a turn's context cannot hold a long piece written through
repeated tool calls. On the PC, 62 cases three times: the 4,500-word story
case wrote 4,104 to 4,629 words in about 90 seconds and passed all three runs,
and 57 of 62 cases passed every run. It is not ready to merge (draft PR #82):
the tool is also called where the owner wanted an answer in chat ("yes please
do that", "make these instructions more in depth"), one refusal case that
passed before now saves a note instead, and a plain recipe answer went from
about 7 seconds to about 68. The text itself is repetitive and has no scenes
or characters.

**Order of work from here.**

1. Done, except target 8: the red-team pass ([`red-team-2026-11.md`](red-team-2026-11.md)).
   Golden re-run after the fixes: 56 of 61 pass all three runs, unchanged.
2. Done in part: a first write gate and the untrusted-content rule shipped
   with the pass (PR #88). The approval cards that let the owner confirm a
   blocked write stay in C7. Next here: the held-out run (target 8) and a
   real-model probe of the gate.
3. C5a: fix the over-triggering, then merge.
4. Writing quality: named characters and a setting from the outline step,
   distinct events per section, a note of what is already covered, and a
   repeated-phrase check in the golden set.
5. C5 coding skill: aim for structure by default (functions or classes, not
   one flat script), check its own work, name what must be installed, say
   what was and was not tested. Prompt audit with it: the task-specific
   paragraphs of the system prompt move into the skill, and each rule that
   stays gets a recorded reason.
6. C6 escalation by rule, C7 streaming, stop button and approval cards.
7. Memory (2.8), with the outside report's claims checked first. The old
   remember/recall rule is a red-team target.
8. 2.3 single `wits` command, then 2.11 self-update.
9. Stage 1 leftovers and Stage 0 (0.4b, 0.6, 0.5, on hold with the heavy
   runs). 0.6, the tool-calling test on other models, should run before any
   more prompt tuning.

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
