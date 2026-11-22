---
title: "WitsV3 Work Packages"
created: "2026-11-21"
last_updated: "2026-11-21"
status: "active"
---

# Work packages

How the [unified execution plan](../unified-plan-2026-11.md) gets built: one
brief, one fresh agent, one small pull request per package. The cheapest tier
that can pass the acceptance tests builds it. The lead reviews diffs, not
transcripts.

## Who does what

| Tier | Used for | Kept away from |
|------|----------|----------------|
| Lead: Fable 5.2, medium effort | Contracts, briefs, diff review, re-planning after each gate | Writing bulk code |
| Opus, medium | Packages that cross files or touch auth: client tokens, the MCP server, the approval queue, providers | Mechanical edits |
| Sonnet, medium | Well-specified builds: tables, tool wrappers, pages, tests, config | Open design questions |
| Haiku, medium | Mechanical work: dependency removal, setup docs, index updates | Anything needing judgment |
| Reviewer: Fable 5.2, high effort, fresh context | Packages marked **high** and every stage gate. Sees only the brief and the diff | Routine review |

## Rules

1. **One package, one agent, one PR.** Target under 400 changed lines and six
   files. A fresh context per package.
2. **Contracts first.** Package 1.0 writes the SQLite schema and MCP tool
   signatures once. Later briefs link to that file.
3. **Briefs name the files.** Each brief lists what to read and what to skip.
   Never read `docs/implementation/`, `docs/archive/` or
   `docs/roadmap/revival-2026-07.md` unless a brief says so (about 210 KB that
   no package needs).
4. **New code goes in new files.** `web/server.py` and `web/static/app.js` are
   already past the 500-line rule. Add modules; touch the big files only to
   register them.
5. **Tests are the first reviewer.** Report done only with targeted tests,
   `ruff` and `black --check` passing. The report is files changed, the test
   tail and open questions, in 20 lines or fewer.
6. **Two strikes, then stop.** After two failed acceptance runs, report back.
   The lead re-briefs or moves the package up one tier.
7. **Parallel only across files.** Packages that share a file run in sequence.
8. **Plan one stage ahead.** Later stages get packages after the gate before them.
9. **Pull requests only.** Branch `claude/wp-<id>-<slug>` from `origin/main`,
   PR to `main`. The owner merges. Never edit the local runtime or worktree
   folders listed in `WORKTREES.md`.

## Review ladder

1. Agent self-check: targeted tests and lint.
2. CI on the PR: pytest on Python 3.10 and 3.13, ruff, black.
3. Lead diff review against the brief's acceptance list.
4. Independent high-effort review for packages marked **high** and every stage gate.
5. Owner hand-test at the stage gate. Owner merges.

## Package index

### Stage 0 — Measure the box ([briefs](stage-0.md))

| ID | Package | Builder | Needs | Done when |
|----|---------|---------|-------|-----------|
| 0.1 | Benchmark script | Sonnet | — | Runs against `qwen3:8b` and writes a table |
| 0.2 | Compute-mode script | Haiku | — | Prints free VRAM and RAM on the host |
| 0.3 | Benchmark run across installed chat models | Owner, one command | 0.1, 0.2 | Table committed |
| 0.4 | llama.cpp MoE-cache test | Lead, on the owner's PC; owner approves the download | 0.3 | Rows in the table; pass or fail against Ollama |
| 0.4b | Benchmark backend for `llama-server` (only if 0.4 passes) | Sonnet | 0.4 | Mocked test passes |
| 0.6 | Tool-calling test | Sonnet | 0.1 | Pass-rate table for the candidate models |
| 0.5 | Lineup decision | Lead | 0.4, 0.6 | Written into the unified plan |

### Stage 1 — Workspace slice

Brief for 1.0: [wp-1.0-contracts.md](wp-1.0-contracts.md). The contract split 1.1 into 1.1a and
1.1b; their briefs are in [stage-1-store.md](stage-1-store.md). Where the index says 1.1,
read both. Briefs for 1.2 to 1.8 follow in plan order.

| ID | Package | Builder | Needs | Done when |
|----|---------|---------|-------|-----------|
| 1.0 | Contracts: schema, tool signatures, token scopes | Lead. Review: **high** | — | Merged before any build starts |
| 1.1a | SQLite store with migrations and async access ([brief](stage-1-store.md)) | Sonnet | 1.0 | Tests for every table, plus a two-writer test |
| 1.1b | Gateway shell: config block, route registry, owner check ([brief](stage-1-store.md)) | Sonnet | 1.1a | Disabled by default; owner-only tests pass |
| 1.2 | Client tokens: create, scope, revoke, hashed at rest ([brief](stage-1-auth.md)) | Opus. Review: **high** | 1.1 | A revoked token is refused; scope tests pass |
| 1.3 | Approval queue, reusing the `core/escalation.py` pattern ([brief](stage-1-auth.md)) | Opus | 1.1 | Request, approve, deny, expire tests |
| 1.4a | Workspace tools: validation, cleaning, permissions, audit; inbound text through `core/injection_guard.py` ([brief](stage-1-gateway.md)) | Opus. Review: **high** | 1.2, 1.3 | Tool tests; refused calls write nothing |
| 1.4b | MCP server over HTTP at `/mcp` with the five tools ([brief](stage-1-gateway.md)) | Opus. Review: **high** | 1.4a | Tool-call tests; unauthenticated call refused |
| 1.5 | Decision Inbox: one static page, two API routes ([brief](stage-1-gateway.md)) | Sonnet | 1.3 | A card appears; approve and deny work |
| 1.6 | Phone push on a new approval ([brief](stage-1-gateway.md)) | Sonnet | 1.3 | Push fires in a test against a fake endpoint |
| 1.7 | Setup guide: Tailscale, registering the MCP server | Haiku | 1.4 | The owner can follow it cold |
| 1.8 | Stage gate review and pilot wiring | Lead. Review: **high** | All | Stage 1 gate passes |

After 1.1, packages 1.2 and 1.3 run side by side. After 1.3, packages 1.5 and
1.6 run beside 1.4a; 1.4b follows 1.4a. `core/injection_guard.py` is 37 lines today; the review of
1.4 decides whether that is enough for text arriving from other agents.

### Stage 2 — Foundations

Briefs are written after the Stage 1 gate.

| ID | Package | Roadmap item | Builder |
|----|---------|--------------|---------|
| 2.1 | Remove unused dependencies | 1.1 | Haiku |
| 2.2 | Move research code into `research/` | 1.2 | Sonnet |
| 2.3 | Single `wits` command | 1.3 | Sonnet |
| 2.4 | Provider interface; OpenAI-compatible and Anthropic providers | 1.5 | Opus |
| 2.5 | Model router heavy class, lineup, `num_ctx` per role | New | Sonnet |
| 2.6 | Native tool calling as default | 1.6 | Opus |
| 2.7 | Official MCP SDK client | 1.7 | Opus |
| 2.8 | Sessions and memory records into SQLite | Rest of 1.4 | Sonnet |
| 2.9 | Data tiers | 1.9 | Opus. Review: **high** |
| 2.10 | Stage gate review | — | Reviewer |

## Brief template

```
WP <id>: <title>
Tier: <model>, medium
Branch: claude/wp-<id>-<slug>, from origin/main
Read: <files, with line ranges where they are long>
Do not read: docs/implementation/, docs/archive/, docs/roadmap/revival-2026-07.md
Contract: <contracts file, section>
Build: <3 to 6 bullets>
Acceptance: <the tests that must pass>
Run: pytest <paths> -q --no-cov; ruff check <paths>; black --check <paths>
Out of scope: <list>
Report: files changed, test tail, open questions. 20 lines at most.
```
