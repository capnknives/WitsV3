# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

WitsV3 is a local-first LLM orchestration system: a Web UI (primary, `run_web.py`) and CLI (`run.py`) backed by Ollama, with a ReAct-style orchestrator, an auto-discovered tool registry, persistent memory, and a verified-edit pipeline that lets agents safely modify project code. It is a personal assistant stack, not a hosted product.

Read `README.md` first for how the product actually runs today, and `planning/roadmap/suggested-features-2026-07.md` for current priorities — architecture docs under `planning/architecture/` lag behind (mid-2025) and should not be trusted for current status.

## Worktree / branch workflow (important — read before committing)

This checkout (`WitsV3-claude`) is the **Claude-only** worktree. Sibling worktrees on this machine: `WitsV3` (personal runtime — never edit code there, it's what actually runs for the user) and `WitsV3-cursor` (Cursor agent). See `WORKTREES.md` for full detail.

- Work on `claude/work` (or `claude/*` feature branches) in this checkout.
- Agent branches merge into the shared integration branch `fix/revive-2026-07`, which is later promoted to `main`.
- Before starting new work, sync `claude/work` against `origin/fix/revive-2026-07` (it accumulates commits from Cursor and other agents) — don't assume `claude/work` alone is current.
- Edits made here are **not** live in the user's runtime (`WitsV3` folder) until merged forward — this matters most for `web/` static assets, which have no build step and are served as-is.
- Root `TASK.md` and `PLANNING.md` are redirects only — do not add task checklists there. Track new work in `planning/roadmap/suggested-features-2026-07.md` and log large shipped chunks in `planning/roadmap/revival-2026-07.md`.

## Commands

```bash
# Setup (first time)
python -m venv .venv
.venv\Scripts\activate                          # Windows
pip install -r requirements.txt
python scripts/setup_local_data.py

# Run
.venv\Scripts\python.exe run_web.py              # Web UI (primary) — http://localhost:8000
python run.py                                    # CLI
python run.py --test                             # Non-interactive self-check (init + LLM + tools + memory + agents)

# Tests
pytest tests/ -q --no-cov                        # Full suite, fast (no coverage)
pytest tests/ -v                                 # Full suite with coverage (matches CI)
pytest tests/agents/test_self_repair_agent.py -v # Single file
pytest tests/agents/test_self_repair_agent.py::TestClass::test_name -v   # Single test
pytest -m "not slow"                             # Skip slow-marked tests

# Lint / format
make lint                                        # ruff check agents core tools web --select E9,F63,F7,F82 (critical-only)
ruff check agents core tools web tests           # Full lint, matches CI
black --check agents core tools web tests        # Matches CI
make format                                      # black + isort, writes changes
```

Markers available: `slow`, `integration`, `unit`, `requires_ollama`, `requires_supabase`. Many features genuinely need a running Ollama instance to exercise end-to-end — unit tests should mock it (`requires_ollama` marks the exception).

CI (`.github/workflows/ci.yml`) runs pytest on 3.10/3.11 plus a separate ruff+black lint job against `agents core tools web tests`; match that scope before pushing.

## Architecture

```
User (Web UI / CLI)
        │
        ▼
WitsControlCenterAgent  ── intent, routing, conversation
        │
        ├──► LLMDrivenOrchestrator  (ReAct + tools + synthesis guard)
        ├──► AdvancedCodingAgent    (scaffolds + verified edits)
        ├──► SelfRepairAgent        (logs / tests → verified fixes)
        └──► BookWritingAgent
                │
                ▼
     Memory Manager · Tool Registry · Ollama · optional Neural Web / MCP
```

Agent hierarchy (`agents/`, all extend `BaseAgent`):

- `WitsControlCenterAgent` — entry point for every user turn; parses intent and routes to specialists **before** falling back to generic paths (the specialists below are load-bearing, not dead code).
- `BaseOrchestratorAgent` → `LLMDrivenOrchestrator` — the ReAct loop (plan → call tools → read observations). Has a **synthesis guard** that rejects final answers ignoring usable search/tool observations (retries once, then auto-synthesizes). Handles JSON robustness for local models (`format=json` + repair-reparse).
- `AdvancedCodingAgent` — new projects scaffold under `workspace/<name>/` with `py_compile` checks; edits to an existing named file go through the same verified-edit pipeline as self-repair.
- `SelfRepairAgent` — targets a named file, or scans `logs/witsv3.log` (then failing tests) when none is named; asks the LLM for a full corrected file; applies via the verified-edit pipeline; optional restart gated by `self_repair.restart_after_fix`.
- `BackgroundAgent` — scheduled maintenance; mainly for the Docker background path. For local use, prefer the in-process daily self-repair schedule that `run.py` / `run_web.py` already register.
- `BookWritingAgent` — long-form content, not part of the filesystem-edit path.

**Verified-edit pipeline** (`core/safe_code_editor.py`) is the safety backbone shared by the coding and self-repair agents: snapshot original bytes → write candidate → run pytest → pass commits via git, fail restores the exact original bytes. Edits are constrained inside the project tree via `resolve_within_project()`. Any new code-writing capability should reuse this pipeline rather than forking its own write path.

**Model routing** (`core/model_router.py`) sizes models by query complexity and is the preferred routing mechanism — a separate adaptive-LLM stack exists under `planning/archive/adaptive_llm/` but is dormant/archived; don't build on it.

Other core packages:

- `core/` — `WitsV3Config` (`config.yaml`-backed), LLM interfaces (`llm_interface.py`, `enhanced_llm_interface.py` for reliability/fallback), memory backends (`basic` / `faiss_cpu` / `faiss_gpu` / `neural`), `schemas.py` (Pydantic models incl. `StreamData`), MCP adapters, `user_errors.py` for friendly LLM-outage messaging.
- `tools/` — auto-discovered `BaseTool` implementations (~26 built-in): files, search, docs, MCP discovery, self-repair, neural-web helpers, etc. New tools must extend `BaseTool` and expose `get_llm_description()` / a schema for LLM tool-calling.
- `web/` — FastAPI app with SSE streaming chat, static UI (no build step — edit `web/static/*` directly), settings/MCP pages.

All agent/tool methods are async generators yielding `StreamData` (`stream_thinking`, `stream_action`, `stream_observation`, `stream_result`) so clients see progressive output. Memory is read/written through the memory manager (`store_memory()` / `search_memory()`) so context survives across turns.

## Conventions

- Python 3.10+, fully async (no synchronous I/O in agent/tool/hot paths), PEP8 + black (line length 100), Pydantic for validation.
- Max ~500 lines per file — split before exceeding.
- Imports: relative within a package, absolute across packages; no circular imports between `agents/` / `core/` / `tools/` / `web/`.
- Config-driven: read from `WitsV3Config` / `config.yaml`, never hardcode; secrets live only in `.env` (gitignored), never in `config.yaml`.
- Comments explain *why*, not *what* (`# Reason: ...` for non-obvious logic); Google-style docstrings.
- Neural web (`memory_manager.backend: neural`) is optional/research — verify tools are actually registered before investing effort there; it is not the default focus of current work.
