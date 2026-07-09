---
title: "WitsV3 File Structure Documentation"
created: "2025-06-10"
last_updated: "2026-07-08"
status: "active"
---

# WitsV3 File Structure

Practical map of the repo as of July 2026. Prefer this + [`README.md`](README.md) over older GUI-centric checklists.

## Packages

| Path | Role |
|------|------|
| `agents/` | Control center, ReAct orchestrator, coding / self-repair / book / background agents |
| `core/` | Config, LLM interfaces, memory backends, schemas, `safe_code_editor.py`, model router, MCP adapters |
| `tools/` | Auto-discovered `BaseTool` implementations (26 built-in names at last count) |
| `web/` | FastAPI app, SSE chat, static UI, settings / MCP pages |
| `tests/` | Pytest suite mirroring packages |
| `config/` | Extra YAML (personality, ethics, background agent) |
| `scripts/` | Setup / maintenance utilities (`setup_local_data.py`, `debug_init.py`, `fix_neural_web.py`, …) |
| `docs/` | Roadmaps, architecture, technical notes, archives (GUI pruned — tag `archive-pre-prune-2026-07`) |
| `planning/` | **Redirect stub only** → see [`docs/README.md`](docs/README.md) |
| `documents/` | User drop folder for Document RAG (contents gitignored) |
| `workspace/` | Coding-agent generated projects (gitignored) |
| `data/` | Local memory / MCP defs (personal data gitignored) |
| `exports/` | Chat exports from the web UI |
| `mcp_servers/` | Optional/on-demand MCP server checkouts |
| `logs/` | Runtime logs (`witsv3.log`) |

## Root entry points

| File | Role |
|------|------|
| `run_web.py` | **Primary** — Web UI + in-process daily self-repair schedule |
| `run.py` | CLI + `--test` self-check + same schedule |
| `run_background_agent.py` | Docker/optional background agent |
| `install.py` | Deps + local data + auth + model pulls |
| `setup_auth.py` | Writes `WITSV3_AUTH_TOKEN_HASH` into `.env` |
| `config.yaml` | Main configuration |
| `.env.example` | Secrets template → copy to `.env` |
| `WORKTREES.md` | Personal vs Cursor vs Claude local folder roles |
| `requirements.txt` / `requirements.lock` | Dependencies |
| `pyproject.toml` | Project / tool config (single source; no separate pytest.ini / mypy.ini) |
| `start_web_ui.bat` | Windows launcher using project `.venv` |
| `Dockerfile` / `Dockerfile.background` / `docker-compose.background.yml` | Parked / background deploy path |

## Documentation entry points

| File | Role |
|------|------|
| `README.md` | Install, run, capabilities |
| `AGENTS.md` | Agent conventions |
| `TASK.md` / `PLANNING.md` | Redirects → roadmap |
| `PATH_MIGRATION_GUIDE.md` | Historical path moves (2025 reorganization) |
| `DOCKER_INSTRUCTIONS.md` | Parked Docker notes |
| `docs/roadmap/suggested-features-2026-07.md` | What's next |

## Conventions

- Prefer async I/O; tools under `tools/` extend `BaseTool`
- Keep modules under ~500 lines
- Secrets never belong in `config.yaml`
- Code-writing tools must stay inside the project tree via `resolve_within_project()`
