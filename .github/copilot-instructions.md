# Copilot Cloud Agent Instructions for WitsV3

## What this repository is
WitsV3 is a local-first LLM orchestration system with a FastAPI Web UI (`run_web.py`) and CLI (`run.py`) on top of Ollama. The main routing path is:
- `agents/wits_control_center_agent.py` (entry/router)
- `agents/llm_driven_orchestrator.py` + `agents/base_orchestrator_agent.py` (ReAct + tool use)
- specialist agents (coding, self-repair, book-writing)

## Read this first (in order)
1. `/home/runner/work/WitsV3/WitsV3/README.md` (current product behavior)
2. `/home/runner/work/WitsV3/WitsV3/planning/roadmap/suggested-features-2026-07.md` (active priorities)
3. `/home/runner/work/WitsV3/WitsV3/AGENTS.md` and `/home/runner/work/WitsV3/WitsV3/CLAUDE.md` (agent architecture + working conventions)

Do **not** treat `planning/architecture/*` as source-of-truth for current behavior; it may lag.

## Setup and validation commands
From repo root:

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt
python scripts/setup_local_data.py

# quick sanity checks
make lint
pytest tests/ -q --no-cov

# CI-equivalent lint scope
ruff check agents core tools web tests
black --check agents core tools web tests
```

## High-value architecture constraints
- Keep agent/tool paths async; stream via `StreamData` helpers.
- Reuse `core/safe_code_editor.py` for code-writing behavior (verified-edit pipeline) instead of adding ad-hoc write paths.
- Prefer `core/model_router.py` for model selection; do not build on archived adaptive-LLM docs.
- Keep config in `config.yaml`/`WitsV3Config`; secrets only in `.env`.

## Editing conventions
- Python style: black, ruff, type hints, line length 100.
- Keep files modular (~500 lines target).
- Avoid circular imports between `agents/`, `core/`, `tools/`, `web/`.
- Add/adjust tests under `tests/` mirroring source layout when behavior changes.
- Update forward planning in `planning/roadmap/suggested-features-2026-07.md`; root `TASK.md`/`PLANNING.md` are redirects.

## Known pitfalls and workarounds (encountered during onboarding)
1. `make lint` failed initially with `No module named ruff`.
   - Workaround: install deps (`python -m pip install -r requirements.txt -r requirements-dev.txt`), then rerun.
2. `pytest tests/ -q --no-cov` failed initially because pytest loaded `pyproject.toml` addopts requiring coverage flags/plugins.
   - Workaround: ensure full dependencies are installed (includes `pytest-cov`) before running tests.
3. After dependencies were installed, tests ran but did not fully pass in this environment:
   - `tests/web/test_access_log.py::test_resolve_caller_label_static_page`
   - `tests/web/test_web_server.py::test_upload_strips_path_traversal`
   Treat these as pre-existing baseline failures unless your change touches those areas.
