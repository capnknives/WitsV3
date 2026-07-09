# Scripts catalog

Utilities under `scripts/` — not imported by the hot path (`run_web.py` / `run.py`).
Run from the repo root with the project venv:

```powershell
.venv\Scripts\python.exe scripts/<name>.py
```

Path layout for runtime data: [`docs/architecture/paths-and-layout.md`](../docs/architecture/paths-and-layout.md).

## Setup & local data

| Script | Purpose |
|--------|---------|
| `setup_local_data.py` | First-run: create `var/` layout, seed templates |
| `setup_mcp_servers.py` | Configure MCP server entries |
| `clone_mcp_servers.py` | Clone optional MCP repos into `mcp_servers/` |
| `debug_init.py` | Step-through Wits init (config, LLM, tools, memory) |

## Smoke & verification

| Script | Purpose |
|--------|---------|
| `conversation_task_smoke.py` | YAML-driven smoke (`--quick` = operator + routing, no Ollama) |
| `smoke_harness.py` | Harness used by conversation_task_smoke |
| `smoke_scenarios.yaml` | Scenario definitions (incl. `op-runtime-layout`) |
| `smoke_ab_compare.py` | A/B comparison helper for smoke runs |
| `guest_smoke_test.py` | Guest auth + route deny checks |
| `tester_live_smoke.py` / `tester_label_smoke.py` | Tester-role live checks |
| `fakecarl_profile_smoke.py` | Family profile smoke |
| `a42ee2e0_live_smoke.py` | Session-specific live regression harness |
| `docker_sandbox_verify.py` | One-shot Docker sandbox preflight |
| `restore_runtime_memory.ps1` | Restore `wits_memory.json` + FAISS from personal `WitsV3` runtime |
| `ollama_probe.py` | Quick Ollama connectivity probe |
| `llm_diagnostic_basic.py` | Minimal LLM round-trip diagnostic |

## MCP & background

| Script | Purpose |
|--------|---------|
| `manage_background_agents.py` | Docker background agent helper |
| `ensure_docker_desktop.ps1` | Windows: ensure Docker Desktop is running (called from launchers) |

## Maintenance & docs

| Script | Purpose |
|--------|---------|
| `fix_neural_web.py` | Repair / reset neural web JSON |
| `analyze_memory.py` | Inspect memory file stats |
| `sync_guest_display_names.py` | Sync guest display names in registry |
| `cleanup_backups.py` / `cleanup_originals.py` | Housekeeping for old backup trees |
| `migrate_docs.py` | Historical doc migration helper |
| `doc_maintenance.py` | Doc metadata / link maintenance |
| `update_readme.py` | README section updater |
| `standardize_format.py` | Format normalization utility |
| `add_metadata.py` | Add front-matter to markdown files |

## Manual tests (ad hoc)

These are developer scratch scripts, not CI tests (see `tests/` for pytest):

- `run_test.py` (repo root) — thin wrapper for quick checks
- Live smoke scripts above when iterating on routing/orchestrator behavior

## Related

- Clutter deletes from [`docs/roadmap/clutter-catalog-2026-07.md`](../docs/roadmap/clutter-catalog-2026-07.md) §1 (dead root scripts) are already removed from the repo.
