# Handoff — Post deep-verify (July 9, 2026)

**Branch:** `fix/revive-2026-07` in `WitsV3-claude`  
**Plan:** Next 20 Steps (do not edit `.cursor/plans/*.plan.md`)  
**Todo truth:** Only **step 1 complete**. Steps **2–20 pending** (large diff exists uncommitted).

## Step 1 done

- Deep verify log: `var/logs/deep_verify_20260709_124734.log`
- **Green:** pytest 38, Docker sandbox, Docker RBAC 4
- **Never finished:** smoke phase (killed; hung on FAISS/embed init + corrupt memory)

## Blocker (step 2)

- `var/data/wits_memory.json` corrupt (~70MB mid-write); valid copy in `WitsV3/var/data/` (152 segments)
- Restore **JSON + FAISS together** after stopping `run_web.py` / smoke:
  `powershell -File scripts/restore_runtime_memory.ps1`

## Uncommitted work (steps 4–20 draft)

| Area | Key files |
|------|-----------|
| Phase 4 paths | `core/runtime_paths.py`, `config.yaml`, `docs/architecture/paths-and-layout.md`, `op-runtime-layout` smoke |
| FAISS hardening | `core/faiss_memory_backend.py`, `tools/document_tools.py` (`persist=False`+`flush`), `core/memory_backend_factory.py` |
| Fast smoke | `scripts/conversation_task_smoke.py` (`--quick` → auto `--isolated`) |
| CI | `.github/workflows/ci.yml` runtime layout check |
| Docs | `memory.md` runbook, roadmaps, `json_react` default in `config.yaml` |

## Verify before commit

```powershell
powershell -File scripts/restore_runtime_memory.ps1
pytest tests/ -q --no-cov
python scripts/conversation_task_smoke.py --quick --only op-runtime-layout
```

**Known failure:** `test_safe_code_editor` Docker path — read-only `var/logs/witsv3.log` in container (fix or skip).

## Next sequence

2 restore → 3 re-verify → 4 commit → 5–7 pytest + merge worktrees → mark 8–20 done after ship

## Do not redo

- Phase 4 path rename (`var/user_files`)
- FAISS atomic save / stale-index quarantine (already coded)
- Clutter Wave A deletes (orphans already gone)
