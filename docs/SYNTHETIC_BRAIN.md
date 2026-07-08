---
title: "WITS Synthetic Brain — consolidated status"
status: "historical"
---

# WITS Synthetic Brain

**Historical / superseded.** This consolidates five overlapping docs written
around 2025-06-12 for the "WITS Synthetic Brain Expansion Plan" (Phase 1:
core cognitive layer integration). The initiative stalled after Phase 1 and
was superseded by the July 2026 revival work
(`planning/roadmap/revival-2026-07.md`). It is **not** the current
architecture.

## What actually shipped

- `core/memory_handler.py`, `core/cognitive_architecture.py` — the "base
  versions" mentioned in these docs are the ones still imported by the live
  system today.
- `config/wits_core.yaml` — identity/memory/cognitive config structure.

## What did not ship / was abandoned

- `core/memory_handler_updated.py`, `core/cognitive_architecture_updated.py`
  — the "enhanced" parallel forks these docs describe as the main
  implementation. Never wired into `run.py` or any live code path; removed
  as dead code (see `composer-orchestrator-search-quality-2026-07.md` Tier 3
  §11).
- Sensorimotor I/O, self-model, autonomous goals, emotion modeling, ethical
  reasoning, symbolic planning (Phases 2–6) — never started.

## Original documents (kept for archival detail, not maintained)

- [`README_SYNTHETIC_BRAIN.md`](README_SYNTHETIC_BRAIN.md) — component overview
- [`IMPLEMENTATION_STATUS.md`](IMPLEMENTATION_STATUS.md) — Phase 1 status snapshot, 2025-06-12
- [`IMPLEMENTATION_SUMMARY.md`](IMPLEMENTATION_SUMMARY.md) — commit-by-commit summary
- [`REMAINING_TASKS.md`](REMAINING_TASKS.md) — Phase 1 follow-up task list
- [`SYNTHETIC_BRAIN_NEXT_STEPS.md`](SYNTHETIC_BRAIN_NEXT_STEPS.md) — Phase 1 completion plan
- [`SYNTHETIC_BRAIN_PR.md`](SYNTHETIC_BRAIN_PR.md) — original PR description

If reviving this initiative, start from the current `revival-2026-07.md`
architecture rather than resuming these docs' plan.
