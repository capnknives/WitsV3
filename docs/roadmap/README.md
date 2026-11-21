---
title: "WitsV3 Roadmap Documentation"
created: "2025-06-09"
last_updated: "2026-11-21"
status: "active"
---

# WitsV3 Roadmap Documentation

Revival status, forward plans, and audits.

## Documents

| Document | Purpose |
|----------|---------|
| **[unified-plan-2026-11.md](unified-plan-2026-11.md)** | **Execution order** — stages, gates and decisions; folds the workspace and local-model work into the roadmap |
| **[packages/](packages/README.md)** | Work packages — tiers, rules, review ladder, briefs |
| **[modernization-2026-11.md](modernization-2026-11.md)** | **Forward roadmap** — item list and tracker for “what’s next” (daily affairs, harness in/out) |
| **[suggested-features-2026-07.md](suggested-features-2026-07.md)** | July 2026 roadmap — revival backlog, closed; superseded for forward work |
| **[guest-tester-access-2026-07.md](guest-tester-access-2026-07.md)** | Guest / family-tester access (plan + safe MVP status) |
| **[revival-2026-07.md](revival-2026-07.md)** | Shipped work log + error triage |
| **[clutter-catalog-2026-07.md](clutter-catalog-2026-07.md)** | Dead / dormant / orphan inventory |
| **[tool-registry-reality-2026-07.md](tool-registry-reality-2026-07.md)** | LIVE / BLOCKED / DEAD per tool |
| **[config-surface-truth-2026-07.md](config-surface-truth-2026-07.md)** | Live knobs vs unused YAML |
| **[composer-orchestrator-search-quality-2026-07.md](composer-orchestrator-search-quality-2026-07.md)** | Historical handoff — manual tests A–F |
| **[neural-web-roadmap.md](neural-web-roadmap.md)** | Historical 2025 neural-web design |

## Workflow

Feature branches → **`fix/revive-2026-07`** → merge to **`main`** after human verification.

## Guidelines

- Keep **one** canonical “what’s next” item list (currently `modernization-2026-11.md`); `unified-plan-2026-11.md` sets the order and must not restate item detail  
- Do not duplicate open items into revival logs  
- Bump `last_updated` when status changes  
- Mark superseded docs in the header; keep for handoff value  
