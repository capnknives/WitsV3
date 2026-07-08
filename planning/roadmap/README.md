---
title: "WitsV3 Roadmap Documentation"
created: "2025-06-09"
last_updated: "2026-07-07"
status: "active"
---
# WitsV3 Roadmap Documentation

This directory holds revival status, forward-looking plans, and historical roadmaps.

## Documents (July 2026)

| Document | Purpose |
|----------|---------|
| **[suggested-features-2026-07.md](suggested-features-2026-07.md)** | **Forward roadmap** — what to add, improve, or remove next (start here) |
| **[revival-2026-07.md](revival-2026-07.md)** | **Shipped work log** — July 2026 revival commits, error triage, closed backlog |
| **[composer-orchestrator-search-quality-2026-07.md](composer-orchestrator-search-quality-2026-07.md)** | **Historical handoff** — manual regression tests A–F, Tier 1–4 audit detail |
| **[neural-web-roadmap.md](neural-web-roadmap.md)** | **Historical** — 2025 neural web design (predates revival) |

## Workflow

Feature branches → **`fix/revive-2026-07`** → Richard merges to **`main`** after manual tests.

## Document guidelines

- Keep **one canonical "what's next"** doc (`suggested-features-*.md`); avoid duplicating open items in revival logs.
- Update `last_updated` when changing status.
- Mark superseded docs with a header note rather than deleting (git history + handoff value).
