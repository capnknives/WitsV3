---
title: "WitsV3 Documentation"
created: "2025-06-09"
last_updated: "2026-07-08"
status: "active"
---

# WitsV3 Documentation

Design notes, roadmaps, and historical material for WitsV3.  
**Product install and day-to-day usage live in the root [`README.md`](../README.md).**

> **July 2026:** Canonical planning docs moved from `planning/` to `docs/`.  
> The `planning/` folder is a redirect stub only.

## Start here

| Need | Document |
|------|----------|
| **What's next** | [`roadmap/suggested-features-2026-07.md`](roadmap/suggested-features-2026-07.md) |
| **What shipped (July 2026)** | [`roadmap/revival-2026-07.md`](roadmap/revival-2026-07.md) |
| **Architecture** | [`architecture/system-architecture.md`](architecture/system-architecture.md) *(component map)* |
| **Memory model** | [`architecture/memory.md`](architecture/memory.md) |
| **Dead/dormant code** | [`roadmap/clutter-catalog-2026-07.md`](roadmap/clutter-catalog-2026-07.md) |
| **Tool registry truth** | [`roadmap/tool-registry-reality-2026-07.md`](roadmap/tool-registry-reality-2026-07.md) |
| **Config surface truth** | [`roadmap/config-surface-truth-2026-07.md`](roadmap/config-surface-truth-2026-07.md) |

Root redirects: [`TASK.md`](../TASK.md), [`PLANNING.md`](../PLANNING.md), [`planning/README.md`](../planning/README.md).

## Directory structure

| Folder | Contents |
|--------|----------|
| [`architecture/`](architecture/) | System design |
| [`roadmap/`](roadmap/) | Forward roadmap + July revival logs + audits |
| [`implementation/`](implementation/) | Implementation write-ups (many historical) |
| [`tasks/`](tasks/) | Superseded task lists → use roadmap |
| [`technical-notes/`](technical-notes/) | Debug / fix notes (prefer consolidated) |
| [`archive/`](archive/) | Originals, historical docs; dormant code pruned — tags `archive-pre-prune-2026-07` (GUI), `archive-pre-prune-2b-2026-07` (adaptive LLM, sphinx, synthetic brain) |

## Documentation standards

New docs should:

1. Live under the right subdirectory  
2. Use `lowercase-with-hyphens.md` (include `YYYY-MM` when time-sensitive)  
3. Carry YAML front matter (`title`, `created`/`last_updated`, `status`)  
4. Get linked from this README and/or [`roadmap/README.md`](roadmap/README.md)  
5. Prefer updating the canonical roadmap over duplicating “what’s next” elsewhere  

Supersede by marking status + pointing to the replacement; don’t delete history casually.

### Optional tooling

```bash
python scripts/doc_maintenance.py list
python scripts/doc_maintenance.py create docs/roadmap/new-note.md "Title"
python scripts/doc_maintenance.py archive docs/path/obsolete.md
```

Older one-shot migration scripts under `scripts/` (`migrate_docs.py`, etc.) are historical.
