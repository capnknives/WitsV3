---
title: "PyQt6 GUI archive (pruned)"
created: "2025-06-09"
last_updated: "2026-07-08"
status: "archived"
---

# PyQt6 GUI — pruned from working tree

**July 8 2026 (Phase 2a):** The full PyQt6 matrix UI + book-writing GUI (~61 files)
was removed from the working tree. The **Web UI** (`run_web.py`) is the client.

## Recover the snapshot

Full tree is preserved in git at tag **`archive-pre-prune-2026-07`**:

```powershell
# Browse what was there
git ls-tree -r --name-only archive-pre-prune-2026-07 docs/archive/gui/

# Restore locally (optional)
git checkout archive-pre-prune-2026-07 -- docs/archive/gui/
```

Or inspect any commit before the Phase 2a prune (parent of the prune commit).

## What was archived

- Matrix-style PyQt6 chat shell (`main.py`, `matrix_ui.py`, …)
- Standalone book-writing GUI (FastAPI + static UI prototype)
- Replaced by the shipped FastAPI + SSE web UI in `web/`
