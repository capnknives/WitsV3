---
title: "Sphinx documentation stubs (pruned)"
created: "2025-06-09"
last_updated: "2026-07-08"
status: "archived"
---

# Sphinx docs — pruned from working tree

**July 8 2026 (Phase 2b):** Unused Sphinx scaffolding (~11 `.rst` files + `conf.py`)
was removed. Product docs live under `docs/` as Markdown; there is no Sphinx build
in CI or the install path.

## Recover the snapshot

Full tree is preserved in git at tag **`archive-pre-prune-2b-2026-07`**:

```powershell
git ls-tree -r --name-only archive-pre-prune-2b-2026-07 docs/archive/sphinx/

# Restore locally (optional)
git checkout archive-pre-prune-2b-2026-07 -- docs/archive/sphinx/
```

Earlier GUI archive recovery uses tag **`archive-pre-prune-2026-07`**.
