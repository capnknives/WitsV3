---
title: "Paths and folder layout"
created: "2026-07-09"
last_updated: "2026-07-09"
status: "active"
---

# Paths and folder layout

WitsV3 uses three buckets. Do not mix them.

## Three buckets

| Bucket | Location | Mutable? | Examples |
|--------|----------|----------|----------|
| **Source** | `agents/`, `core/`, `tools/`, `web/`, `tests/`, `scripts/` | No (git) | Python packages, static UI |
| **Config + docs** | `config.yaml`, `config/`, `docs/`, root `*.md` | Mostly git | Roadmaps, guest policy, ethics |
| **Runtime** | `var/` (default) | Yes (local) | Memory, logs, uploads, exports |

Personal data **never** belongs at the repo root. Legacy top-level `data/`, `logs/`, `documents/`, etc. are merged into `var/` on startup via [`core/runtime_paths.py`](../../core/runtime_paths.py).

## Glossary (easy to confuse)

| Path | What it is | What it is NOT |
|------|------------|----------------|
| **`docs/`** | Repository documentation (architecture, roadmaps) | Your PDFs / uploads |
| **`var/user_files/`** | Your RAG corpus (drop folder, uploads) | Repo documentation |
| **`var/data/`** | Memory JSON, MCP defs, guest profiles | Application source |
| **`var/logs/`** | `witsv3.log`, MCP logs | Git history |
| **`var/exports/`** | Saved chat transcripts | `docs/` |
| **`var/workspace/`** | Agent-generated project scaffolds | Production code |
| **`var/sessions/`** | Persisted Web UI chat sessions | Guest registry |
| **`mcp_servers/`** | Optional on-demand MCP clones (gitignored) | Shipped tools |

### Phase 4 rename

`var/documents/` was renamed to **`var/user_files/`** to avoid confusion with `docs/`. Config key `document_rag.documents_path` is unchanged; default value is `var/user_files`. Legacy paths upgrade automatically:

- `documents/` → `var/user_files/`
- `var/documents/` → merged into `var/user_files/`

REST routes stay `/api/documents/*` for compatibility.

## Legacy short paths

User text and old config may use short forms. `upgrade_runtime_path()` maps them:

| Short / legacy | Canonical |
|----------------|-----------|
| `data/wits_memory.json` | `var/data/wits_memory.json` |
| `documents/report.pdf` | `var/user_files/report.pdf` |
| `exports/chat.txt` | `var/exports/chat.txt` |
| `logs/witsv3.log` | `var/logs/witsv3.log` |
| `workspace/myapp/` | `var/workspace/myapp/` |

Orchestrator save/export playbooks accept `exports/foo.txt`; the upgrade layer resolves to `var/exports/foo.txt`.

## `var/` subdirectories

```
var/
├── data/          # memory, FAISS index, guest profiles, MCP JSON templates
├── user_files/    # RAG uploads (formerly var/documents/)
├── exports/       # conversation exports
├── logs/          # application logs
├── workspace/     # coding agent output
├── cache/         # ephemeral caches
└── sessions/      # persisted chat JSON
```

## Do not recreate at repo root

After migration, these must not reappear as mutable roots:

`data/`, `logs/`, `documents/`, `exports/`, `workspace/`, `cache/`, `sessions/`

If you see them, restart Wits (runs `ensure_runtime_layout()`) or:

```powershell
python -c "from core.runtime_paths import migrate_legacy_runtime_dirs; print(migrate_legacy_runtime_dirs())"
```

## Related docs

- [`FILE_STRUCTURE.md`](../../FILE_STRUCTURE.md) — repo map
- [`PATH_MIGRATION_GUIDE.md`](../../PATH_MIGRATION_GUIDE.md) — 2025 moves + Phase 3–4 addendum
- [`memory.md`](memory.md) — what lives in `var/data/`
