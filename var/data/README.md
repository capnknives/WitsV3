# WitsV3 Runtime Data (`var/data/`)

Mutable runtime files live under **`var/`** (see `core/runtime_paths.py`). This folder holds memory, MCP config, and guest registry data.

## Tracked in git (templates / defaults)

- `mcp_config.json` — MCP server configuration
- `mcp_tools.json` — MCP tool definitions
- `*.template` — Template files for first-run setup

## Gitignored at runtime

- `wits_memory.json` — Personal conversation memory
- `neural_web.json` — Neural web state
- `guest_profiles.json`, `guest_user_profiles/`, `guest_audit/`
- `*_backup.json`, `faiss_index.*`, smoke reports

## Setup

```bash
python scripts/setup_local_data.py
```

Creates `var/` subdirs and copies templates to live files when missing.

## Legacy migration

On first startup, existing top-level `data/`, `logs/`, `documents/`, etc. are merged into `var/` automatically (`var/user_files/` for uploads; not repo `docs/`).
