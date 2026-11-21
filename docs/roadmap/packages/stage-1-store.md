---
title: "Stage 1 Briefs — Store and Gateway Shell (1.1a, 1.1b)"
created: "2026-11-21"
last_updated: "2026-11-21"
status: "draft"
---

# Stage 1 briefs: 1.1a and 1.1b

Draft until [`workspace-contracts.md`](../../architecture/workspace-contracts.md) (WP 1.0) is
merged; neither package starts before that. Rules and the review ladder are in
[README.md](README.md). The contract split package 1.1 in two: 1.1a first, then 1.1b. Packages 1.2
and 1.3 need both.

## WP 1.1a: Workspace store

```
Tier: Sonnet, medium
Branch: claude/wp-1.1a-workspace-store, from origin/main
Read: docs/architecture/workspace-contracts.md sections 1, 2, 3, 4 and the seams in 9;
      core/runtime_paths.py lines 52-62; tests/conftest.py; pyproject.toml (pytest settings)
Do not read: docs/implementation/, docs/archive/, docs/roadmap/revival-2026-07.md, web/
Contract: sections 1 (schema, migrations), 2 (driver), 3 (result shapes), 4 (locks), 9 (seams)
Build:
  - core/workspace/__init__.py, db.py, store.py, text.py, migrations/0001_init.sql (section 1 DDL
    verbatim), with exactly the signatures in section 9
  - WorkspaceError(code, message, extra=None) in store.py; every store function accepts a
    keyword-only now: int | None = None so tests control time
  - store functions enforce project binding and lock rules; they do not clean text, check scopes
    or validate uri (1.4 does)
  - requirements.txt: add aiosqlite with a pin
Acceptance (tests/core/test_workspace_store.py, each on a temp database file):
  - a fresh database migrates to version 1; reopening applies nothing; a migration that fails
    leaves no objects behind
  - update, delete and INSERT OR REPLACE on handoffs and audit_log all abort
  - claim: fresh claim; second claimer gets task_locked with holder and expiry; holder renewal
    keeps claimed_at; takeover at expiry; a lapsed holder gets not_lock_holder on done
  - handoff kinds: note keeps the lock and never shortens the lease; handoff, blocked and done
    release it; done closes the task; a closed task gives task_closed
  - a task_id from another project gives not_found in claim_task, append_handoff, link_artifact
  - link_artifact twice with one uri returns created false and the same id
  - project_state: caps, ordering and since_handoff_id as in section 3
  - two connections writing at once: both rows land (the two-writer test)
  - clean_text: drops tag, zero-width, bidirectional and variation-selector characters; keeps
    newline and tab; rejects over-limit text instead of truncating
Run: pytest tests/core/test_workspace_store.py -q --no-cov; ruff check core/workspace tests/core/test_workspace_store.py; black --check core/workspace tests/core/test_workspace_store.py
Out of scope: clients, tokens, approvals logic, config, anything under web/, the MCP server
Report: files changed, test tail, open questions. 20 lines at most.
```

Tests need client rows; insert them with plain SQL in a fixture. `clients.py` belongs to 1.2.

## WP 1.1b: Gateway shell

```
Tier: Sonnet, medium
Branch: claude/wp-1.1b-gateway-shell, from origin/main (after 1.1a merges)
Read: docs/architecture/workspace-contracts.md sections 5 (owner routes only), 7 (host
      allowlist) and 9; web/server.py lines 125-165; web/owner_controls.py lines 40-80;
      core/config.py lines 320-345 and 525-600; config.yaml (top-level keys only)
Do not read: docs/implementation/, docs/archive/, docs/roadmap/revival-2026-07.md
Contract: sections 5 (require_owner), 9 (registration, config block, seams)
Build:
  - core/config.py and config.yaml: the workspace block from section 9, enabled false by default
  - web/workspace_routes.py: register_workspace, get_db, require_owner, add_lifespan, as in
    section 9. It wraps the app's existing lifespan and never replaces it
  - web/server.py: one added call to register_workspace(app), nothing else
Acceptance (tests/web/test_workspace_routes.py):
  - disabled: nothing is mounted and no database file is created
  - enabled with WITSV3_WEB_TOKEN unset or empty: startup refuses with a clear error
  - require_owner: no token, an empty bearer, a guest-shaped token and a wits_ client-shaped
    token are refused; the owner token passes; a non-ASCII token gives 401, not 500
  - a listed module that does not exist is skipped; a stub module's register(app) is called;
    an ImportError raised inside an existing module is not swallowed
  - contexts added with add_lifespan are entered on startup and exited on shutdown, and the
    app's existing startup still runs
  - the existing web test suite still passes
Run: pytest tests/web -q --no-cov; ruff check web/workspace_routes.py core/config.py tests/web/test_workspace_routes.py; black --check on the same paths
Out of scope: client routes, approval routes, the MCP mount, any other edit to web/server.py
Report: files changed, test tail, open questions. 20 lines at most.
```
