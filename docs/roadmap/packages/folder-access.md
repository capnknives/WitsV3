---
title: "Folder Access Briefs"
created: "2026-11-22"
last_updated: "2026-11-22"
status: "active"
---

# Folder access briefs

Contract: [`../../architecture/folder-access.md`](../../architecture/folder-access.md).
Rules and the review ladder are in [README.md](README.md). Order: F1, F2, F3, F4.
F2 and F3 get an independent review at medium.

## WP F1: Safe defaults and the base folders

```
Tier: Sonnet, medium
Branch: claude/wp-f1-folder-defaults, from origin/main
Read: contract sections 1 and 2
      core/config.py lines 235-250, 420-430 ; config.yaml lines 48-60, 166-176
      core/runtime_paths.py lines 52-110, 288-end (ensure_runtime_layout)
      run.py lines 188-198
Do not read: docs/implementation/, docs/archive/, docs/roadmap/revival-2026-07.md, agents/
Build:
  - FolderSettings (`folders.base`, default var/wits_files) on WitsV3Config;
    runtime_paths helpers base_dir(), inbox_dir(), outbox_dir(), projects_dir();
    ensure_runtime_layout creates the three subfolders
  - code and yaml defaults: read roots and ingest roots empty, auto_ingest_on_startup false
  - the effective ingest root list is [inbox] plus document_ingest_roots;
    the effective read root list adds inbox, outbox and projects
Acceptance (tests/core/test_folder_defaults.py):
  - a default config has no path outside the repo in any root list
  - the three folders are created under a temp base and appear in the effective lists
  - a config.local.yaml root list replaces the base list (lock in today's behaviour)
  - pytest tests/ -q --no-cov still passes
Run: pytest tests/core/test_folder_defaults.py tests/ -q --no-cov; ruff and black on the CI scope
Out of scope: any path check logic (F2), grants (F3)
Report: files changed, test tail, open questions. 20 lines at most.
```

## WP F2: One path check

```
Tier: Opus, medium. Review: independent, medium
Branch: claude/wp-f2-path-access, from origin/main (after F1 is merged)
Read: contract sections 3, 4 and 6
      core/filesystem_policy.py (whole file) ; core/safe_code_editor.py lines 105-130
      tools/file_tools.py lines 25-45, 95-110, 165-180 ; tools/document_tools.py lines 205-230
      web/server.py lines 550-576 ; core/memory_export.py lines 425-440
Do not read: docs/implementation/, docs/archive/, docs/roadmap/revival-2026-07.md
Build:
  - core/path_access.py: decide() pure, check() loading grants through an
    injectable provider that returns no grants until F3
  - read_file and list_directory call check(read); write_file calls check(write)
    and no longer writes inside the repo; the indexer calls decide(index) on each
    root and each file, skips refused files and counts them; export_memory
    calls check(write)
  - resolve_allowed_read_path stays as a thin wrapper so other callers keep working
Acceptance (tests/core/test_path_access.py, temp folders, Windows and POSIX path forms):
  - one test per rule in section 3, in order, and one per bullet in section 4
  - symlink and `..` escapes from a base folder are refused
  - `.env` in the repo and `client_secret_x.json` in the inbox are refused for
    read and skipped by the indexer, with the skip counted
  - write_file to the repo is refused; to outbox and projects it works
  - a case-different path on Windows cannot dodge a never rule
  - pytest tests/ -q --no-cov passes (update tests that wrote into the repo)
Run: pytest tests/core/test_path_access.py tests/ -q --no-cov; ruff and black on the CI scope
Out of scope: grants and requests (F3), the verified-edit pipeline, guests
Report: files changed, test tail, open questions. 20 lines at most.
```

## WP F3: Requests and grants

```
Tier: Opus, medium. Review: independent, medium
Branch: claude/wp-f3-folder-grants, from origin/main (after F2 is merged)
Read: contract section 5 ; docs/architecture/workspace-contracts.md sections 1, 2, 6
      core/workspace/approvals.py lines 105-275 ; core/workspace/db.py (signatures)
      core/workspace/clients.py (list and revoke functions only)
      web/workspace_approvals_api.py (whole file) ; core/path_access.py
      tools/file_tools.py lines 1-45 (as the pattern for a tool class)
Do not read: docs/implementation/, docs/archive/, docs/roadmap/revival-2026-07.md, agents/
Build:
  - migration 0002_folder_access.sql exactly as in the contract; the internal
    client is hidden from the list route and refused by revoke
  - core/workspace/folder_access.py: request(db, path, mode, reason) -> dict;
    grant_from_resolution(conn, approval_id, choice, now) inside the resolve
    transaction; live_grants(db) ; revoke(db, grant_id)
  - the resolve route calls grant_from_resolution; path_access.check uses live_grants
  - tools/folder_access_tool.py: request_folder_access, owner role only
  - owner routes: GET /api/workspace/folder-grants, POST .../{id}/revoke
Acceptance (tests/core/test_folder_access.py, tests/web/test_folder_access_api.py):
  - request for a never path creates nothing; a normal request creates one
    approval and one folder_requests row; a repeat returns the same approval
  - approve gives a 30-day grant, "one day" a 24-hour grant, deny nothing;
    resolving twice cannot create two grants
  - an expired or revoked grant stops allowing access at once
  - a file request is recorded against its parent folder
  - no code path creates a grant without a resolved approval (search the diff:
    one INSERT INTO folder_grants, inside grant_from_resolution)
  - every route refuses no token, an empty token, a guest token and a client token
  - the internal client cannot authenticate at /mcp
Run: pytest tests/core/test_folder_access.py tests/web/test_folder_access_api.py tests/ -q --no-cov; ruff and black on the CI scope
Out of scope: the Inbox page changes (F4), the native loop's tool list (C4)
Report: files changed, test tail, open questions. 20 lines at most.
```

## WP F4: Inbox page and guide

```
Tier: Sonnet, medium
Branch: claude/wp-f4-folder-inbox, from origin/main (after F3 is merged)
Read: contract section 5 ; web/static/inbox.html, web/static/inbox.js ;
      web/workspace_approvals_api.py ; docs/guides/workspace-setup.md (headings only)
Do not read: docs/implementation/, docs/archive/, docs/roadmap/revival-2026-07.md
Build:
  - folder request cards show the path, the mode in plain words and the reason,
    with buttons "Approve 30 days", "One day", "Deny" (textContent only)
  - a "Folders WITS can use" list on the Inbox page: base folders, live grants
    with their expiry, and a Revoke button
  - docs/guides/folder-access.md: where to drop files, how requests look, how to revoke
Acceptance: tests/web/test_workspace_inbox.py extended: the page lists a grant,
      revoke removes it, card text never uses innerHTML
Run: pytest tests/web/test_workspace_inbox.py -q --no-cov; ruff and black on the CI scope
Out of scope: the UI overhaul look (Horizon) — plain markup that the overhaul will restyle
Report: files changed, test tail, open questions. 20 lines at most.
```
