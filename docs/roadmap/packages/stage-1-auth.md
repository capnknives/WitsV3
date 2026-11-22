---
title: "Stage 1 Briefs — Client Tokens and Approval Queue (1.2, 1.3)"
created: "2026-11-21"
last_updated: "2026-11-21"
status: "active"
---

# Stage 1 briefs: 1.2 and 1.3

Both start after 1.1a and 1.1b merge, and run side by side: they share no file. The authority is
[`workspace-contracts.md`](../../architecture/workspace-contracts.md) (WP 1.0); where a brief and
the contract differ, the contract wins. Rules and the review ladder are in [README.md](README.md).

## WP 1.2: Client tokens

Review: **high** (independent reviewer sees this brief and the diff only).

```
Tier: Opus, medium
Branch: claude/wp-1.2-client-tokens, from origin/main (after 1.1b merges)
Read: docs/architecture/workspace-contracts.md sections 1 (clients, task_locks, approvals,
      audit_log), 5 and the seams in 9; core/workspace/db.py; core/workspace/store.py (audit,
      WorkspaceError only); web/workspace_routes.py; core/guest_access.py lines 68-95 (the shape
      permissions.py follows) and 441-486 (to mint a guest token in tests)
Do not read: docs/implementation/, docs/archive/, docs/roadmap/revival-2026-07.md,
      core/auth_manager.py (section 5 forbids reusing it)
Contract: section 5 (format, hashing, scopes, verification, revocation, rate limits, owner
      routes), section 9 (seams)
Build:
  - core/workspace/clients.py: ClientIdentity, create_client, verify_token, revoke_client and a
    list function for the owner route, with the section 9 signatures. Token format, sha256 at
    rest, hmac.compare_digest, prefix collision retry, last_seen_at at most once a minute
  - core/workspace/permissions.py: scope to tool-name map, default deny; check() raises
    WorkspaceError forbidden_scope or forbidden_project
  - core/workspace/ratelimit.py: allow_call token bucket per client; auth_failure_blocked with
    the 10-a-minute rule, a 1,024-entry cap and an injectable clock
  - web/workspace_clients_api.py: register(app) adding GET and POST /api/workspace/clients and
    POST /api/workspace/clients/{id}/revoke, each depending on require_owner. POST returns the
    token once; GET never returns a token, hash or anything that rebuilds one
  - every create and revoke writes an audit_log row (client.create, client.revoke); the token
    never reaches a log line, an audit row or an exception message
Acceptance (tests/core/test_workspace_clients.py, each on a temp database file):
  - a created token matches the section 5 regex; only its prefix and sha256 are stored
  - verify_token: the right token passes; a wrong secret with a real prefix, a malformed
    token, an empty string, a non-ASCII string, an expired client and a revoked client all
    return None without raising
  - revocation in one transaction: revoked_at set, the client's locks deleted, its live
    pending approvals cancelled with resolved_at, already-resolved approvals untouched, rows
    kept; a second revoke returns False
  - last_seen_at changes on first use and not again within a minute
  - permissions: each scope reaches only its own tools; an unknown tool is denied; scopes do
    not imply each other; a project outside clients.projects is refused; '*' allows any
  - rate limit: the bucket refuses past rate_per_min and refills; after 10 failures from one
    address further failures are blocked; a success is never blocked; the table never
    exceeds 1,024 entries
  - routes: no token, an empty bearer, a guest token and a client token are refused on all
    three routes; the owner creates, lists and revokes; the list response holds no token or hash
  - nothing logged during the test run contains a full token (caplog check)
Run: pytest tests/core/test_workspace_clients.py tests/core/test_workspace_store.py tests/web/test_workspace_routes.py -q --no-cov; ruff check core/workspace web/workspace_clients_api.py tests/core/test_workspace_clients.py; black --check on the same paths
Size: under about 450 changed lines outside the test file
Out of scope: the /mcp mount and its ASGI auth wrapper (1.4), approvals logic beyond the
  cancel statement in revocation, the Inbox, any edit to web/server.py or web/workspace_routes.py
Report: files changed, test tail, open questions. 20 lines at most.
```

## WP 1.3: Approval queue

```
Tier: Opus, medium
Branch: claude/wp-1.3-approval-queue, from origin/main (after 1.1b merges)
Read: docs/architecture/workspace-contracts.md sections 1 (approvals, audit_log), 3
      (request_decision and the approvals part of get_project_state), 6 and the seams in 9;
      core/workspace/db.py; core/workspace/store.py (audit, WorkspaceError, project binding);
      core/escalation.py lines 85-135 (the pattern being reused, not the code)
Do not read: docs/implementation/, docs/archive/, docs/roadmap/revival-2026-07.md, web/
Contract: section 6 (states, options, limits, resolution, listener), section 9 (seams)
Build:
  - core/workspace/approvals.py: create, resolve, list_for_project, list_pending and
    add_created_listener, with the section 9 signatures and a keyword-only
    now: int | None = None on each query function
  - options: unique under NFKC plus casefold; one that folds to approve or deny must be
    exactly that string; deny appended if absent; the final list stored as JSON at creation.
    Text arrives already cleaned (1.4 cleans); this module does not clean or scan
  - create counts the limits inside the same BEGIN IMMEDIATE as the insert; resolve is the
    single UPDATE in section 6
  - listeners run after commit, outside the write lock, each as a task with a 10-second
    timeout; a failing or slow listener is logged and never affects the caller
  - approval.create and approval.resolve audit rows
Acceptance (tests/core/test_workspace_approvals.py, each on a temp database file):
  - create returns pending with expires_at; default options become approve and deny
  - options: duplicates under casefold or NFKC are refused; "Approve" and "DENY" are refused;
    a custom option is kept and deny is appended
  - limits: a sixth live pending for one client gives too_many_pending; an expired one frees a
    slot; a twenty-first creation in 24 hours is refused; two creates racing at the limit
    cannot both succeed
  - resolve: approve gives approved, deny gives denied, a custom choice gives answered; a
    choice outside the stored options is refused; a second resolve, an expired approval and a
    cancelled approval each return None and change nothing
  - a task_id from another project gives not_found
  - list_for_project: live pending plus those finished in the last seven days, with pending
    past expiry reported as expired and client names, not ids; list_pending holds live ones only
  - listener: called once after commit with the approval id and client name; a listener that
    raises and one that hangs both leave create successful and returning promptly
  - audit rows exist for create and resolve
Run: pytest tests/core/test_workspace_approvals.py tests/core/test_workspace_store.py -q --no-cov; ruff check core/workspace/approvals.py tests/core/test_workspace_approvals.py; black --check on the same paths
Size: under about 400 changed lines outside the test file
Out of scope: the resolve route and the Inbox (1.5), push (1.6), cleaning or scanning text
  (1.4), client revocation (1.2), any change to core/escalation.py
Report: files changed, test tail, open questions. 20 lines at most.
```

Tests in both packages need client rows. 1.3 inserts them with plain SQL in a fixture, because
`clients.py` belongs to 1.2 and the two run side by side.
