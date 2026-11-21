---
title: "WP 1.0 — Workspace Contracts"
created: "2026-11-21"
last_updated: "2026-11-21"
status: "active"
---

# WP 1.0: Workspace contracts

```
Tier: Lead (Fable 5.2, medium). Review: high, fresh context
Branch: claude/wp-1.0-contracts, from origin/main
Read: core/session_store.py (76 lines), core/escalation.py (218),
      core/guest_access.py (490), core/guest_policy_loader.py (70),
      core/auth_manager.py (202), core/injection_guard.py (37),
      core/tool_registry.py lines 1-120, web/guest_auth.py (89),
      docs/roadmap/modernization-2026-11.md (design rules and approval policy)
Do not read: docs/implementation/, docs/archive/, docs/roadmap/revival-2026-07.md
Output: docs/architecture/workspace-contracts.md, under 300 lines. No code.
```

The file must settle, with one short rationale each:

1. **SQLite schema** (DDL, ready to paste): `clients`, `tasks`, `task_locks`,
   `handoffs` (append-only: no UPDATE or DELETE path), `approvals`,
   `artifacts`. Include indexes, a `schema_version` table and the migration
   approach.
2. **Driver and access pattern.** The repo is fully async with no synchronous
   I/O in hot paths. Pick the driver (for example `aiosqlite`, or stdlib
   `sqlite3` in a thread) and state WAL mode, busy timeout and the
   one-writer rule.
3. **Five tool signatures**: `claim_task`, `append_handoff`,
   `request_decision`, `link_artifact`, `get_project_state`. For each: JSON
   schema of arguments, result shape, errors, required scope, and whether it
   needs owner approval under the approval policy.
4. **Lock semantics**: lease length, renewal, expiry, what a second claimer sees.
5. **Token model**: format, hashing at rest, scopes (at least `read`, `write`,
   `decide:request`), revocation, rate limits, and how it relates to the
   existing owner token and guest tokens. The guest allowlist becomes the
   general permission model (design rule 2).
6. **Approval lifecycle**: states, expiry, who may resolve, the audit record.
   Reuse the pattern in `core/escalation.py`.
7. **MCP transport**: SDK choice and version, mount point `/mcp`, how auth
   headers map to a client row. Confirm the current MCP spec version; do not
   assume the one named in the roadmap.
8. **Inbound text**: where `core/injection_guard.py` runs, and whether 37
   lines is enough for text arriving from other agents.
9. **File map**: the new modules packages 1.1 to 1.6 will create, so no two
   packages touch the same file. Nothing new goes into `web/server.py` beyond
   registering a router.

Acceptance:

- Every later brief (1.1 to 1.6) can be written by citing a section number here.
- The high-effort reviewer signs off on sections 5, 6 and 8.
- The owner has read sections 3 and 5.

Out of scope: writing any module, changing `requirements.txt` (1.1 does that),
A2A, OAuth.
