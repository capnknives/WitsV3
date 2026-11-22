---
title: "WitsV3 Folder Access Contract"
created: "2026-11-22"
last_updated: "2026-11-22"
status: "active"
---

# Folder access contract

Decided by the owner on November 22, 2026. WITS works inside one base folder.
Anything else needs the owner's approval, per folder, with an expiry. Nothing is
granted or indexed automatically. Briefs F1 to F4 in
[`../roadmap/packages/folder-access.md`](../roadmap/packages/folder-access.md)
cite section numbers here. Reviewed against `main` @ `5bad82b`.

## 1. Why

On November 22 a test message ("list the files in the docs folder") made the old
path index the configured ingest root, `D:\Downloads`, which held a credentials
file. What the code does today:

- `D:\Downloads` is the default read root and ingest root in code
  (`core/config.py:241,245`) and in `config.yaml:170-173`.
  `document_rag.auto_ingest_on_startup` defaults to true in code (`:425`).
- Reads go through one check, `resolve_allowed_read_path`
  (`core/filesystem_policy.py:113`). The project root is always readable,
  including `.env`.
- Writes go through `resolve_within_project` (`core/safe_code_editor.py:109`):
  anywhere inside the repo, including `.env` and `.git`.
- Indexing (`tools/document_tools.py:214-222`) walks its roots with no per-file
  check and filters on extension only. `.json`, `.py` and `.log` are indexed.
- There is no filter for secret files anywhere.

## 2. Folders

`folders.base` (repo default `var/wits_files`; the owner's PC sets `D:\WITS` in
`config.local.yaml`) with three fixed subfolders, created at start-up:

| Folder | WITS may | Purpose |
|--------|----------|---------|
| `inbox` | read, index | Files the owner drops in for WITS |
| `outbox` | read, write | Everything WITS saves or exports for the owner |
| `projects` | read, write | Code and documents WITS creates |

Code defaults for `security.filesystem_read_roots` and
`security.document_ingest_roots` become empty lists, and
`auto_ingest_on_startup` becomes false. A list in `config.local.yaml` replaces
the base list (`core/config.py:713`), it does not add to it.

## 3. One check

`core/path_access.py` is the only place that decides. Every file tool, the
indexer and the export routes call it. It never raises anything but
`PermissionError`, and it resolves symlinks and `..` before deciding.

```python
Mode = Literal["read", "write", "index"]
def decide(path: str | Path, mode: Mode, role: str, grants: Sequence[Grant]) -> Decision
# Decision(allowed: bool, reason: str, requestable: bool, resolved: Path)
async def check(path, mode, role="owner") -> Path      # loads live grants, raises PermissionError
```

Order of rules, first match wins:

1. **Never** (section 4): refused, `requestable=False`.
2. Guest role: project documentation paths only, as today. Guests cannot request.
3. Base folders, by the table in section 2.
4. The repo itself: read allowed (minus section 4). Write only through the
   verified-edit pipeline, which keeps its own check.
5. A live grant covering the path with a mode that allows the action
   (`read` allows read; `read_write` allows read and write; `index` allows read
   and index).
6. Otherwise refused, `requestable=True`.

## 4. Never

Refused for every mode, even with a grant, and never requestable:

- A drive root, the Windows and Program Files trees, the user profile root
  itself and its `AppData`, and the home directory root on other systems.
- Any path with a component named `.ssh`, `.gnupg`, `.aws`, `.azure`, `.kube`,
  `.git` (write and index only; reading `.git` inside the repo stays allowed),
  or a browser profile folder (`User Data`, `.mozilla`, `Profiles` under a
  browser vendor folder).
- Any file whose name matches, case-insensitively: `.env`, `.env.*`, `*.pem`,
  `*.key`, `*.pfx`, `*.p12`, `*.kdbx`, `id_rsa*`, `id_ed25519*`,
  `client_secret*.json`, `*credentials*.json`, `token*.json`, `*.keystore`,
  `secrets.*`.

The file-name rule applies inside the base folders and the repo too. The
indexer skips such files and counts them in its summary.

## 5. Requests and grants

A request is an approval in the existing queue (workspace contracts, section 6)
with a row that says what it is for. Migration `0002_folder_access.sql`:

```sql
INSERT INTO clients (id, name, token_prefix, token_hash, scopes, projects, created_at)
  VALUES ('cli_internal_wits', 'WITS', 'internal', 'x', '', '', strftime('%s','now'));
CREATE TABLE folder_requests (
  approval_id TEXT PRIMARY KEY REFERENCES approvals(id),
  path TEXT NOT NULL, mode TEXT NOT NULL CHECK (mode IN ('read','read_write','index')),
  reason TEXT NOT NULL, created_at INTEGER NOT NULL);
CREATE TABLE folder_grants (
  id TEXT PRIMARY KEY,                         -- grt_ + token_hex(8)
  approval_id TEXT NOT NULL UNIQUE REFERENCES approvals(id),
  path TEXT NOT NULL, mode TEXT NOT NULL CHECK (mode IN ('read','read_write','index')),
  created_at INTEGER NOT NULL, expires_at INTEGER NOT NULL, revoked_at INTEGER);
CREATE INDEX idx_folder_grants_live ON folder_grants(expires_at, revoked_at);
```

- The internal client has no usable token: `token_hash` is not a sha256 value,
  so `verify_token` can never match it. It exists only so the foreign key holds.
  It never appears in the clients list route and cannot be revoked.
- **Request.** `request_folder_access(path, mode, reason)` runs section 3 with
  no grants. A "never" path is refused at once. Otherwise it creates the
  approval (project `wits`, title "Folder access: <mode> <path>", body the
  reason, options `["approve", "one day"]`; the server appends `deny`) and the
  `folder_requests` row in one transaction. It returns the approval id and
  does not wait. At most 3 live pending folder requests; a second request for
  the same path and mode returns the existing one.
- **Grant.** Only the owner's resolve route creates a grant, in the same
  transaction as the resolution: `approve` gives 30 days, `one day` gives 24
  hours, `deny` gives nothing. This amends workspace contracts section 6
  ("approving records a choice and runs nothing"): a folder request is the one
  case where the server acts on a resolution, and the act is one insert.
- **Folders, not files.** A grant covers a folder and everything under it,
  minus section 4. A request for a file is recorded against its parent folder.
- **Expiry and revoke.** A grant is live while `expires_at > now` and
  `revoked_at` is null. The owner can revoke from the Inbox page. Nothing
  renews by itself.
- **Indexing** runs only on the `inbox`, on a folder with a live `index` grant
  that the owner names, or from the owner's reindex route. No chat message
  starts indexing of anything else.
- **Audit.** `folder.request`, `folder.grant`, `folder.revoke`, and every
  refused `check` outside the base folders, each one row in `audit_log`.

## 6. What does not change

- The verified-edit pipeline and self-repair keep `resolve_within_project`.
- `var/` runtime data stays where it is.
- Guests keep today's limits.
- The five MCP coordination tools are untouched; outside agents cannot request
  folders.
