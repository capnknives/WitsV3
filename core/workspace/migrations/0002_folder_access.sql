-- Folder access requests and grants (docs/architecture/folder-access.md, section 5).
-- The internal client exists only so approvals.client_id keeps its foreign key; its token_hash
-- is not a sha256 hex value and its prefix is not hex, so no token can ever verify as it.
INSERT INTO clients (id, name, token_prefix, token_hash, scopes, projects, created_at)
  VALUES ('cli_internal_wits', 'WITS', 'internal', 'x', '', '', strftime('%s','now'));
CREATE TABLE folder_requests (
  approval_id TEXT PRIMARY KEY REFERENCES approvals(id),
  path TEXT NOT NULL, mode TEXT NOT NULL CHECK (mode IN ('read','read_write','index')),
  reason TEXT NOT NULL, created_at INTEGER NOT NULL);
CREATE TABLE folder_grants (
  id TEXT PRIMARY KEY,
  approval_id TEXT NOT NULL UNIQUE REFERENCES approvals(id),
  path TEXT NOT NULL, mode TEXT NOT NULL CHECK (mode IN ('read','read_write','index')),
  created_at INTEGER NOT NULL, expires_at INTEGER NOT NULL, revoked_at INTEGER);
CREATE INDEX idx_folder_grants_live ON folder_grants(expires_at, revoked_at);
