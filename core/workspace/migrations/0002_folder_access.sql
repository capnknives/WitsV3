-- Folder access requests and grants (docs/architecture/folder-access.md, section 5).
-- The internal client exists only so approvals.client_id keeps its foreign key; its token_hash
-- is not a sha256 hex value and its prefix is not hex, so no token can ever verify as it.
-- On an existing database a client may already hold the name or prefix; the suffixed spelling
-- keeps the migration from failing (both columns are UNIQUE).
INSERT INTO clients (id, name, token_prefix, token_hash, scopes, projects, created_at)
  SELECT 'cli_internal_wits',
    CASE WHEN EXISTS (SELECT 1 FROM clients WHERE name = 'WITS')
      THEN 'WITS (internal_wits)' ELSE 'WITS' END,
    CASE WHEN EXISTS (SELECT 1 FROM clients WHERE token_prefix = 'internal')
      THEN 'internal_wits' ELSE 'internal' END,
    'x', '', '', strftime('%s','now')
  WHERE NOT EXISTS (SELECT 1 FROM clients WHERE id = 'cli_internal_wits');
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
-- A grant row is fixed once written: the only change allowed is revoking it, once.
CREATE TRIGGER folder_grants_revoke_only BEFORE UPDATE ON folder_grants
  WHEN NOT (OLD.revoked_at IS NULL AND NEW.revoked_at IS NOT NULL
    AND NEW.id IS OLD.id AND NEW.approval_id IS OLD.approval_id AND NEW.path IS OLD.path
    AND NEW.mode IS OLD.mode AND NEW.created_at IS OLD.created_at
    AND NEW.expires_at IS OLD.expires_at)
  BEGIN SELECT RAISE(ABORT, 'folder_grants rows only change by revoking'); END;
CREATE TRIGGER folder_grants_no_delete BEFORE DELETE ON folder_grants
  BEGIN SELECT RAISE(ABORT, 'folder_grants rows are never deleted'); END;
