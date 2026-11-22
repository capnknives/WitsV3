CREATE TABLE schema_version (
  version INTEGER PRIMARY KEY, applied_at INTEGER NOT NULL, description TEXT NOT NULL);
CREATE TABLE clients (
  id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE,
  token_prefix TEXT NOT NULL UNIQUE,          -- 8 hex chars, the lookup key (section 5)
  token_hash TEXT NOT NULL,                   -- sha256 hex of the full token
  scopes TEXT NOT NULL,                       -- space-separated
  projects TEXT NOT NULL DEFAULT '*',         -- space-separated slugs, or '*'
  rate_per_min INTEGER NOT NULL DEFAULT 60, created_at INTEGER NOT NULL,
  expires_at INTEGER, last_seen_at INTEGER, revoked_at INTEGER);
CREATE TABLE tasks (
  id TEXT PRIMARY KEY, project TEXT NOT NULL, title TEXT NOT NULL, body TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open','done','cancelled')),
  created_by TEXT NOT NULL REFERENCES clients(id), flagged INTEGER NOT NULL DEFAULT 0,
  created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL);
CREATE INDEX idx_tasks_project_status ON tasks(project, status, updated_at);
CREATE TABLE task_locks (
  task_id TEXT PRIMARY KEY REFERENCES tasks(id),
  client_id TEXT NOT NULL REFERENCES clients(id),
  lease_seconds INTEGER NOT NULL, claimed_at INTEGER NOT NULL, expires_at INTEGER NOT NULL);
CREATE INDEX idx_task_locks_client ON task_locks(client_id);
CREATE TABLE handoffs (
  id INTEGER PRIMARY KEY AUTOINCREMENT, project TEXT NOT NULL,
  task_id TEXT REFERENCES tasks(id),          -- NULL for a project-level note
  client_id TEXT NOT NULL REFERENCES clients(id),
  kind TEXT NOT NULL CHECK (kind IN ('note','handoff','blocked','done')),
  body TEXT NOT NULL,
  flagged INTEGER NOT NULL DEFAULT 0,         -- section 8
  created_at INTEGER NOT NULL);
CREATE INDEX idx_handoffs_project ON handoffs(project, id);
CREATE TRIGGER handoffs_no_update BEFORE UPDATE ON handoffs
  BEGIN SELECT RAISE(ABORT, 'handoffs is append-only'); END;
CREATE TRIGGER handoffs_no_delete BEFORE DELETE ON handoffs
  BEGIN SELECT RAISE(ABORT, 'handoffs is append-only'); END;
CREATE TABLE approvals (
  id TEXT PRIMARY KEY, project TEXT NOT NULL, task_id TEXT REFERENCES tasks(id),
  client_id TEXT NOT NULL REFERENCES clients(id),
  title TEXT NOT NULL, body TEXT NOT NULL DEFAULT '',
  options TEXT NOT NULL,                      -- JSON array, final list as shown to the owner
  status TEXT NOT NULL DEFAULT 'pending'
    CHECK (status IN ('pending','approved','denied','answered','cancelled')),
  choice TEXT, note TEXT, flagged INTEGER NOT NULL DEFAULT 0,
  created_at INTEGER NOT NULL, expires_at INTEGER NOT NULL,
  resolved_at INTEGER, resolved_by TEXT);
CREATE INDEX idx_approvals_status ON approvals(status, expires_at);
CREATE INDEX idx_approvals_client ON approvals(client_id, created_at);
CREATE TABLE artifacts (
  id TEXT PRIMARY KEY, project TEXT NOT NULL, task_id TEXT REFERENCES tasks(id),
  client_id TEXT NOT NULL REFERENCES clients(id),
  kind TEXT NOT NULL CHECK (kind IN ('pr','commit','branch','file','url','doc')),
  uri TEXT NOT NULL, title TEXT NOT NULL DEFAULT '', created_at INTEGER NOT NULL,
  UNIQUE (project, uri));
CREATE TABLE audit_log (                      -- beyond the brief's list: policy says "logged"
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER NOT NULL,
  actor TEXT NOT NULL,                        -- client id, or 'owner'
  action TEXT NOT NULL,                       -- tool name, or 'client.create' etc.
  target TEXT, outcome TEXT NOT NULL, detail TEXT);
CREATE TRIGGER audit_no_update BEFORE UPDATE ON audit_log
  BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;
CREATE TRIGGER audit_no_delete BEFORE DELETE ON audit_log
  BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;
