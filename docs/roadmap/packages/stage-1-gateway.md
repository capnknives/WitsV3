---
title: "Stage 1 Briefs — Tools, MCP Mount, Inbox and Push (1.4a, 1.4b, 1.5, 1.6)"
created: "2026-11-21"
last_updated: "2026-11-21"
status: "active"
---

# Stage 1 briefs: 1.4a, 1.4b, 1.5 and 1.6

The authority is [`workspace-contracts.md`](../../architecture/workspace-contracts.md); where a
brief and the contract differ, the contract wins. Rules and the review ladder are in
[README.md](README.md). Package 1.4 is split: 1.4a holds the tool logic with no network code, and
1.4b mounts it on the MCP SDK. 1.4a, 1.5 and 1.6 share no file and run side by side; 1.4b follows
1.4a.

## WP 1.4a: Workspace tools

Review: **high**.

```
Tier: Opus, medium
Branch: claude/wp-1.4a-workspace-tools, from origin/main
Read: docs/architecture/workspace-contracts.md sections 3, 4, 5 (revoked mid-call, permission
      model), 6 (options), 8 and the seams in 9; core/workspace/store.py, approvals.py,
      clients.py, permissions.py, text.py; core/injection_guard.py; tests/core/test_workspace_clients.py
      (fixtures only, lines 1-60)
Do not read: docs/implementation/, docs/archive/, docs/roadmap/revival-2026-07.md, web/
Contract: sections 3 (arguments, results, errors, uri rule), 5, 8 (cleaning order), 9 (seams)
Build:
  - core/injection_guard.py: scan_text(text) -> list[str], using the existing patterns. Do not
    change check_tool_injection
  - core/workspace/clients.py: rate_per_min on ClientIdentity (verify_token and create_client
    fill it) and enforce_not_revoked(db, client_id), as section 5 describes. Nothing else in
    that file changes
  - core/workspace/service.py: the five functions and configure() with the section 9
    signatures. Order inside each: identity not None, permissions.check, validate arguments
    (types, ranges, the project slug, exactly one of task_id or title, the uri rule),
    clean_text with the section limits, scan_text to set flagged, call the store or
    approvals, then enforce_not_revoked after a write, then one audit row
  - get_project_state merges store.project_state with approvals.list_for_project
  - one audit row per call, success or failure, with the tool name as action and the error
    code as outcome. Audit detail never holds body text
Acceptance (tests/core/test_workspace_service.py, temp database; clients made with create_client):
  - each tool: a valid call returns the section 3 shape
  - a None identity, a missing scope, and a project outside the client's list are refused for
    every tool, and nothing is written except the audit row
  - bad arguments give invalid_argument: a bad slug, both or neither of task_id and title,
    body with task_id on claim, out-of-range lease_minutes, limit and expires_minutes, wrong
    types, over-limit text (never truncated), an unknown kind
  - uri: http and https pass; a relative path passes; a leading slash, a .. segment, a colon
    in a relative path, a space, a non-ASCII character, javascript: and file: are refused
  - cleaning: zero-width and bidirectional characters are gone from stored text; text that is
    empty after cleaning is refused where the field is required
  - flagged: text matching a guard pattern is stored with flagged = 1 and shows flagged in
    get_project_state; clean text is 0
  - revoked mid-call: with the client revoked between verification and the call, a claim
    leaves no lock, a request_decision leaves no pending approval, and both raise revoked
  - audit: exactly one row per call, with outcome ok or the error code, and no body text
  - existing tests in tests/core/test_workspace_clients.py and tests/core/test_injection_guard.py
    still pass; add a test for rate_per_min and for enforce_not_revoked
Run: pytest tests/core/test_workspace_service.py tests/core/test_workspace_clients.py tests/core/test_workspace_approvals.py tests/core/test_workspace_store.py tests/core/test_injection_guard.py -q --no-cov; ruff check core/workspace core/injection_guard.py tests/core/test_workspace_service.py; black --check on the same paths
Size: under about 450 changed lines outside the test files
Out of scope: the MCP SDK, HTTP, the auth wrapper, rate limiting (1.4b), any change to store.py,
  approvals.py, permissions.py or ratelimit.py
Report: files changed, test tail, open questions. 20 lines at most.
```

## WP 1.4b: MCP mount

Review: **high**. Starts after 1.4a merges.

```
Tier: Opus, medium
Branch: claude/wp-1.4b-mcp-mount, from origin/main (after 1.4a merges)
Read: docs/architecture/workspace-contracts.md sections 3, 5 (rate limits, other tokens), 7
      and 9; core/workspace/service.py, clients.py (verify_token, ClientIdentity),
      ratelimit.py; web/workspace_routes.py; web/owner_controls.py lines 44-54; the installed
      mcp package's own source and README for MCPServer, streamable_http_app and
      transport_security
Do not read: docs/implementation/, docs/archive/, docs/roadmap/revival-2026-07.md
Contract: section 7 (transport, SDK pin, host allowlist, auth wrapper, the confirm-first
      list), section 3 (schemas, error shape, the closing sentence of every description)
Confirm first, and say what you found in the report: the transport_security argument shape in
      mcp 2.3; that /mcp and /mcp/ both answer; how a tool handler returns isError with
      structured content. If the SDK cannot do what section 7 says, stop and report
Build:
  - requirements.txt: mcp pinned >=2.3,<3
  - web/workspace_mcp.py: register(app) builds the MCPServer with the five tools, mounts it
    at /mcp behind an ASGI auth wrapper, and calls add_lifespan for the session manager and
    service.configure from config
  - auth wrapper: Authorization Bearer only (never the query string); verify_token on every
    request; no valid token gives 401 before the SDK sees the request; the owner token and a
    guest token are refused; after a failure call auth_failure_blocked with the ASGI scope's
    client host (never X-Forwarded-For) and answer 429 when it says so; any exception from
    verify_token is a 5xx refusal, never a pass; identity goes in a ContextVar, default None
  - each tool handler: read the ContextVar, allow_call(identity.id, identity.rate_per_min) or
    return rate_limited, call the service function, turn WorkspaceError into an isError
    result with code and message (plus extra fields such as holder)
  - input schemas with additionalProperties false; every description ends with the section 3
    sentence
  - from the 1.4a review: refuse a request body over 256 KB with HTTP 413 before the SDK reads
    it; catch every exception from the service and never send exception text to the client
    (anything that is not a WorkspaceError becomes internal_error); an omitted argument is
    passed as the service default (None for claim_task body, lease_minutes, expires_minutes
    and options; an empty string for request_decision body and link_artifact title);
    integers must arrive as integers (20.0 is refused); never call the service without a
    verified identity
Acceptance (tests/web/test_workspace_mcp.py, through the mounted app with a real MCP client or
      raw JSON-RPC over httpx):
  - no token, an empty bearer, a malformed token, the owner token, a guest token and a revoked
    client token each get 401; a token in the query string is ignored
  - eleven failures from one address give 429 on the eleventh; a valid token from that
    address still works
  - tools/list returns exactly the five tools, each schema closed, each description ending
    with the required sentence
  - each tool is callable end to end and returns structured content plus the same JSON as text
  - a service error comes back as isError with its code; an unknown argument is refused
  - two clients interleaving calls each see their own identity (the required section 7 test)
  - a client past its rate gets rate_limited, as a tool error and not as an HTTP error
  - /mcp and /mcp/ both work; a Host header outside workspace.allowed_hosts is refused
  - a request body over 256 KB gets 413; a service function that raises an unexpected
    exception comes back as isError internal_error with no exception text
  - a client that disconnects mid-call still leaves exactly one audit row for that call
  - with the workspace disabled nothing is mounted at /mcp
  - the existing web test suite still passes
Run: pytest tests/web -q --no-cov; pytest tests/core/test_workspace_service.py -q --no-cov; ruff check web/workspace_mcp.py tests/web/test_workspace_mcp.py; black --check on the same paths
Size: under about 350 changed lines outside the test file
Out of scope: tool logic (1.4a), OAuth (item 5.4), registry tools (Stage 3), any edit to
  web/server.py or web/workspace_routes.py
Report: files changed, test tail, what the confirm-first checks found, open questions. 20 lines at most.
```

## WP 1.5: Decision Inbox

```
Tier: Sonnet, medium
Branch: claude/wp-1.5-decision-inbox, from origin/main
Read: docs/architecture/workspace-contracts.md sections 5 (owner routes), 6 (resolution) and
      8 (rendering, cleaning the owner's note); core/workspace/approvals.py (resolve,
      list_pending and their docstrings); web/workspace_routes.py; web/workspace_clients_api.py
      (the pattern to follow); web/static/app.js lines 1-70 (where the owner token lives:
      localStorage key wits_token); web/static/settings.html (page shell and stylesheet link)
Do not read: docs/implementation/, docs/archive/, docs/roadmap/revival-2026-07.md, the rest of
      web/static/app.js, web/server.py
Contract: sections 5, 6 and 8
Build:
  - web/workspace_approvals_api.py: register(app) adding GET /api/workspace/approvals (live
    pending, from list_pending) and POST /api/workspace/approvals/{id}/resolve with JSON
    {choice, note}, both depending on require_owner. The note is cleaned with clean_text
    (limit 2,000, not required). resolve returning None gives 409; a choice outside the
    options gives 400. The same module serves GET /inbox and GET /inbox.js with
    Cache-Control: no-store
  - web/static/inbox.html and inbox.js: one static page, no build step, no framework. It
    reads wits_token from localStorage, lists pending cards (client name, project, title,
    body, time left, a warning when flagged), one button per option, an optional note, and
    refreshes after a resolve and every 30 seconds. Usable one-handed on a phone
  - rendering rule from section 8: textContent only, never innerHTML or insertAdjacentHTML;
    no agent text in an attribute, a URL or a style
Acceptance (tests/web/test_workspace_inbox.py):
  - both API routes refuse no token, an empty bearer, a guest token and a client token
  - the owner lists a pending approval with its options, client name and flagged value
  - approve, deny and a custom option each resolve to the right status; a second resolve,
    an expired approval and an unknown id give 409; a choice outside the options gives 400
  - the note is cleaned (a zero-width character is gone) and an over-limit note is refused
  - /inbox and /inbox.js are served; with the workspace disabled they are not
  - inbox.js contains no innerHTML, outerHTML, insertAdjacentHTML or document.write (a
    source check in the test)
Run: pytest tests/web/test_workspace_inbox.py tests/web/test_workspace_routes.py -q --no-cov; ruff check web/workspace_approvals_api.py tests/web/test_workspace_inbox.py; black --check on the same paths
Size: under about 400 changed lines outside the test file
Out of scope: push, the PWA manifest and service worker, links from other pages, any edit to
  web/server.py, web/static/app.js or approvals.py
Report: files changed, test tail, open questions. 20 lines at most.
```

## WP 1.6: Phone push

```
Tier: Sonnet, medium
Branch: claude/wp-1.6-phone-push, from origin/main
Read: docs/architecture/workspace-contracts.md sections 6 (listener) and 8 (item 6) and the
      config paragraph in 9; core/workspace/approvals.py (add_created_listener and the listener
      section); core/config.py (WorkspacePushSettings only); web/workspace_routes.py (how
      optional modules are registered)
Do not read: docs/implementation/, docs/archive/, docs/roadmap/revival-2026-07.md
Contract: section 6 (listener signature, timeout, failures only logged), section 8 item 6
Build:
  - core/workspace/notify.py: register(app) reads workspace.push from config. With provider
    none it does nothing. With provider ntfy it adds one created-listener that POSTs to
    <ntfy_url>/<topic> with httpx (async, 5-second timeout)
  - the topic comes only from the WITSV3_NTFY_TOPIC environment variable. Provider ntfy with
    no topic logs one warning at registration and sends nothing
  - the message holds the client name and the words "decision waiting", nothing else: no
    title, body, project or approval id. The topic never appears in a log line
  - register is safe to call twice without adding a second listener
Acceptance (tests/core/test_workspace_notify.py, with an httpx mock transport, no network):
  - provider none: no listener is added and no request is made
  - provider ntfy with a topic: creating an approval sends exactly one POST to the right URL,
    and the request body and headers contain the client name and no title or body text
  - a 500 response, a connection error and a slow endpoint each leave approvals.create
    successful; the failure is logged without the topic
  - provider ntfy with no topic: one warning, no request
  - registering twice sends one push per approval
Run: pytest tests/core/test_workspace_notify.py tests/core/test_workspace_approvals.py -q --no-cov; ruff check core/workspace/notify.py tests/core/test_workspace_notify.py; black --check on the same paths
Size: under about 150 changed lines outside the test file
Out of scope: Web Push, the service worker, the Inbox page, any change to approvals.py or config.py
Report: files changed, test tail, open questions. 20 lines at most.
```
