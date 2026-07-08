# WitsV3 — Suggested Features & Roadmap

**Last updated:** July 8, 2026  
**Integration branch:** `fix/revive-2026-07` @ `2a9044f`  
**Staging branch:** `cursor/iron-delta-w0r5` (synced with integration)  
**Test suite:** 406 collected (404 passed / 2 skipped on last full run)

This is the **forward-looking** roadmap: what to add, improve, or remove next.  
For **what already shipped** during the July 2026 revival, see
[`revival-2026-07.md`](revival-2026-07.md). For **manual regression tests**
before promoting to `main`, see
[`composer-orchestrator-search-quality-2026-07.md`](composer-orchestrator-search-quality-2026-07.md)
§ "Suggested manual test plan" (tests A–F).

### Git workflow

All agent/feature work lands on **`fix/revive-2026-07`** first. Richard merges
→ **`main`** after manual testing looks good.

---

## Current snapshot

| Area | Status |
|------|--------|
| Web UI (chat, settings, personality, MCP manager) | ✅ Shipped |
| Document RAG + `document_search` | ✅ Shipped |
| Smart model routing (`core/model_router.py` + `/settings` UI) | ✅ Shipped |
| Orchestrator JSON robustness + WCCA intent JSON repair | ✅ Shipped |
| Web search multi-provider (Tavily → Brave → DDG) | ✅ Shipped |
| MCP registry discover/install + OCI/Docker + browse-before-install | ✅ Shipped |
| Tier 1–4 repo hygiene (CI, dead-code cleanup, 500-line splits, MCP on-demand-only) | ✅ Shipped |
| Coding agent + self-repair (verified edit pipeline, daily schedule) | ✅ Shipped July 8 2026 — see §1a |
| Orchestrator synthesis guard, CLI Ollama errors, chat export, adaptive-LLM deprecation | ✅ Shipped July 8 2026 (Cursor) — see updated §2 P1 |
| 3 routing bugs (word-boundary keyword matching, whole-codebase scan fallback, clarification-bypass override) | ✅ Shipped July 8 2026 — see §1a note and README changelog |
| July revival feature backlog | ✅ **Closed** — only gates + polish remain |

---

## 1. Gates before `main` promotion

These are **human verification**, not code tasks.

| # | Action | Notes |
|---|--------|-------|
| G1 | ~~Run manual tests **A–F**~~ | ✅ DONE July 8 2026 — driven live against a dedicated `claude/work` instance (no browser available): A (model routing toggle), B (Ollama-down friendly card, verified via a deliberately-unreachable Ollama URL rather than killing the real service), C (single `web_search` call, numbered sources, grounded answer), D (document upload + question answered via the orchestrator's automatic `search_memory` context injection, not a stray `web_search`), E (107 passed), F (save-to-file wrote a real file with correct USER/ASSISTANT lines). Test residue (uploaded doc, memory segments, exported log) was cleaned up afterward via the real `delete_segments` API. |
| G2 | Optional: add `ANTHROPIC_API_KEY` | Required only for ask-Claude escalation — still open, Richard's call |
| G3 | ~~Watch logs on document Q&A~~ | ✅ DONE July 8 2026 — verified with an unguessable planted codeword (`ZEBRA-QUASAR-77`) in an uploaded test document; the orchestrator answered correctly, which is only possible if it actually used the retrieved content rather than hallucinating. |

**Already done (no action):** Tavily key, Brave key, Supabase token revoke (N/A — no project),
WCCA JSON repair, MCP follow-ups, Tier 4 splits, G1/G3 manual verification.

---

## 1a. Coding agent & self-repair — from stub to real (SHIPPED July 8 2026)

**Audit finding (before this work):** both `AdvancedCodingAgent` and `SelfRepairAgent`
were LLM-prose generators with no filesystem/process I/O at all. The coding agent's
"project creation" built a `CodeProject.files` dict from **hardcoded scaffold
template strings**, held only in memory — never written to disk, never compiled,
never tested. Its debugging/refactoring handlers asked the LLM to write a generic
guide as prose. `SelfRepairAgent.run()` was a literal 2-line LLM passthrough: send
the raw user message to the LLM, stream back whatever came out — no log parsing,
no diagnosis, no fix, no verification. The more elaborate `self_repair_handlers.py`/
`self_repair_utils.py` "fix" functions were dead code (never imported by the live
agent) and were themselves simulations (`await asyncio.sleep(0.5); return True`).
Separately, `FileWriteTool` had a real path-guard check in source, but it was
**unreachable dead code** — two full `try/except` blocks back to back, where the
first always returned before the second (which held the only safety check) could
run — so `write_file` had zero enforced path restriction in practice. Process
restart existed (`run.py`'s file-watcher-triggered relaunch) but was off by
default and never invoked by any agent. Scheduling infrastructure was real
(APScheduler + cron config) but lived only in a Docker-only, normally-unstarted
`BackgroundAgent`, with no repair task defined.

**Plan:** build one shared, safety-first verified-edit pipeline and give both
agents (and the tool registry generally) a real detect → diagnose → fix → verify
→ (optionally) restart loop on top of it, then wire a scheduled autonomous run.

**Shipped:**

1. **`core/safe_code_editor.py`** — the shared pipeline every code-writing path
   now goes through: snapshot original bytes → write candidate → run pytest →
   commit to git *only* on a passing verification, or restore the exact original
   bytes (or delete a new file) on failure. A failed attempt never leaves stray
   git state and is never left broken in the working tree. `resolve_within_project()`
   is the one enforced boundary — refuses any path escaping the project directory.
2. **`tools/self_repair_tools.py`** — four new registry tools any agent can call:
   `diagnose_log_errors` (parses `logs/witsv3.log` tracebacks for file/line,
   read-only), `run_test_suite` (pytest wrapper), `apply_code_fix` (the verified
   pipeline above), `restart_app` (deliberately schedules a relaunch a few
   seconds out so the current response can finish streaming first — reuses the
   same `subprocess.Popen` + process-exit pattern as the existing file-watcher
   restart, just triggered on purpose instead of on any `.py` save).
3. **`agents/self_repair_agent.py` rewritten**: if the request names a real
   existing file, targets it directly; otherwise scans recent logs via
   `diagnose_log_errors`. For each actionable issue, reads the file, asks the LLM
   for a full corrected version, and applies it through `apply_code_fix` — verified
   fixes are reported with their commit sha, failed ones report that the file was
   reverted, never silently swallowed. Optional restart after a verified fix,
   gated by `self_repair.restart_after_fix` (default off).
4. **`AdvancedCodingAgent` gets two real capabilities**: project creation now
   writes generated files to `workspace/<project_name>/` and runs `py_compile` on
   each `.py` file, reporting per-file syntax status instead of leaving everything
   as an in-memory string; and a request naming a real existing file (e.g. "fix
   the bug in agents/foo.py") routes straight to a new `_handle_fix_existing_file`
   that reuses the exact same verify-before-commit pipeline as self-repair —
   so "coding agent" and "self-repair agent" share one trustworthy edit path
   rather than two different half-measures.
5. **`FileWriteTool` dead-code bug fixed** — collapsed the two duplicate
   try/except blocks into one, now actually enforced via
   `resolve_within_project()`, with a regression test locking in the fix.
6. **Daily autonomous schedule**: `self_repair.daily_schedule_enabled` (default
   on, cron `0 3 * * *`) runs the scan-and-fix loop from inside `WitsV3System`
   itself (`run.py`, APScheduler — no Docker required), plus a matching
   `self_repair` task added to the Docker-only `BackgroundAgent` dispatcher for
   deployment parity.
7. **New config section** `self_repair` in `core/config.py`/`config.yaml`:
   `enabled`, `daily_schedule_enabled`, `daily_schedule_cron`, `max_issues_per_run`,
   `log_scan_lines`, `restart_after_fix` (default `false`), `test_timeout_seconds`.

**Tests:** `tests/core/test_safe_code_editor.py` (9), `tests/tools/test_self_repair_tools.py` (8),
`tests/agents/test_self_repair_agent.py` (12, rewritten), `tests/agents/test_advanced_coding_agent.py`
(5, new), `tests/tools/test_file_tools.py` (2, new — locks in the path-guard fix),
`tests/agents/test_background_agent.py` (+2), `tests/test_run_self_repair_schedule.py` (5, new).

**Known limitations / honest next steps** (not done here — good follow-ups):

- The LLM is asked for a **full corrected file**, not a diff/patch — reliable for
  small-to-medium files, but large files risk truncation or an expensive
  regeneration. A diff-based apply path would scale better.
- `diagnose_log_errors` only resolves file/line from a real Python traceback
  (`Traceback (most recent call last): ... File "...", line N`). Bare
  `ERROR`/`CRITICAL` log lines without a traceback are surfaced but marked
  non-actionable — nothing tries to guess a file for those.
- FAISS/neural/Supabase memory backends never wired `_prune_if_needed()`-style
  hooks into their own `add_segment`, and likewise nothing here extends
  self-repair awareness into those backends specifically (out of scope — `basic`
  is the configured default).
- Process restart is still the same blunt `subprocess.Popen` + process-exit
  primitive that predates this work — ungraceful for the uvicorn web-server path
  (no in-flight-request draining). Fine for a CLI/dev restart; a production
  supervisor process would be a better foundation if this needs to be graceful.
- `restart_after_fix` defaults to `false` for the *scheduled* daily run
  specifically so it can never surprise someone using the app live; it's fully
  wired and tested, just off by default.

---

## 2. Recommended next work (prioritized)

### P0 — Verify & promote

1. Complete manual tests A–F on `fix/revive-2026-07` / `cursor/iron-delta-w0r5`. G1/G3 already verified live July 8 (see §1 table above) — remaining: G2 (Richard's `ANTHROPIC_API_KEY` call) and a final re-check of the merged state before promoting.
2. Merge `fix/revive-2026-07` → `main` when satisfied.

**Shipped July 8 2026 (Cursor session)** — no longer P1:
- ~~Orchestrator result synthesis guard~~ — `agents/orchestrator_tool_helpers.py::_validate_final_answer_synthesis`, guards final answers against unused search observations with a bounded retry.
- ~~Friendlier Ollama-down in CLI~~ — `core/user_errors.py::format_cli_error`, shared with the web UI's existing `format_chat_error`.
- ~~Conversation export UX~~ — one-click `POST /api/export` writes the session transcript to `exports/`.
- Also: `core/adaptive_llm_interface.py` formally deprecated (`llm_interface.default_provider: adaptive` now logs a warning and falls back to Ollama) and `tools/enhanced_reasoning.py` split under 500 lines (`enhanced_reasoning_models.py`, `enhanced_reasoning_patterns.py`).

### P1 — High leverage (code)

| Item | Why | Effort |
|------|-----|--------|
| **Conversation-history-aware intent classification** | Documented limitation from the July 8 self-repair routing fix: a short reply to WITS's own clarifying question (e.g. "Specifically the wits v3 codebase") is judged in isolation and misclassified as casual chat, never reaching task routing. See §1b for the concrete plan. | Medium |
| **Expand CI lint** | CI runs critical ruff only; pre-commit has full ruff/black/isort but ~124 black files / ~3.7k ruff findings in legacy code block full CI enforcement | Medium (incremental) |

### P2 — Structure & hygiene (second-pass splits)

Files still over the 500-line rule (excluding archived GUI):

| Lines | File | Suggestion |
|------:|------|------------|
| 690 | `tools/enhanced_reasoning.py` | Split reasoning strategies vs. prompt builders |
| 680 | `core/tool_composition.py` | Split composition engine vs. registry glue |
| 649 | `tools/neural_web_visualization.py` | Split render/export from graph queries |
| 631 | `core/neural_memory_backend.py` | Split persistence vs. query/index |
| 614 | `agents/self_repair_handlers.py` | Split handler mixins by repair category |
| 603 | `tools/neural_web_nlp.py` | Split NLP ops from visualization helpers |
| 570 | `core/content_fallback_system.py` | Evaluate: wire up or archive if unused |
| 567 | `core/response_parser.py` | Split format-specific parsers |
| 531 | `core/knowledge_graph.py` | Split graph ops from serialization |
| 527 | `core/enhanced_mcp_adapter.py` | Split transport vs. tool mapping |
| 525 | `core/memory_manager.py` | Split backend factory from segment CRUD |
| 513 | `core/concrete_meta_reasoning.py` | Split meta-reasoning steps |

**Already under 500:** `llm_driven_orchestrator.py` (494), `wits_control_center_agent.py` (399), `web/server.py` (343).

### P3 — Features (when P0–P2 are boring)

| Feature | Value | Notes |
|---------|-------|-------|
| **Scheduled background tasks UI** | Visibility into `BackgroundAgent` jobs | Read-only status page first |
| **Memory browser in web UI** | Inspect/search/prune segments without CLI | Builds on existing memory APIs |
| **Multi-session chat history** | Named sessions, resume later | Needs session store + UI |
| **Tool usage analytics** | Which tools fire, latency, failure rate | Log aggregation or lightweight metrics |
| **MCP server health dashboard** | Connection state, last error, restart | Extend `/mcp` page |
| **Local model pull helper** | Web UI button to `ollama pull` missing models | Reduces "model not found" friction |
| **Document upload via web** | Drag-drop into `documents/` + ingest | Complements folder drop |
| **Streaming tool progress** | Richer SSE for long tool runs | Better UX for search/RAG |

### P4 — Neural web (only if actively using it)

The neural web stack is **wired** (tools register, backend optional) but is not
the primary user-facing path today. Before investing here, decide whether neural
web remains a product direction or a research artifact.

If continuing: see historical ideas in
[`neural-web-roadmap.md`](neural-web-roadmap.md) — concept clustering, adaptive
learning, Docker tool execution — but treat that doc as **2025-era design**, not
current priority.

---

## 3. Remove, archive, or simplify (redundancy & defunct)

### Code — candidates for archive or deletion

| Item | Verdict | Rationale |
|------|---------|-----------|
| `core/adaptive_llm_interface.py` | **Archive or delete** | Marked dormant; routes to non-existent on-disk "modules". Replaced by `core/model_router.py`. Safe to move to `planning/archive/` after confirming no imports in hot path |
| `planning/archive/gui/` (PyQt6) | **Keep archived** | Web UI is permanent replacement; do not revive unless explicit new requirement |
| Supabase memory backend + MCP | **Park / optional trim** | No active Supabase project; backend code can stay for opt-in users but remove from default docs/examples. `supabase-mcp` already removed from default `data/mcp_tools.json` |
| `core/content_fallback_system.py` | **Audit usage** | 570 lines; grep for callers — archive if nothing in production path uses it |
| `core/cognitive_architecture*.py` remnants | **Audit** | May overlap with enhanced reasoning; consolidate or archive |
| Root-level duplicate configs | **Done** | pytest.ini, mypy.ini, etc. already removed (Tier 1) |

### Documentation — consolidate to reduce confusion

| Document | Verdict |
|----------|---------|
| **`suggested-features-2026-07.md`** (this file) | **Canonical forward roadmap** |
| **`revival-2026-07.md`** | **Canonical history + shipped log** — keep, but do not duplicate "what's next" here |
| **`composer-orchestrator-search-quality-2026-07.md`** | **Historical handoff** — keep for manual test plan A–F and Tier 1–4 audit detail; marked superseded for "what's next" |
| **`neural-web-roadmap.md`** | **Historical** — 2025 design; header already flags this |
| **`planning/tasks/task-management.md`** | **Supersede** — 2025 phases; redirect to this file |
| **`TASK.md`** | **Redirect only** — point here + revival doc |
| **Synthetic brain doc set** (`docs/SYNTHETIC_BRAIN.md` + archives) | **Keep with historical headers** — canonical entry is `docs/SYNTHETIC_BRAIN.md` |

### Config / deps — optional cleanup

| Item | Action |
|------|--------|
| `config.yaml` `supabase:` section | Keep but document as optional; ensure README says "skip if unused" |
| `requirements.lock` still lists `supabase` | Keep for optional backend; or move to `[optional]` extra |
| Gmail MCP entry in `data/mcp_tools.json` | Remove if not planning to connect; harmless config-only today |
| `.cursorrules` PLANNING.md pointer | Verify still points at `planning/architecture/system-architecture.md` |

---

## 4. Parked (explicitly out of scope)

Do **not** start these unless requirements change:

- **Docker packaging** — no container story yet; local dev + web UI is the path
- **Supabase cloud sync** — no project; backend remains opt-in code only
- **PyQt6 desktop GUI** — archived; web UI + PWA is the client
- **Git submodules for MCP servers** — replaced by on-demand clone (`scripts/clone_mcp_servers.py`) + registry `npx`/`uvx`/`docker run`
- **Phase 2 "Neural Web" as primary product** — research direction only until explicitly prioritized

---

## 5. Quick reference — document map

```
planning/roadmap/
├── suggested-features-2026-07.md   ← YOU ARE HERE (what's next)
├── revival-2026-07.md              ← what shipped + error triage log
├── composer-orchestrator-search-quality-2026-07.md  ← manual tests A–F + Tier audit history
├── clutter-catalog-2026-07.md       ← July 8 audit: orphan/dormant/dead code inventory + delete waves
├── config-surface-truth-2026-07.md  ← July 8 audit: which config.yaml keys are actually live vs. silently ignored
├── tool-registry-reality-2026-07.md ← July 8 audit: which registered tools are actually used vs. dead weight
├── neural-web-roadmap.md           ← historical (2025)
└── README.md                       ← index
```

The three July 8 audit docs are read-only inventories (no code changes implied by
existing); their "cleanup waves" feed into §3 below. One finding was fixed
immediately since it was a live correctness bug directly caused by this
session's own work: `config/background_agent.yaml`'s `self_repair` task ran on
the identical `0 3 * * *` cron as `run.py`'s in-process scheduler — running the
Docker background agent alongside the main app would have fired self-repair
twice. Now `enabled: false` by default; see
[`config-surface-truth-2026-07.md`](config-surface-truth-2026-07.md) §7.

**Workflow:** implement on feature branch → push/merge to `fix/revive-2026-07` → manual test → Richard merges to `main`.

---

## 6. Suggested sprint order (if picking one chunk)

1. **Day 1:** Manual tests A–F + promote to `main` if green.
2. **Day 2–3:** Orchestrator synthesis guard + CLI Ollama-down errors.
3. **Week 2:** Incremental ruff/black cleanup (10–20 files per PR) until CI can run full lint.
4. **Week 3+:** Second-pass 500-line splits starting with `enhanced_reasoning.py` and `tool_composition.py`.
5. **When bored:** Web export button, memory browser, document upload.

---

*When this doc goes stale, update the header date and reconcile against `git log fix/revive-2026-07` and `pytest --collect-only`.*
