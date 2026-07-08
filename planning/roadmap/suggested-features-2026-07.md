# WitsV3 — Suggested Features & Roadmap

**Last updated:** July 7, 2026  
**Integration branch:** `fix/revive-2026-07` @ `14adeed`  
**Staging branch:** `cursor/iron-delta-w0r5` (synced with integration)  
**Test suite:** 337 collected (329 passed / 2 skipped on last full run)

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
| July revival feature backlog | ✅ **Closed** — only gates + polish remain |

---

## 1. Gates before `main` promotion

These are **human verification**, not code tasks.

| # | Action | Notes |
|---|--------|-------|
| G1 | Run manual tests **A–F** | See composer handoff doc; especially **F** (save-to-file): *"Save a log of our conversations as exports/chat_log.txt"* |
| G2 | Optional: add `ANTHROPIC_API_KEY` | Required only for ask-Claude escalation |
| G3 | Watch logs on document Q&A | Confirm orchestrator **uses** `document_search` results instead of hallucinating (should improve after structured JSON, but verify) |

**Already done (no action):** Tavily key, Brave key, Supabase token revoke (N/A — no project),
WCCA JSON repair, MCP follow-ups, Tier 4 splits.

---

## 2. Recommended next work (prioritized)

### P0 — Verify & promote

1. Complete manual tests A–F on `fix/revive-2026-07` / `cursor/iron-delta-w0r5`.
2. Merge `fix/revive-2026-07` → `main` when satisfied.

### P1 — High leverage (code)

| Item | Why | Effort |
|------|-----|--------|
| **Orchestrator result synthesis guard** | Logs still occasionally show answers that ignore tool observations; add explicit post-tool check or synthesis prompt tightening | Medium |
| **Expand CI lint** | CI runs critical ruff only; pre-commit has full ruff/black/isort but ~124 black files / ~3.7k ruff findings in legacy code block full CI enforcement | Medium (incremental) |
| **Friendlier Ollama-down in CLI** | Web UI has `web/user_errors.py`; CLI still surfaces raw connection errors | Small |
| **Conversation export UX** | Save-to-file works via orchestrator; add a one-click "Export chat" button or slash command in web UI | Small–Medium |

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
├── neural-web-roadmap.md           ← historical (2025)
└── README.md                       ← index
```

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
