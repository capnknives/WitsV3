# WitsV3 — Suggested Features & Roadmap

**Last updated:** July 8, 2026  
**Working branch tip:** `cursor/work` (merges into `fix/revive-2026-07`)  
**Test suite:** 435 collected — re-run `pytest -q` before claiming green

This is the **canonical forward roadmap**: what to add, improve, or remove next.  
For **what already shipped**, see [`revival-2026-07.md`](revival-2026-07.md) and the root [`README.md`](../../README.md).  
For audits (clutter / tools / config), see the July 8 catalog docs in this folder.

### Git workflow

Feature work → **`fix/revive-2026-07`** → Richard merges to **`main`** after manual verification.

---

## Current snapshot

| Area | Status |
|------|--------|
| Web UI (chat, settings, personality, MCP manager, chat export) | ✅ Shipped |
| Document RAG + `document_search` | ✅ Shipped |
| Smart model routing (`core/model_router.py` + `/settings`) | ✅ Shipped |
| Orchestrator JSON robustness + WCCA intent JSON repair | ✅ Shipped |
| Orchestrator result synthesis guard | ✅ Shipped (`4e37d6a`) |
| Friendlier Ollama-down errors (Web + CLI) | ✅ Shipped |
| Web search multi-provider (Tavily → Brave → DDG) | ✅ Shipped |
| MCP registry discover/install + OCI/Docker + browse-before-install | ✅ Shipped |
| Tier 1–4 repo hygiene (CI, dead-code cleanup, splits, MCP on-demand) | ✅ Shipped |
| Coding agent + self-repair (verified-edit pipeline, daily schedule) | ✅ Shipped July 8 |
| Routing bugfixes (keyword word-boundaries, whole-codebase scan fallback, clarification-bypass) | ✅ Shipped July 8 |
| Dual self-repair schedule fix (Docker bg agent off by default) | ✅ Shipped July 8 |
| Docs pass (README install guide, redirects, roadmap truth) | ✅ Shipped July 8 |
| Full CI lint (ruff E/F/W/I/UP/B + black --check) | ✅ Shipped July 8 |
| Owner-gated `/shutdown` + `/restart` (chat + API, token-gated) | ✅ Shipped July 8 |
| Web UI modernization (centered header, sleeker chat/composer) | ✅ Shipped July 8 |
| Memory browser (search + recent w/ pagination + gated prune) | ✅ Shipped July 8 |
| Adaptive LLM stack archived; clutter Wave A/B cleanup | ✅ Shipped July 8 |
| July revival feature backlog | ✅ **Closed** — polish + optional features remain |

---

## 1. Gates before `main` promotion

Human verification, not code tasks.

| # | Action | Notes |
|---|--------|-------|
| G1 | ~~Manual tests A–F~~ | ✅ DONE July 8 2026 — see revival / composer handoff notes |
| G2 | Optional: `ANTHROPIC_API_KEY` | Ask-Claude only — still open, owner's call |
| G3 | ~~Document Q&A grounded answers~~ | ✅ DONE July 8 2026 |

**Already done:** Tavily/Brave keys (when available), WCCA JSON repair, MCP follow-ups, Tier 4 splits, G1/G3.

---

## 1a. Coding agent & self-repair — SHIPPED July 8 2026

Summary of what landed (detail also in README § Self-repair):

- Shared pipeline: `core/safe_code_editor.py` (snapshot → write → pytest → commit or restore)
- Tools: `diagnose_log_errors`, `run_test_suite`, `apply_code_fix`, `restart_app`
- Agents: real `SelfRepairAgent` + coding agent disk writes / verified file fixes
- Schedule: in-process daily cron + BackgroundAgent parity (`self_repair` task **disabled by default** in `config/background_agent.yaml` so running Docker bg + main app does not double-fire)
- Path-guard fix on `write_file`; control-center specialized-agent routing fix

**Honest follow-ups (not blocking):**

- Full-file LLM rewrite (diff/patch path would scale better for large files)
- Log diagnosis needs real Python tracebacks (bare ERROR lines stay non-actionable)
- Restart is still blunt `Popen` + exit (fine for CLI/dev; not a graceful uvicorn drain)
- `restart_after_fix` defaults **off** so scheduled runs never interrupt a live session

---

## 2. Recommended next work (prioritized)

### P0 — Verify & promote

1. ~~Complete manual tests A–F~~ ✅  
2. Keep the suite green (including traceback remapping across sibling worktree log paths)  
3. Merge `fix/revive-2026-07` → `main` when satisfied  

### P1 — High leverage

| Item | Why | Effort | Status |
|------|-----|--------|--------|
| Orchestrator synthesis guard | Ground answers in tool observations | Medium | ✅ Done |
| Friendlier Ollama-down (CLI) | Match web UX | Small | ✅ Done |
| Conversation export UX | One-click export in web UI | Small–Medium | ✅ Done |
| Conversation-history-aware intent | Short replies after a clarifying question misclassified as casual chat | Medium | Open (Claude, WCCA branch) |
| **Guest / family tester access** | LAN phone testing without owner token; device identity + safe tool subset | Medium–Large | ✅ Safe MVP July 8 — [`guest-tester-access-2026-07.md`](guest-tester-access-2026-07.md); Phase 3 content policy + Phase 4 owner admin still open |
| Expand CI lint | Full ruff/black blocked by legacy noise | Medium (incremental) | ✅ Done July 8 — CI runs full `ruff check` + `black --check` |
| Docs / README modernity | Install + honest status | Medium | ✅ Done July 8 |
| Clutter cleanup wave 1 | Delete orphans + relocate scripts (see clutter catalog §1/Wave B) | Small–Medium | ✅ Done July 8 — Wave A deletes confirmed; `ollama_probe`/`analyze_memory`/`llm_diagnostic_basic` → `scripts/` |

### P2 — Structure & hygiene (second-pass splits)

Files still over ~500 lines (excluding archived GUI). Re-measure before splitting — several already dropped after July work:

| Approx. lines | File | Suggestion |
|-------------:|------|------------|
| 735 | `tools/neural_web_nlp.py` | Split NLP ops from helpers |
| 772 | `tools/neural_web_visualization.py` | Split render/export from graph queries |
| 704 | `core/tool_composition.py` | Composition engine vs registry glue |
| 632 | `agents/self_repair_handlers.py` | **Prefer delete** — orphan after rewrite (see clutter catalog) |
| 607 | `core/memory_manager.py` | Backend factory vs segment CRUD |
| ~500+ | `core/neural_memory_backend.py`, `knowledge_graph.py`, `response_parser.py`, … | Split or archive if unused |

**Recently under control:** `enhanced_reasoning.py` split into models/patterns modules; `wits_control_center_agent.py` / `web/server.py` remain manageable.

### P3 — Features (when P0–P2 are boring)

| Feature | Value | Notes |
|---------|-------|-------|
| Scheduled background tasks UI | Visibility into jobs | Read-only status first |
| ~~Memory browser in web UI~~ | ✅ Done July 8 — search + recent (filters, Prev/Next pagination) + gated prune | `/api/memory/{search,recent,prune}` |
| Multi-session chat history | Named sessions | Session store + UI |
| Tool usage analytics | Latency / failure rates | Lightweight metrics |
| MCP server health dashboard | State + last error | Extend `/mcp` |
| Local model pull helper | `ollama pull` from UI | Reduces friction |
| Streaming tool progress | Richer SSE for long tools | UX for search/RAG |

Document upload via web already exists in the UI side panel — prefer polish over a net-new feature.

### P4 — Neural web (only if actively using it)

Wired (tools register; `memory_manager.backend: neural` optional) but not the default path. Decide product vs research before investing. Historical ideas: [`neural-web-roadmap.md`](neural-web-roadmap.md).

---

## 3. Remove, archive, or simplify

See also [`clutter-catalog-2026-07.md`](clutter-catalog-2026-07.md), [`tool-registry-reality-2026-07.md`](tool-registry-reality-2026-07.md), [`config-surface-truth-2026-07.md`](config-surface-truth-2026-07.md).

| Item | Verdict |
|------|---------|
| Adaptive LLM stack | ✅ Archived July 8 → `planning/archive/adaptive_llm/core/` (superseded by `model_router.py`; `test_adaptive_llm.py` + `tests/config.yaml` removed). Root `torch.py` shim deleted |
| Synthetic brain / cognitive cluster | ✅ Archived July 8 → `planning/archive/synthetic_brain/` (`memory_handler`, `cognitive_architecture`, stubs, integration + tests) |
| `planning/archive/gui/` | Keep archived; web UI is the client |
| Supabase | Optional; skip in default install docs |
| Synthetic brain doc set | Historical — entry [`docs/SYNTHETIC_BRAIN.md`](../../docs/SYNTHETIC_BRAIN.md) |
| Root `TASK.md` / `PLANNING.md` | Redirects only |
| `DOCKER_INSTRUCTIONS.md` | Parked / background-agent only |

---

## 4. Parked (out of scope unless requirements change)

- Docker packaging as primary deploy path  
- Supabase cloud sync as default  
- PyQt6 desktop GUI revival  
- Git submodules for MCP servers (replaced by on-demand install)  
- Neural web as primary product surface  

---

## 5. Document map

```
planning/roadmap/
├── suggested-features-2026-07.md   ← YOU ARE HERE (what's next)
├── guest-tester-access-2026-07.md  ← guest / family-tester access (safe MVP)
├── revival-2026-07.md              ← what shipped + error triage log
├── clutter-catalog-2026-07.md      ← dead/dormant inventory
├── tool-registry-reality-2026-07.md
├── config-surface-truth-2026-07.md
├── composer-orchestrator-search-quality-2026-07.md  ← historical handoff / tests A–F
├── neural-web-roadmap.md           ← historical (2025)
└── README.md                       ← index
```

The July 8 audit docs are inventories; cleanup waves feed §3. Dual-schedule fix for Docker bg `self_repair` is documented in [`config-surface-truth-2026-07.md`](config-surface-truth-2026-07.md) §7.

---

## 6. Suggested next chunk

1. Promote `fix/revive-2026-07` → `main` when ready  
2. ~~Clutter cleanup wave 1 (orphans from catalog §1)~~ ✅ July 8  
3. Conversation-history-aware intent classification (Claude, WCCA branch)  
4. ~~Incremental lint hygiene until CI can enforce fuller ruff/black~~ ✅  
5. ~~Memory browser~~ ✅ July 8 · multi-session chat still open  
6. ~~Archive synthetic-brain / cognitive cluster (catalog §3a)~~ ✅ July 8  
7. ~~**Guest / family tester access** (safe MVP)~~ ✅ July 8 — `/join`, invite auth, tool/route locks; Phase 3–4 (content policy, owner admin) still open — [`guest-tester-access-2026-07.md`](guest-tester-access-2026-07.md)  

---

*When this doc goes stale: bump the header date, re-run `pytest --collect-only`, and reconcile shipped items against `git log`.*
