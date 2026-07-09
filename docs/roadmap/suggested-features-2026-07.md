# WitsV3 — Suggested Features & Roadmap

**Last updated:** July 9, 2026 (Next 10 actions batch)  
**Working branch tip:** `cursor/work` / `claude/work` (merge into `fix/revive-2026-07` or `main`)  
**Test suite:** **677 passed, 2 skipped** (July 9, 2026 — re-run `pytest -q` before claiming green)

This is the **canonical forward roadmap**: what to add, improve, or remove next.  
For **what already shipped**, see [`revival-2026-07.md`](revival-2026-07.md) and the root [`README.md`](../../README.md).  
For **Top 50 Local AI feature mapping** (has vs gap), see [`local-ai-50-gap-analysis-2026-07.md`](local-ai-50-gap-analysis-2026-07.md).  
For audits (clutter / tools / config), see the July 8 catalog docs in this folder.

### Git workflow

Feature work → **`fix/revive-2026-07`** or feature branches → **`main`** after verification.  
Richard promoted **`fix/revive-2026-07` → `main`** July 8, 2026 (Phase 0 complete).

---

## Current snapshot

| Area | Status |
|------|--------|
| Web UI (chat, settings, personality, MCP manager, chat export) | ✅ Shipped |
| Document RAG + `document_search` | ✅ Shipped |
| Runtime path cohesion (`var/user_files`, glossary, migration) | ✅ Shipped July 9 — see [`paths-and-layout.md`](../architecture/paths-and-layout.md) |
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
| Guest / family tester access (full Phase 3–4 + fact editor) | ✅ Shipped July 8 |
| Chat slash-command picker (`/` menu, `GET /api/commands`) | ✅ Shipped July 8 |
| Ollama model pull/status in `/settings` | ✅ Shipped July 9 (Phase 2.1) |
| Phase 2 operator UX (2.2–2.6) | ✅ Shipped July 9 |
| Memory Phase 3a (FAISS default + session memory search) | ✅ Shipped July 9 |
| July revival feature backlog | ✅ **Closed** — Phase 3b+ research next |
| Next 10 actions batch (July 9) | ✅ Shipped — verbose export, patch repair, auto facts, log diagnosis, injection guard, neural research gating, orchestrator split |

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

- ~~Full-file LLM rewrite~~ → SEARCH/REPLACE patch path for files >150 lines (July 9)
- ~~Log diagnosis bare ERROR lines~~ → signature dedupe + traceback-first grouping (July 9)
- Restart is still blunt `Popen` + exit (fine for CLI/dev; not a graceful uvicorn drain)
- `restart_after_fix` defaults **off** so scheduled runs never interrupt a live session

---

## 2. Post–Top-50 roadmap (July 8, 2026)

Mapped against [`FEATURE_IDEAS/Top Local AI System Features.docx`](FEATURE_IDEAS/Top%20Local%20AI%20System%20Features.docx) — detail in [`local-ai-50-gap-analysis-2026-07.md`](local-ai-50-gap-analysis-2026-07.md).  
**Score:** 6 full matches · 26 partial · 17 gaps · 4 N/A (hardware/OS).

**Product stance:** WitsV3 is a **personal LAN orchestration stack**, not a full sovereign AI OS. Do not chase voice, fine-tuning, microVM sandboxes, or Windows ODR unless requirements change.

### Phase 0 — Ship gate ✅ complete (July 8, 2026)

| # | Item | Maps to | Effort | Status |
|---|------|---------|--------|--------|
| 0.1 | Merge `fix/revive-2026-07` → `main` | — | Human | ✅ Done July 8 — fast-forward to `4c676c3` |
| 0.2 | Keep pytest green | — | Ongoing | ✅ **571 passed, 2 skipped** (July 8) |
| 0.3 | Optional `ANTHROPIC_API_KEY` for ask-Claude | #18 | Owner | Open — optional; escalation works when key is set |

### Phase 1 — Trust & daily-use quality ✅ complete (July 8, 2026)

| # | Item | Maps to | Effort | Notes |
|---|------|---------|--------|-------|
| 1.1 | ~~**Conversation-history-aware intent**~~ | #21–22 | Medium | ✅ Done July 8 — WCCA follow-up routing |
| 1.2 | ~~**Guest Phase 3–4 completion**~~ | #27, #43–45 | Medium | ✅ Done July 8 — policy YAML, safesearch, admin UI |
| 1.3 | ~~**Hybrid document search**~~ | #31, #33 | Medium–Large | ✅ Done July 8 — BM25 + vector fuse in `document_tools` |
| 1.4 | ~~**Pre-compaction memory flush**~~ | #22 | Medium | ✅ Done July 8 — flush before history window drops turns |
| 1.5 | ~~**Evidence sufficiency gate**~~ | #33 | Small–Medium | ✅ Done July 8 — synthesis guard blocks weak doc answers |
| 1.6 | ~~**Multi-session chat UI**~~ | #21 | Medium | ✅ Done July 8 — `/api/sessions` + Chats panel |

### Phase 2 — Operator UX & observability ✅ complete (July 9, 2026)

**Pre-2.1 polish (July 8 evening):** Chat slash-command picker — type `/` in the composer for a Cursor-style menu (`web/slash_commands.py`, `GET /api/commands`). Role-filtered: guests see `/help`, `/new`, `/export`, `/chats`, `/tools`; owner adds panels, navigation, `/shutdown`, `/restart`.

| # | Item | Maps to | Effort |
|---|------|---------|--------|
| 2.1 | ~~Ollama model pull/status helper in `/settings`~~ | #8–9 | Small | ✅ Done July 9 |
| 2.2 | ~~MCP server health panel (state, last error)~~ | #14, #20 | Small–Medium | ✅ Done July 9 |
| 2.3 | ~~Tool usage analytics (latency, failures)~~ | #20 | Medium | ✅ Done July 9 |
| 2.4 | ~~Streaming tool progress in SSE~~ | #20 | Medium | ✅ Done July 9 |
| 2.5 | ~~Scheduled background tasks UI (read-only)~~ | — | Small | ✅ Done July 9 |
| 2.6 | ~~Offline / air-gap mode flag~~ | #6 | Small | ✅ Done July 9 |

### Phase 3 — Memory depth **(current — 3a shipped July 9)**

| # | Item | Maps to | Effort | Verdict |
|---|------|---------|--------|---------|
| 3a | **Pragmatic:** FAISS default + session memory search in orchestrator | #21, #25, #29 | Medium | ✅ Done July 9 — see [`memory.md`](../architecture/memory.md) |
| 3b | **Research:** Wire KG into document RAG (light GraphRAG) | #28 | Large | **Deferred** — `doc_qa` playbook covers daily use; revisit if compliance docs become primary |
| 3c | **Research:** Neural web as product surface | #21, #28 | Large | See [`neural-web-roadmap.md`](neural-web-roadmap.md) — decide before investing |
| 3d | SKILL.md-style orchestrator playbooks | #16 | Medium | ✅ Done July 9 — `save_conversation`, `codebase_tour`, `doc_qa` |

### Phase 4 — Parked (explicitly out of scope)

| Area | Features | Why |
|------|----------|-----|
| Voice / ambient | #34–37 | No STT/TTS pipeline; huge scope |
| Smart home | #38 | Home Assistant integration |
| Fine-tuning | #46–50 | Training infra ≠ orchestration product |
| Enterprise sandbox | #40–41 | microVM / SEB overkill for personal LAN |
| Windows ODR | #15 | OS-level; MCP registry covers discover |
| Multimodal chat | #11 | Wait for stable local vision models + UX |
| Digital twin / persona emulation | #39 | Sensitive; guest profiles + personality enough for now |

---

## 2a. Legacy priority table (July revival — mostly done)

### P0 — Verify & promote

1. ~~Complete manual tests A–F~~ ✅  
2. ~~Keep the suite green~~ ✅ 571 passed, 2 skipped (July 8, 2026)  
3. ~~Merge `fix/revive-2026-07` → `main`~~ ✅ Done July 8, 2026  

### P1 — High leverage (historical)

| Item | Why | Effort | Status |
|------|-----|--------|--------|
| Orchestrator synthesis guard | Ground answers in tool observations | Medium | ✅ Done |
| Friendlier Ollama-down (CLI) | Match web UX | Small | ✅ Done |
| Conversation export UX | One-click export in web UI | Small–Medium | ✅ Done |
| Conversation-history-aware intent | Short replies after a clarifying question misclassified as casual chat | Medium | ✅ Done July 8 (Phase 1.1) |
| **Guest / family tester access** | LAN phone testing without owner token; device identity + safe tool subset | Medium–Large | ✅ Done July 8 — Phase 3–4 complete (Phase 1.2) |
| Expand CI lint | Full ruff/black blocked by legacy noise | Medium (incremental) | ✅ Done July 8 |
| Docs / README modernity | Install + honest status | Medium | ✅ Done July 8 |
| Clutter cleanup wave 1 | Delete orphans + relocate scripts | Small–Medium | ✅ Done July 8 |

### P2 — Structure & hygiene (second-pass splits)

Files still over ~500 lines (excluding archived GUI). Re-measure before splitting — several already dropped after July work:

| Approx. lines | File | Suggestion |
|-------------:|------|------------|
| ~900 | `agents/orchestrator_tool_helpers.py` | ✅ Split July 9 — `orchestrator_preflight.py`, `orchestrator_codebase.py` |
| 772 | `tools/neural_web_visualization.py` | Split render/export from graph queries |
| 704 | `core/tool_composition.py` | Composition engine vs registry glue |
| 632 | `agents/self_repair_handlers.py` | **Prefer delete** — orphan after rewrite (see clutter catalog) |
| 607 | `core/memory_manager.py` | Backend factory vs segment CRUD |
| ~500+ | `core/neural_memory_backend.py`, `knowledge_graph.py`, `response_parser.py`, … | Split or archive if unused |

**Recently under control:** `enhanced_reasoning.py` split into models/patterns modules; `wits_control_center_agent.py` / `web/server.py` remain manageable.

### P3 — Features (superseded by §2 Phases 1–2)

| Feature | Value | Status |
|---------|-------|--------|
| ~~Memory browser in web UI~~ | Search + prune | ✅ Done July 8 |
| Multi-session chat history | Named sessions | ✅ Done July 8 (Phase 1.6) |
| Tool usage analytics | Latency / failure rates | **→ Phase 2.3** |
| MCP server health dashboard | State + last error | **→ Phase 2.2** |
| Local model pull helper | `ollama pull` from UI | **→ Phase 2.1** |
| Streaming tool progress | Richer SSE for long tools | **→ Phase 2.4** |
| Scheduled background tasks UI | Visibility into jobs | **→ Phase 2.5** |
| Hybrid RAG | Lexical + semantic | ✅ Done July 8 (Phase 1.3) |

Document upload via web already exists in the UI side panel — prefer polish over a net-new feature.

### P4 — Neural web (only if actively using it)

Wired (tools register; `memory_manager.backend: neural` optional) but not the default path. Decide product vs research before investing. Historical ideas: [`neural-web-roadmap.md`](neural-web-roadmap.md).

---

## 3. Remove, archive, or simplify

See also [`clutter-catalog-2026-07.md`](clutter-catalog-2026-07.md), [`tool-registry-reality-2026-07.md`](tool-registry-reality-2026-07.md), [`config-surface-truth-2026-07.md`](config-surface-truth-2026-07.md).

| Item | Verdict |
|------|---------|
| Adaptive LLM stack | ✅ Archived July 8; **pruned Phase 2b** — stub `docs/archive/adaptive_llm/README.md` + tag `archive-pre-prune-2b-2026-07` (superseded by `model_router.py`) |
| Synthetic brain / cognitive cluster | ✅ Archived July 8; **pruned Phase 2b** — stub README + tag (historical docs under `historical-docs/`) |
| `docs/archive/gui/` | ✅ Pruned Phase 2a — stub + tag `archive-pre-prune-2026-07`; web UI is the client |
| `docs/archive/sphinx/` | ✅ Pruned Phase 2b — stub README; Markdown docs are canonical |
| Supabase | Optional; skip in default install docs |
| Synthetic brain doc set | Historical — entry [`SYNTHETIC_BRAIN.md`](../archive/historical-docs/SYNTHETIC_BRAIN.md) |
| Root `TASK.md` / `PLANNING.md` | Redirects only |
| `DOCKER_INSTRUCTIONS.md` | Parked / background-agent only |

### Repo footprint consolidation (July 2026)

| Phase | Status | Notes |
|-------|--------|-------|
| **0 — Safe deletes** | ✅ Done | Removed `models/` (dummy adaptive-LLM safetensors), `WitsV3/` stray package, `enhanced_config_with_fallback.txt`; moved `debug_init.py` + `fix_neural_web.py` → `scripts/` |
| **1 — Doc consolidation** | ✅ Done | Merged `planning/` into `docs/`; `planning/` is redirect stub |
| **2a — GUI archive prune** | ✅ Done | Tag `archive-pre-prune-2026-07`; removed 61 files under `docs/archive/gui/` (stub README remains) |
| **2b — Further archive prune** | ✅ Done | Tag `archive-pre-prune-2b-2026-07`; removed 27 files (`adaptive_llm/core/`, `sphinx/`, synthetic_brain code); stub READMEs remain |
| **3 — Runtime `var/` layout** | ✅ Done | `core/runtime_paths.py`; legacy dirs auto-migrate on startup |
| **4 — Root surface cleanup** | ✅ Done | Instruction-file redirects (`.cursorrules`, `TASK.md`, `PLANNING.md`, `planning/`) — July 9 |

**July 9:** Config Wave 1 — removed YAML ghosts (`adaptive_llm`, `docker`, unwired nested `agents.*`, ghost ollama `*_model` keys); added explicit `escalation:` block.

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
docs/roadmap/
├── suggested-features-2026-07.md   ← YOU ARE HERE (what's next)
├── local-ai-50-gap-analysis-2026-07.md  ← Top 50 feature map (has / partial / gap)
├── FEATURE_IDEAS/Top Local AI System Features.docx  ← source reference doc
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

## 5. Post–Phase 2 backlog

| Item | Notes | Status |
|------|-------|--------|
| ~~**Prompt-injection classifier**~~ | Owner-path pattern guard on file-write tools | ✅ July 9 |
| ~~**Automatic fact extraction**~~ | Heuristic owner-path promotion + WCCA remember | ✅ July 9 |
| ~~**Verbose export with tool traces**~~ | `/export verbose` + `POST /api/export?verbose` | ✅ July 9 |
| ~~**Diff/patch self-repair**~~ | SEARCH/REPLACE via `apply_verified_patch` for large files | ✅ July 9 |
| **PII redaction** | Guest/owner export and memory hygiene | ✅ July 9 (competitive landscape batch) |

---

## 7. Competitive landscape track (July 2026 PDF)

Source: `WitsV3_Competitive_Landscape.pdf` — phased incorporation vs AnythingLLM, Open WebUI, OpenHands, Letta, etc.

| Item | Notes | Status |
|------|-------|--------|
| **Filesystem read allowlist + D:\\Downloads** | `security.filesystem_read_roots`, `core/filesystem_policy.py` | ✅ July 9 |
| **Tiered memory (core block + archival)** | `core/core_memory.py`, promotion tools | ✅ July 9 |
| **Additive RBAC roles** | `config/guest_policy.yaml` roles + `guest_access.is_tool_allowed` | ✅ July 9 |
| **PII redaction** | `core/pii_redaction.py` — memory store + guest responses | ✅ July 9 |
| **Sandbox runner (optional)** | `core/sandbox_runner.py`, `Dockerfile.sandbox`, `sandbox_mode: off` default | ✅ July 9 |
| **Agent hand-off graph** | `config/agent_graph.yaml`, `agents/agent_handoff.py` | ✅ July 9 |
| **Telegram / desktop wrapper** | Deferred — revisit after family rollout | Open |

---

## 6. Suggested next chunk

1. ~~**Phase 0:** Promote `fix/revive-2026-07` → `main`~~ ✅ Done July 8, 2026  
2. ~~**Phase 1:** Trust & daily-use quality (all six items)~~ ✅ Done July 8, 2026  
3. ~~**Phase 2:** Operator UX (2.1–2.6)~~ ✅ Done July 9, 2026  
4. ~~**Phase 3a:** FAISS default + session memory search~~ ✅ Done July 9, 2026  
5. ~~**Next 10 actions (July 9):** verbose export, patch repair, auto facts, log diagnosis, injection guard, neural research gating, orchestrator split, doc sync~~ ✅ Done July 9, 2026  
6. ~~**Pipeline forward batch (July 9):** codebase playbook routing, perf smoke tier, native A/B tooling, multiturn smoke, embedding cap, ADR guest table~~ ✅ Done July 9, 2026  
7. ~~**PII redaction** on export/memory (§5)~~ ✅ July 9  
8. **Competitive landscape deferred:** Telegram / native desktop wrapper (§7)  
9. **Optional `ANTHROPIC_API_KEY`** — owner gate G2  
10. **Phase 3b GraphRAG** — deferred unless compliance-style docs become primary (playbook `doc_qa` covers daily use)

**July 9 afternoon — chosen next sprint:** **Polish & stability** (memory hardening, fast smoke `--isolated`, Phase 4 path cohesion, clutter Wave A). GraphRAG / Telegram / neural surface remain deferred.

**Shipped (July 9):** System audit remediation + Next 10 batch + pipeline forward batch — see [`revival-2026-07.md`](revival-2026-07.md) §5.

---

*When this doc goes stale: bump the header date, re-run `pytest --collect-only`, and reconcile shipped items against `git log`.*
