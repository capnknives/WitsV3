# Cursor Phase 1 handoff — July 8, 2026

**Branch:** `cursor/work` (Cursor agent)  
**Base at start of Phase 1:** `99e158e` (guest profile smoke script)  
**Parallel work:** `claude/work` in `WitsV3-claude` — may touch overlapping files for different goals.

This note is for **merge coordination**, not the product backlog. Forward priorities remain in [`suggested-features-2026-07.md`](suggested-features-2026-07.md) (Phase 2 next).

---

## What landed (Phase 1 — all six items)

| # | Feature | Primary files |
|---|---------|---------------|
| 1.1 | Conversation-history-aware intent | `agents/wcca_routing_mixin.py`, `agents/wcca_intent_mixin.py`, `agents/wits_control_center_agent.py`, `tests/agents/test_wcca_routing.py` |
| 1.2 | Guest Phase 3–4 (policy + prompts) | `config/guest_policy.yaml`, `core/guest_policy_loader.py`, `core/content_policy.py`, `tools/web_search_tool.py`, `web/guest_auth.py`, `core/guest_access.py` |
| 1.3 | Hybrid document search (BM25 + vector) | `core/document_hybrid_search.py`, `tools/document_tools.py` |
| 1.4 | Pre-compaction memory flush | `core/conversation_compaction.py`, `run.py`, `web/server.py` (chat handler), `agents/base_orchestrator_agent.py`, `agents/llm_driven_orchestrator.py` |
| 1.5 | Evidence sufficiency gate | `agents/orchestrator_tool_helpers.py`, `tests/agents/test_orchestrator_synthesis_guard.py` |
| 1.6 | Multi-session chat UI | `web/server.py` (`/api/sessions*`), `web/static/{index.html,app.js,style.css}`, `core/schemas.py` (`ConversationHistory.title`), `web/schemas.py` |

**Tests added/extended:** 138 Phase-1-related tests green (WCCA, guest, RAG, compaction, synthesis guard, web sessions).

**Docs updated:** `suggested-features-2026-07.md`, `revival-2026-07.md`, `local-ai-50-gap-analysis-2026-07.md`.

---

## High-conflict files (check these first on merge)

These files are **hot** on both guest/orchestrator and web paths:

| File | Cursor changes | Likely Claude overlap |
|------|----------------|----------------------|
| `agents/wits_control_center_agent.py` | Follow-up routing override; guest rules in casual-chat prompt | Guest routing, personalization, intent |
| `agents/llm_driven_orchestrator.py` | Guest ReAct rules; `flush_context` in prompt | Orchestrator prompt / tool loop |
| `agents/orchestrator_tool_helpers.py` | Evidence gate; `user_role` → `web_search` | Synthesis guard, tool arg injection |
| `web/server.py` | Session API, auto-title, compaction flush in chat, `getattr` for `llm_interface` | Guest chat, audit, auth middleware |
| `web/static/app.js` | Chats tab, session switch/restore, new-chat via API | Guest chrome, auth bootstrap |
| `core/content_policy.py` | YAML-backed blocklists; `guest_system_prompt_slice()` | Age-band / policy tiers |
| `core/guest_access.py` | `/api/sessions` in allowed prefixes | Guest allowlists |
| `tools/web_search_tool.py` | Strict safesearch when `user_role=guest` | Search providers / guest tools |

**Lower conflict but touched:** `core/schemas.py`, `run.py`, `README.md`, `.env.example`, guest test files.

**New files (safe adds):** `config/guest_policy.yaml`, `core/guest_policy_loader.py`, `core/document_hybrid_search.py`, `core/conversation_compaction.py`, compaction/hybrid tests.

---

## Behavioral notes (for manual QA)

1. **Follow-up intent:** Short replies after a clarifying assistant question (e.g. “yes”, “summarize it”) should hit the orchestrator, not casual chat. “Thanks” after a generic greeting should stay casual.
2. **Guest sessions:** Storage key `guest:{guest_id}:{client_session_id}`; list/switch/rename only sees own chats. `/api/sessions` added to guest allowlist in `web/guest_auth.py`.
3. **Hybrid search:** `document_search` fuses BM25 + vector scores; results expose `lexical_score` / `vector_score`.
4. **Compaction flush:** Runs before owner chat turns when history exceeds window+buffer; skipped for guests (`skip_global_store=True`).
5. **Evidence gate:** Confident answers blocked when `document_search` scores are weak; auto-synthesize returns honest “not in your docs”.
6. **Multi-session UI:** In-memory only (survives refresh, not server restart). Chats tab is default in side panel. `＋` creates via `POST /api/sessions`.

---

## Suggested merge order

1. Merge or rebase `cursor/work` into integration branch (`fix/revive-2026-07` or `claude/work`) **after** pulling latest Claude changes.
2. Resolve hot files using **behavioral tests** as tie-breakers:
   - `pytest tests/agents/test_wcca_routing.py`
   - `pytest tests/web/test_web_server.py tests/web/test_guest_access.py`
   - `pytest tests/agents/test_orchestrator_synthesis_guard.py`
   - `pytest tests/core/test_conversation_compaction.py tests/core/test_document_hybrid_search.py`
3. Fast-forward personal runtime `WitsV3` from integration branch; hard-refresh static assets.

---

## What Cursor did *not* do

- No Phase 2 (Ollama pull UI, MCP health, tool analytics, etc.)
- No persistent session store (disk/DB) — still `system.session_histories` in process memory
- No commits on `claude/work` or `fix/revive-2026-07` from this session
- Did not merge into `WitsV3` runtime worktree (owner step per `WORKTREES.md`)

---

## Commit series on `cursor/work` (this push)

Six feature commits + docs commit, in dependency-friendly order:

1. `feat(wcca): conversation-history-aware follow-up intent routing`
2. `feat(guest): externalize content policy and tighten guest search`
3. `feat(rag): hybrid BM25+vector document search`
4. `feat(memory): pre-compaction conversation flush before agent runs`
5. `feat(orchestrator): evidence sufficiency gate for weak document retrieval`
6. `feat(web): multi-session chat API and Chats panel UI`
7. `docs(roadmap): Phase 1 shipped log, gap analysis, and handoff notes`
