# WitsV3 Clutter Catalog — Unfinished, Obsolete & Dead Code

**Created:** July 8, 2026  
**Branch tip:** `cursor/iron-delta-w0r5` / `fix/revive-2026-07`  
**Scope:** Production Python under `agents/`, `core/`, `tools/`, and repo-root scripts.  
Excluded from deep inventory: vendored `mcp_servers/`, `coverage_html/`, `__pycache__/`.

**Hot path (alive today):** `run.py` / `run_web.py` → WCCA → `LLMDrivenOrchestrator` + Ollama + `memory_manager.backend: basic` + registered tools. Everything else is optional, dormant, or dead.

---

## How to read this

| Category | Meaning |
|----------|---------|
| **ORPHAN** | No hot-path importers; safe delete/archive |
| **STUB** | Explicit placeholders / `pass` / fake implementations |
| **DORMANT** | Exists and may be wired, but off by default config |
| **EMPTY_IMPL** | Constructed or scheduled, but methods do little / never finish |
| **DEAD_ALT** | Superseded by a newer implementation |
| **OBSOLETE_DOC** | Historical docs that confuse current status |

**Recommendations:** `delete` · `archive` · `wire-up` · `keep`

≈ **5–7k lines** of clear orphan/dormant surface (excluding archived GUI and adaptive stack nested helpers).

---

## 1. Highest-impact cleanup (do first)

Safe, high confusion / line-count wins:

| # | Path | Lines | Category | Recommendation |
|---|------|------:|----------|----------------|
| 1 | `agents/self_repair_handlers.py` | 614 | ORPHAN | **delete** — rewritten agent does not import it |
| 2 | `agents/self_repair_utils.py` | 394 | ORPHAN | **delete** |
| 3 | `agents/self_repair_models.py` | 44 | ORPHAN | **delete** |
| 4 | `run_backup.py` | 287 | DEAD_ALT | **delete** — stale fork of `run.py` |
| 5 | `run_clean.py` | 292 | DEAD_ALT | **delete** — same |
| 6 | `agents/conversation_handler.py` | 190 | ORPHAN | **delete** — superseded by WCCA intent/routing mixins |
| … | Sphinx under `docs/*.rst` | — | OBSOLETE_DOC | **keep stubbed** — `index`/`installation`/`quickstart` now point at root README (July 8 docs pass); remaining rst leaves still historical |
| 7 | `tools/intent_analysis_tool.py` | 118 | DEAD_ALT | **delete** (or stop registering) — blocked on orchestrator; real intent is `wcca_intent_mixin.py` |
| 8 | `create_dummy_model.py` | 40 | STUB | **delete** — fake `.safetensors` for dead adaptive experts |
| 9 | `update_config.py` | 51 | ORPHAN | **delete** — one-shot “force ollama provider” migrator, already done |
| 10 | `embedding_dimension_fix.py` | 64 | ORPHAN | **delete** — one-shot patcher |

**Subtotal ≈ 2,100 lines** of clear deletes with almost no risk to the hot path.

---

## 2. Agents

| Path | Lines | Category | Evidence | Recommendation |
|------|------:|----------|----------|----------------|
| `self_repair_{handlers,utils,models}.py` | 1052 | ORPHAN | Only import each other; live agent is `self_repair_agent.py` + `tools/self_repair_tools.py` + `core/safe_code_editor.py` | **delete** |
| `conversation_handler.py` | 190 | ORPHAN | Only its own `__main__`; overlaps WCCA casual/intent | **delete** |
| `neural_orchestrator_agent.py` | 477 | DORMANT | `run.py` loads it only when `memory_manager.backend == "neural"` (default: `basic`) | **keep** until neural is productized |
| `enhanced_book_agent_with_fallback.py` | 436 | LIVE | `run.py` imports this (not plain `BookWritingAgent`) | **keep** |
| `book_writing_agent.py` + handlers/models/helpers | ~split | LIVE lib | Parent of enhanced agent | **keep** |
| `advanced_coding_agent.py` + coding_* | LIVE | Verify-before-commit pipeline (July 8) | **keep** |
| `background_agent.py` | 247 | EMPTY_IMPL (partial) | See §6 | wire-up or strip jobs |
| `base_*` / WCCA / orchestrator mixins | LIVE | Hot path | **keep** |

---

## 3. Core modules

### 3a. Synthetic brain / cognitive cluster (orphan + stub)

| Path | Lines | Category | Evidence | Recommendation |
|------|------:|----------|----------|----------------|
| `synthetic_brain_stubs.py` | 94 | STUB | Stub LLM/KG/WM; only used by integration | ✅ **archived** July 8 → `docs/archive/synthetic_brain/` |
| `synthetic_brain_integration.py` | 87 | ORPHAN/STUB | Fallback wrappers; no production importer | ✅ **archived** |
| `cognitive_architecture.py` | 367 | ORPHAN | Only tests + `__main__`; wrong KG constructor vs real API | ✅ **archived** |
| `memory_handler.py` | 320 | STUB/ORPHAN | Procedural memory “not yet implemented”; wrong KG/WM APIs; not on `MemoryManager` path | ✅ **archived** |

### 3b. Adaptive LLM cluster (explicitly dormant)

| Path | Lines | Category | Evidence | Recommendation |
|------|------:|----------|----------|----------------|
| `adaptive_llm_interface.py` | 302 | DORMANT | Header + `get_llm_interface()` **ignores** `adaptive` → Ollama | **archive** (docs already in `docs/archive/adaptive_llm/`) |
| `adaptive_llm_config.py` | ~163 | DORMANT | Only adaptive stack | **archive** |
| `adaptive/*` (tokenizer, tracker, response_generator) | ~500 | DORMANT | Placeholder tokenize/decode | **archive** |
| `complexity_analyzer.py` | 265 | DORMANT | Adaptive + `tests/test_adaptive_llm.py` | **archive** |
| `dynamic_module_loader.py` | 448 | DORMANT | Quantized-load placeholder | **archive** |
| `semantic_cache.py` | 421 | DORMANT | Adaptive only | **archive** |

**Adaptive subtotal ≈ 2.1k lines** — already superseded by `model_router.py`.

### 3c. Neural / knowledge (dormant unless backend = neural)

| Path | Lines | Category | Evidence | Recommendation |
|------|------:|----------|----------|----------------|
| `working_memory.py` | 466 | DORMANT | Tests + memory_handler; not in default MemoryManager | **keep** if committed to neural; else archive with cognitive |
| `knowledge_graph.py` | 531 | DORMANT | Same; API mismatch vs older consumers (`add_concept` etc.) | decide product intent |
| `cross_domain_learning.py` | 406 | DORMANT | Used by neural orchestrator / enhanced_reasoning | **keep** with neural stack |
| `neural_web_core.py` / `neural_memory_backend.py` | LIVE optional | Backend gate | **keep** |
| `supabase_backend.py` | 432 | DORMANT optional | Enabled only via `backend: supabase*`; no active project | **keep** as opt-in |

### 3d. Constructed but shallow

| Path | Lines | Category | Evidence | Recommendation |
|------|------:|----------|----------|----------------|
| `tool_composition.py` | 680 | EMPTY_IMPL | WCCA builds `IntelligentToolComposer`, then delegates to orchestrator (“not fully implementing workflow execution”) | **wire-up** or **archive** until ready |
| `concrete_meta_reasoning.py` + `meta_reasoning.py` | 513+184 | LIVE/shallow | Used by WCCA enhanced path; still often falls through to orchestrator | **keep**; finish or simplify |
| `content_fallback_system.py` | 570 | LIVE (narrow) | Only via `EnhancedBookWritingAgent` | **keep** while enhanced book stays entry |
| `mcp_adapter.py` + `enhanced_mcp_adapter.py` | both LIVE | Inheritance; production uses Enhanced | **keep** |

---

## 4. Tools

| Path | Lines | Category | Evidence | Recommendation |
|------|------:|----------|----------|----------------|
| `intent_analysis_tool.py` | 118 | DEAD_ALT | Auto-registered then **blocked** (`ORCHESTRATOR_BLOCKED_TOOLS`); WCCA mixin owns intent | **delete** |
| `json_tool.py` (`json_manipulate`) | 349 | DEAD_ALT / blocked | Same block list; still has tests | **delete** *or* deliberately unblock |
| `ollama_probe.py` | 113 | ORPHAN script | Not a `BaseTool`; misfiled under `tools/` | **move** → `scripts/` |
| `network_control_tool.py` | 175 | LIVE niche | Auth-gated; used by setup/docs | **keep** |
| Neural / document / web / MCP / self-repair / ask_claude / file / math / python | — | LIVE | Auto-discovered & used | **keep** |

---

## 5. Root scripts & clutter files

| Path | Lines | Category | Recommendation |
|------|------:|----------|----------------|
| `run_backup.py` | 287 | DEAD_ALT | **delete** |
| `run_clean.py` | 292 | DEAD_ALT | **delete** |
| `create_dummy_model.py` | 40 | STUB | **delete** |
| `update_config.py` | 51 | ORPHAN | **delete** |
| `embedding_dimension_fix.py` | 64 | ORPHAN | **delete** |
| `analyze_memory.py` | 28 | ORPHAN script | **move** → `scripts/` |
| `torch.py` | 35 | STUB | Fake torch for tests | ✅ **deleted** July 8 (adaptive stack archived) |
| `llm_diagnostic_basic.py` | 86 | utility | **move** → `scripts/` (used by `run_test.py`) |
| `run.py` / `run_web.py` / `run_background_agent.py` / `run_test.py` | — | LIVE | **keep** |
| `setup_auth.py` / `setup_dependencies.py` / `install.py` | — | LIVE | **keep** |

---

## 6. Empty scheduled jobs (`BackgroundAgent`)

| Method | Status | Recommendation |
|--------|--------|----------------|
| `_maintain_memory` | Removes nothing — body is `pass` | **wire-up** (call real prune) or remove task |
| `_optimize_semantic_cache` | Pure `pass` | **delete** job (semantic cache is adaptive/dormant) |
| `_build_knowledge_graph` | Pure `pass` | **wire-up** or remove task from `config/background_agent.yaml` |
| `_monitor_system` | Real | **keep** |
| `_run_self_repair` | Real (July 8) | **keep** |

Silent no-ops in cron jobs are worse than missing jobs — they look healthy in logs without doing work.

---

## 7. Already archived (don’t reinvent)

| Location | Contents |
|----------|----------|
| `docs/archive/gui/` | Pruned July 8 2026 — stub + tag `archive-pre-prune-2026-07` (web UI is the client) |
| `docs/archive/adaptive_llm/README.md` | Deprecation note (July 2026) |
| `docs/archive/originals/` | Pre-revival docs & roadmaps |
| `scripts/manual_tests/` | Old root smoke scripts moved in Tier 3 |

---

## 8. Obsolete / confusing documentation

| Path | Issue | Recommendation |
|------|-------|----------------|
| `docs/IMPLEMENTATION_STATUS.md` / `*_NEXT_STEPS.md` / `SYNTHETIC_BRAIN_*.md` | Still describe stubs as active 2025 work | Historical headers already partly added — finish truth pass |
| `FILE_STRUCTURE.md` | Lists dormant modules as first-class | Update after deletes |
| `docs/tasks/task-management.md` | Marked superseded | leave |
| `docs/roadmap/neural-web-roadmap.md` | 2025 design | historical — leave |

Canonical: `suggested-features-2026-07.md` (forward) + `revival-2026-07.md` (shipped log).

---

## 9. What is *not* clutter (avoid deleting)

- `EnhancedBookWritingAgent` + `content_fallback_system` — **chosen** book entry in `run.py`
- `NeuralOrchestratorAgent` — optional feature gate
- `supabase_backend` — opt-in backend
- `enhanced_mcp_adapter` / base MCP — both live
- Self-repair agent + `safe_code_editor` + `self_repair_tools` — July 8 rewrite (**the old triad is clutter; this is not**)
- Coding agent verify pipeline — live
- Web UI, RAG, model router, synthesis guard, export API — live

---

## 10. Suggested cleanup waves

### Wave A — Safe deletes (≈2.1k lines) — ✅ DONE July 8 2026
1. ✅ Delete orphan self-repair triad (`handlers` / `utils` / `models`).
2. ✅ Delete `run_backup.py`, `run_clean.py`, `create_dummy_model.py`, `update_config.py`, `embedding_dimension_fix.py`.
3. ✅ Delete `conversation_handler.py` and `intent_analysis_tool.py` (capability strings updated).
4. Decide: delete or unblock `json_tool.py`. *(still open — kept, has passing tests)*

### Wave B — Relocate scripts — ✅ DONE July 8 2026
1. ✅ Move `ollama_probe.py`, `analyze_memory.py`, `llm_diagnostic_basic.py` → `scripts/` (`run_test.py` string updated).
2. ✅ Delete root `torch.py` (unused after adaptive archive).

### Wave C — Archive dormant clusters
1. ✅ **DONE July 8 2026** — Adaptive LLM + `core/adaptive/` → `docs/archive/adaptive_llm/`; `tests/test_adaptive_llm.py` + `tests/config.yaml` removed; **code pruned Phase 2b** (tag `archive-pre-prune-2b-2026-07`).
2. ✅ **DONE July 8 2026** — Synthetic brain + cognitive + `memory_handler` → `docs/archive/synthetic_brain/`; **code pruned Phase 2b**.
3. ✅ **DONE July 8 2026 (Phase 2b)** — Unused Sphinx stubs → pruned; stub README under `docs/archive/sphinx/`.

### Wave D — Product decisions (don’t delete until decided)
1. Neural stack (`working_memory`, `knowledge_graph`, `cross_domain_learning`, neural orchestrator): **ship as default** or **mark research-only** in README.
2. `tool_composition.py`: finish workflow execution or archive.
3. BackgroundAgent empty jobs: wire prune/graph or remove from YAML schedules.

---

## 11. Quick reference by recommendation

### Delete soon
```
agents/self_repair_handlers.py
agents/self_repair_utils.py
agents/self_repair_models.py
agents/conversation_handler.py
tools/intent_analysis_tool.py
run_backup.py
run_clean.py
create_dummy_model.py
update_config.py
embedding_dimension_fix.py
```
Plus optional: `tools/json_tool.py` (if staying blocked forever).

### Archive / relocate
```
core/synthetic_brain_stubs.py
core/synthetic_brain_integration.py
core/cognitive_architecture.py
core/memory_handler.py
core/adaptive_llm_* + core/adaptive/
core/complexity_analyzer.py
core/dynamic_module_loader.py
core/semantic_cache.py
tools/ollama_probe.py → scripts/
analyze_memory.py → scripts/
llm_diagnostic_basic.py → scripts/
torch.py → tests/ or delete
```

### Keep (with caveat)
```
agents/neural_orchestrator_agent.py          # neural backend only
core/content_fallback_system.py              # enhanced book agent
core/tool_composition.py                     # shallow today
core/working_memory.py / knowledge_graph.py  # neural research
core/supabase_backend.py                     # opt-in
tools/network_control_tool.py                # auth feature
```

---

*Next update: re-run this catalog after Wave A deletes; reconcile line counts with `pytest --collect-only` and update `suggested-features-2026-07.md` remove/archive section.*
