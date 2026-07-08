# Config Surface Truth Pass — July 2026

**Created:** July 8, 2026  
**Companion:** [`clutter-catalog-2026-07.md`](clutter-catalog-2026-07.md)  
**Loader:** `config.yaml` → deep-merge `config.local.yaml` → env overrides (`core/config.py:load_config`)

Unknown YAML keys are **silently dropped** (Pydantic default — `extra` not forbid). Nested unknown agent blocks under `agents:` disappear while known leaf fields are kept.

---

## Status legend

| Status | Meaning |
|--------|---------|
| **LIVE** | Loaded and used on a hot path |
| **DORMANT** | Implemented but off by default / gated |
| **EMPTY_IMPL** | Enabled or scheduled but code is `pass` / no-op |
| **DEPRECATED** | Explicitly ignored or superseded |
| **DOCS_ONLY** | Policy / design YAML; not executable settings |
| **YAML_GHOST** | In YAML but **not** on Pydantic → silently ignored |
| **MODEL_ONLY** | On Pydantic model; never (or barely) read |

---

## 0. Override surfaces

| Key / source | Status | Notes |
|--------------|--------|-------|
| `config.yaml` | LIVE | Committed primary |
| `config.local.yaml` | LIVE | Gitignored; written by `/settings` |
| `WITSV3_SUPABASE_URL` / `KEY` | LIVE | Optional memory backend |
| `WITSV3_AUTH_TOKEN_HASH` | LIVE | Auth |
| `TAVILY_API_KEY` / `BRAVE_SEARCH_API_KEY` | LIVE | Search quality |
| `WITSV3_WEB_TOKEN` | LIVE | Web bearer (not on Pydantic) |
| `ANTHROPIC_API_KEY` | LIVE | Escalation only (`core/escalation.py`) |

---

## 1. Biggest lies (read first)

| Look configurable | Reality |
|-------------------|---------|
| Top-level `adaptive_llm:` | **YAML_GHOST** — not on `WitsV3Config`; adaptive provider DEPRECATED → Ollama |
| Nested `agents.control_center_agent.*` etc. | **YAML_GHOST** — only `default_temperature` / `max_iterations` stick |
| `docker:` in config.yaml | **YAML_GHOST** — compose file is truth |
| Ghost ollama `book_writing_model` / `coding_agent_model` / `neural_reasoning_model` / `self_repair_model` | **YAML_GHOST** — routing uses `model_routing` / agent defaults |
| Cron “memory maintenance” / “semantic cache” / “knowledge graph” | **EMPTY_IMPL** — BackgroundAgent methods are `pass` or don’t delete |
| `model_routing` / `escalation` | **LIVE via Pydantic defaults** but **missing from committed config.yaml** |

---

## 2. Hot-path sections (LIVE)

| Section | Verdict | Notes |
|---------|---------|-------|
| `ollama_settings` (core fields) | LIVE | url, models, fallbacks, health — yes |
| `agents.default_temperature` / `max_iterations` | LIVE | |
| `agents.history_window` | LIVE | Model default; settings UI / local yaml |
| `memory_manager` (basic + prune fields) | LIVE | Default backend `basic` |
| `self_repair.*` | LIVE | Except `test_timeout_seconds` → MODEL_ONLY |
| `web_search.*` | LIVE | Keys from env |
| `web_ui.*` | LIVE | |
| `document_rag.*` | LIVE | |
| `tool_system.mcp_*` | LIVE | `enable_mcp_tools` is a weak gate with connect-on-startup |
| `cli.show_thoughts` | LIVE | |
| `security.*` (most) | LIVE | |
| `model_routing.*` | LIVE | Defaults only until local yaml / settings |
| `escalation.*` | LIVE | Defaults; needs Anthropic key |

---

## 3. YAML ghosts (delete from `config.yaml`)

These currently do **nothing** at load:

- Entire `adaptive_llm:` tree  
- Entire `docker:` tree  
- Nested `agents.control_center_agent.*`  
- Nested `agents.orchestrator_agent.*`  
- Nested `agents.coding_agent.*`  
- Nested `agents.book_writing_agent.*`  
- Nested `agents.self_repair_agent.*`  
- `ollama_settings.book_writing_model`  
- `ollama_settings.coding_agent_model`  
- `ollama_settings.neural_reasoning_model`  
- `ollama_settings.self_repair_model`  

---

## 4. MODEL_ONLY (wire or drop)

| Field | Prefer |
|-------|--------|
| `logging_level` | Wire to `basicConfig` (run.py hardcodes INFO) |
| `debug_mode` | Drop unless needed |
| `llm_interface.timeout_seconds` | Drop — Ollama uses `request_timeout` |
| `llm_interface.streaming_enabled` | Drop |
| `cli.show_tool_calls` | Wire to CLI stream filter or drop |
| `ollama_settings.model_timeout` | Drop |
| `personality.profile_id` / `allow_runtime_switching` | Drop or implement switching |
| `supabase.enable_realtime` | Drop |
| `self_repair.test_timeout_seconds` | Wire into apply_code_fix timeout |
| Neural `reasoning_patterns` + 3 cross-domain numeric fields | Drop or wire to `cross_domain_learning` |

---

## 5. DORMANT / gated

| Surface | Gate | Notes |
|---------|------|-------|
| Memory backends `faiss_*` / `neural` / `supabase*` | `memory_manager.backend` | Code exists; default basic |
| Adaptive LLM modules | provider never activates | Superseded by `model_router` |
| `config/wits_core.yaml` | CognitiveArchitecture only | Not on `run.py` path → design sidecar |
| Large personality / ethics sections | Prompt-only slices LIVE | Rest DOCS_ONLY |

---

## 6. Background agent cron honesty

Loaded only when `BackgroundAgent` runs (`config/background_agent.yaml`).

| Task | Status | Recommendation |
|------|--------|----------------|
| `system_monitoring` | LIVE | Keep |
| `self_repair` | LIVE | Keep — but **dedupe** vs `run.py` APScheduler (`0 3 * * *` both) |
| `memory_maintenance` | EMPTY_IMPL | `_maintain_memory` never deletes (`pass`) — wire prune or disable |
| `semantic_cache_optimization` | EMPTY_IMPL | Pure `pass`; adaptive cache dormant — **disable** |
| `knowledge_graph_construction` | EMPTY_IMPL | Pure `pass` — disable or wire |

Silent cron “success” on empty jobs is worse than disabled jobs.

---

## 7. Dual self-repair schedule

**RESOLVED July 8 2026:** `config/background_agent.yaml`'s `self_repair` task is now
`enabled: false` by default, with a comment pointing at `run.py`'s in-process
scheduler as the owner. Only flip it on for a Docker-only deployment that
doesn't also run `run.py`/`run_web.py`.

| Scheduler | Source | Cron | Default |
|-----------|--------|------|---------|
| In-process | `run.py` + `self_repair.daily_schedule_*` | `0 3 * * *` | **on** (the owner) |
| Background process | `config/background_agent.yaml` `tasks.self_repair` | `0 3 * * *` | off — enable only if not also running the main app |

---

## 8. Sidecar YAML map

| File | Role | Status |
|------|------|--------|
| `config/wits_personality.yaml` | System prompt feed | Partial LIVE; most sections DOCS_ONLY |
| `config/ethics_overlay.yaml` | Principle names + override user | Partial LIVE; rest DOCS_ONLY |
| `config/personality_overrides.yaml` | Runtime UI (`/personality`) | LIVE |
| `config/background_agent.yaml` | BackgroundAgent tasks | Mixed (see §6) |
| `config/wits_core.yaml` | Synthetic-brain design | DOCS_ONLY / DORMANT |

---

## 9. Deprecated providers

| Setting | Behavior |
|---------|----------|
| `llm_interface.default_provider: adaptive` | Warning → still Ollama / ReliableOllama. Documented in `planning/archive/adaptive_llm/README.md` |

Live smart routing = `model_routing` (+ `/settings`), not adaptive YAML.

---

## Cleanup waves

### Wave 1 — Truth in `config.yaml` (behavior-neutral)

1. Delete `adaptive_llm:` and `docker:` blocks.  
2. Flatten `agents:` to only fields that exist on `AgentSettings` (+ `history_window`).  
3. Remove ghost ollama `*_model` fields.  
4. Add explicit `model_routing:` and `escalation:` matching Pydantic defaults.  
5. Optionally log (or forbid) unknown keys so ghosts cannot return.

### Wave 2 — Cron honesty

1. Disable `semantic_cache_optimization` and `knowledge_graph_construction`.  
2. Fix or disable `memory_maintenance`.  
3. Pick a single self-repair scheduler (`run.py` **or** background), not both.

### Wave 3 — Wire or drop MODEL_ONLY

See §4 table.

### Wave 4 — Sidecar intent

1. Mark / move `wits_core.yaml` under planning.  
2. Label non-enforced personality/ethics sections as documentation.  
3. Align README so “adaptive LLM” is never described as a live config path.

---

## Scorecard

| Surface | Live share | Biggest lie |
|---------|------------|-------------|
| `WitsV3Config` hot path | High | Nested agent / adaptive / docker look configurable but aren’t |
| `model_routing` / `escalation` | High (defaults) | Invisible in committed yaml |
| Personality / ethics | Partial | Large policy trees aren’t rule engines |
| Background tasks | Mixed | Cron succeeds on empty bodies |
| Adaptive / synthetic brain | None on boot | Survives as yaml ghosts + unused modules |

---

*Next: apply Wave 1 as a docs-neutral PR, then Wave 2 so logs stop lying about maintenance/cache/graph jobs.*
