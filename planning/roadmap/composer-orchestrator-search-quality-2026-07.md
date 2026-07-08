# Composer branch — changes & test guide

**Branch:** `composer/orchestrator-search-quality` (Tier 1 CI/tooling work), then `claude/tier2-tier3-cleanup-2026-07` (Tier 2/3 cleanup, branched from the former)  
**Base:** `fix/revive-2026-07` @ `856af3b` (Claude's July 7 revival work)  
**Last updated:** July 7, 2026  
**Author:** Composer (Cursor agent session) + Claude (Tier 1 CI commit co-authored; Tier 2/3 cleanup)

This file is the handoff doc for Richard. It lists **what Composer changed on this branch**, what was already on the shared revival base, and **what to manually test** before merging anything.

> **Note on concurrent editing:** the Cursor/Composer session and a Claude
> Code session both worked in this checkout at points on July 7. The Cursor
> session committed Tier 1 (CI/tooling) directly to
> `composer/orchestrator-search-quality` as `3080da2`. Claude Code's Tier 2/3
> work stacked on top as `90c7460`, then moved to its own branch
> `claude/tier2-tier3-cleanup-2026-07` to avoid further collisions — that
> branch currently contains everything from both sessions. If merging, treat
> `claude/tier2-tier3-cleanup-2026-07` as the up-to-date tip unless the
> Cursor session has since pushed more commits to
> `composer/orchestrator-search-quality` that need reconciling.

---

## Branch isolation

| Branch | Purpose |
|--------|---------|
| `fix/revive-2026-07` | **Integration branch** — all revival work merges and pushes here before Richard promotes to `main` |
| `composer/orchestrator-search-quality` | Composer experiments + web UI polish + Tier 1 CI/tooling |
| `claude/tier2-tier3-cleanup-2026-07` | Tier 2/3 cleanup, branched from the above |
| `cursor/*` | Cursor agent feature branches — merge into `fix/revive-2026-07` when done |

> **Workflow (July 7, 2026):** Do not push directly to `main` from agent
> sessions. Land work on `fix/revive-2026-07`; Richard merges to `main` after
> manual testing.

Commits on `composer/orchestrator-search-quality` (newest first):

```
3080da2  Add CI, consolidate tooling config, and wire web coverage. (Cursor)
28f3c5c  Enhance MCP tool integration and orchestrator functionality
7660664  Fix embedding truncation config lookup and ship revival orchestrator/MCP work.
ab0936b  Web UI: model routing settings and friendly Ollama-down errors
e2f538f  Improve ReAct search quality: lower reasoning temp, richer observations, merged providers
```

`claude/tier2-tier3-cleanup-2026-07` adds on top:

```
90c7460  Tier 2/3 cleanup: doc truth pass, wire up neural web tools, archive dead code
```

Everything below `856af3b` on this branch is shared revival history (orchestrator JSON, web search rewrite, MCP registry, WCCA routing, etc.) — see `planning/roadmap/revival-2026-07.md` for Claude's log.

---

## Composer commit 1 — ReAct search quality (`e2f538f`)

### Goal

Stop the orchestrator from wandering on search tasks: redundant searches, ignoring Tavily's answer, misusing `document_search` for public web questions, and dumping raw dicts into observations.

### Files changed

| File | What changed |
|------|----------------|
| `agents/base_orchestrator_agent.py` | `REASONING_TEMPERATURE = 0.2` for ReAct JSON calls; `_format_tool_observation()` / `_format_search_observation()` render search results as numbered sources + labeled summary |
| `agents/llm_driven_orchestrator.py` | Tighter reasoning prompt: one search is enough, `web_search` vs `document_search`, trust summary unless sources contradict, answer the specific question asked |
| `tools/web_search_tool.py` | In `auto` mode with both keys: query Tavily + Brave **concurrently**, merge/dedupe results, keep Tavily AI summary + `answer_provider` |
| `data/mcp_tools.json` | Added Gmail MCP server entry (`com.mcparmory/google-gmail`) — config only, not auto-connected |

### Behaviour you should notice

- Current-events questions should do **one** `web_search`, then answer — not 3–5 repeated searches.
- Observations in logs/thinking should show `[1] title / snippet / source:` blocks instead of a Python dict.
- With both `TAVILY_API_KEY` and `BRAVE_SEARCH_API_KEY` in `.env`, search uses provider label like `tavily+brave` and more cross-check sources.

---

## Composer commit 2 — Web UI polish (`ab0936b`)

### Goal

Roadmap items from `revival-2026-07.md` §4 steps 2 & 3:

1. Expose **model routing** on `/settings`
2. **Friendlier Ollama-down** errors in chat (instead of raw connection exceptions after retries)

### Files changed

| File | What changed |
|------|----------------|
| `web/user_errors.py` | **New** — detects Ollama connection failures, returns `{code, message, hint}` |
| `web/server.py` | Settings API: read/write `model_routing`; chat SSE enriches errors; `/api/status` adds `ollama.available` |
| `web/static/settings.html` | "Smart model routing" section (toggle, 3 model dropdowns, trivial max chars) |
| `web/static/app.js` | `addErrorMsg()` for Ollama cards; amber status dot when Ollama down |
| `web/static/style.css` | `.dot.warn`, `.ollama-error` hint styling |
| `tests/web/test_user_errors.py` | **New** — 3 unit tests |
| `tests/web/test_web_server.py` | Settings routing POST/GET + friendly Ollama-down chat test |

### Settings → model routing

Saved to **`config.local.yaml`** (gitignored), applied live — no restart:

```yaml
model_routing:
  enabled: true
  trivial_model: llama3.2:3b
  code_model: qwen2.5-coder:7b
  complex_model: qwen3:8b
  trivial_max_chars: 140
```

### Ollama-down UX

When Ollama isn't running, chat should show:

> **Can't reach Ollama — WITS needs it to think.**  
> Start the Ollama app… `ollama serve`… Windows path hint…

Status dot in the chat header turns **amber** (not red/green).

---

## Inherited on this branch (not Composer — test if merging)

These shipped on the shared revival line before/at `856af3b`. Worth regression-testing when you merge:

- Orchestrator JSON robustness (`format=json`, repair-reparse)
- WCCA routes current-events / document questions to orchestrator
- Multi-provider `web_search` (Tavily → Brave → DuckDuckGo scrape)
- MCP registry discover + install on `/mcp`
- Smart `ModelRouter` in Python (now also exposed in `/settings` via Composer)

---

## Suggested manual test plan

Run from the project root on branch `composer/orchestrator-search-quality`:

```bash
git checkout composer/orchestrator-search-quality
python run_web.py
```

### A. Model routing settings (Composer)

1. Open `http://localhost:8000/settings` (with your token if required).
2. Confirm **Smart model routing** section loads with current models.
3. Toggle routing **off**, set trivial max chars to **80**, Save.
4. Check `config.local.yaml` was written with a `model_routing:` block.
5. Send a short casual message in chat ("hey") — should feel snappier or use different model if routing was on before.
6. Toggle routing back **on** and restore defaults.

### B. Ollama-down handling (Composer)

1. With web UI running, **quit Ollama** (tray app exit).
2. Confirm status dot turns **amber**.
3. Send any chat message.
4. Expect the friendly Ollama card — **not** `Error processing request: Failed to connect…` or a stack trace.
5. Start Ollama again; dot should go green after ~30s (or refresh page).

### C. ReAct search quality (Composer)

**Prerequisite:** `TAVILY_API_KEY` in `.env` (already done per revival doc). Optional: add `BRAVE_SEARCH_API_KEY` to exercise merge path.

1. Ask: **"Who died on June 14, 2026?"** (or any dated current-events question).
2. Watch the thinking/tool chips — expect **one** `web_search`, not a loop.
3. Answer should be a **single direct sentence**, not a long list of unrelated names.
4. Check `logs/witsv3.log` — observation text should show numbered `[1]`, `[2]` sources, not a raw dict.

### D. document_search vs web_search (Composer prompt + inherited routing)

1. With a doc in `documents/` (e.g. an audit PDF), ask about **that file** — should use `document_search`, not web.
2. Ask about a **public figure / news event** — should use `web_search`, **not** `document_search`.

### E. Automated tests (Composer touched)

```bash
pytest tests/web/ -q -o addopts=
pytest tests/agents/test_orchestrator_json.py tests/agents/test_wcca_routing.py -q -o addopts=
pytest tests/tools/test_web_search_tool.py -q -o addopts=
```

Expected: all pass (web suite was **31 passed** on July 7, 2026).

---

### F. Save conversation to file (Composer — new)

1. After a few chat turns, ask: **"Save a log of our conversations as exports/chat_log.txt"**
2. Expect orchestrator delegation (not a direct "I saved it" reply with no tool).
3. In thinking/logs: `read_conversation_history` then `write_file` with `file_path` only (content injected).
4. Confirm `exports/chat_log.txt` exists with USER/ASSISTANT lines.

```bash
pytest tests/agents/test_orchestrator_save_file.py tests/core/test_tool_registry_kwargs.py -q -o addopts=
```

---

## Not done yet (next on roadmap)

Composer did **not** implement these — still open in `revival-2026-07.md`:

| # | Item | Notes |
|---|------|-------|
| 11 | MCP discover follow-ups | OCI/Docker packages, browse-before-install preview |
| — | Merge to `main` | `fix/revive-2026-07` merged into `main` when manual tests A–F look good (Richard's call — held back July 7 pending manual testing) |
| — | Manual ops | Add `ANTHROPIC_API_KEY` if using ask-Claude. Supabase token revoke: N/A, no project exists anymore |
| — | WCCA intent JSON repair | Same repair-reparse pattern as the orchestrator, applied to WCCA intent parsing |

**Recently closed** (were on this list): embedding truncation on large memory stores (shipped `7660664`), `BRAVE_SEARCH_API_KEY` support (shipped), the `set_dependencies(tool_registry=...)` startup crash (shipped `28f3c5c`), and merging `composer/orchestrator-search-quality` + Tier 2/3 cleanup into `fix/revive-2026-07` (shipped `dc7f813`, fast-forward, July 7).

---

## Codebase audit — obviously missing / suggested additions (July 7, 2026)

A read-only pass over the whole repo (not just the search/MCP work) surfaced the
gaps below. None are blockers for the current branch merge, but they're the
obvious next investments. Ordered by leverage-to-effort.

### Tier 1 — cheap, high value (do next)

**SHIPPED July 7, 2026** (Composer session):

1. ✅ **Project-level CI** — `.github/workflows/ci.yml`: tests on Python 3.10/3.11 (`python -m pytest`), coverage artifact upload, critical ruff gate (`F821,E9,E722`).
2. ✅ **De-duplicated tooling config** — single source in `pyproject.toml`; removed `pytest.ini`, `mypy.ini`, `.isort.cfg`, `.coveragerc`, `.flake8`.
3. ✅ **`[tool.ruff]` + `.pre-commit-config.yaml`** — ruff/black/isort + basic hygiene hooks. Install: `pip install pre-commit && pre-commit install`.
4. ✅ **`web/` under coverage** — `--cov=web` in pytest addopts and `[tool.coverage.run] source`.
5. ✅ **`.cursorrules` PLANNING.md pointer** — now points at `planning/architecture/system-architecture.md` + revival roadmap.

Also fixed while wiring CI: `resolve_max_embedding_chars()` ignores non-int MagicMock values (FAISS tests), missing `import os` in `file_tools.py`, four bare `except:` → `except Exception:`.

**Follow-up (not Tier 1):** expand CI lint to full ruff/black/isort once the legacy baseline (~124 black files, ~3.7k ruff findings) is cleaned up — pre-commit enforces on new commits locally until then.

### Tier 2 — documentation truth (low effort, avoids confusion)

**SHIPPED July 7, 2026** (Claude session, branch `claude/tier2-tier3-cleanup-2026-07`):

6. ✅ **`TASK.md` retired** — now a one-line redirect to `revival-2026-07.md`.
7. ✅ **`neural-web-roadmap.md` flagged historical** — header note added.
8. ✅ **Synthetic-brain doc set consolidated** — new `docs/SYNTHETIC_BRAIN.md`
   is canonical; `IMPLEMENTATION_STATUS.md`, `IMPLEMENTATION_SUMMARY.md`,
   `REMAINING_TASKS.md`, `SYNTHETIC_BRAIN_NEXT_STEPS.md`, `SYNTHETIC_BRAIN_PR.md`,
   `README_SYNTHETIC_BRAIN.md` kept for archival detail with a historical
   header pointing back at it.

### Tier 3 — dead / dormant code (decide: wire up, archive, or delete)

**SHIPPED July 7, 2026** (Claude session, branch `claude/tier2-tier3-cleanup-2026-07`):

9. ✅ **Neural web tools wired up.** `EnhancedReasoningTool`, `NeuralWebNLPTool`,
   `NeuralWebVisualizationTool` now use zero-arg constructors + `set_dependencies()`
   (same DI pattern as `document_tools.py`/`web_search_tool.py`), so
   `tool_registry` auto-discovery actually picks them up. Fixed the
   `super().__init__(config)` vs `BaseTool.__init__(name, description)`
   mismatch — it turned out to affect **all three** tools, not just the
   visualization one. They now prefer the live `NeuralWeb` from
   `memory_manager.backend` when the neural backend is active instead of an
   always-empty scratch instance. `tests/tools/test_neural_web_tools.py` added.
10. ✅ **`core/adaptive_llm_interface.py` marked dormant/experimental** in its
    module docstring (not removed — still imports cleanly, `core/adaptive`
    turned out to be a package, not a missing file as first assumed).
11. ✅ **Orphaned `*_fixed.py` / `*_updated.py` files deleted** —
    `core/memory_handler_fixed.py`, `core/memory_handler_updated.py`,
    `core/neural_memory_backend_fixed.py`, `core/tool_registry_fixed.py`,
    `core/cognitive_architecture_updated.py` — plus their two standalone
    tests, which only tested the orphans themselves.
12. ✅ **`gui/` archived** to `planning/archive/gui/` (git history preserved
    via `git mv`).
13. ✅ **Root-level `test_*.py` cleaned up.** 9 still-valid standalone smoke
    scripts (`test_authentication.py`, `test_automatic_pruning.py`,
    `test_enhanced_features.py`, `test_enhanced_validation.py`, `test_fixes.py`,
    `test_memory_pruning.py`, `test_model_reliability.py`,
    `test_streaming_context.py`, `test_witsv3.py`) moved to
    `scripts/manual_tests/`. 4 deleted because they no longer match the
    current implementation and error out under pytest
    (`test_enhanced_reasoning.py`, `test_neural_web_nlp.py`,
    `test_neural_web_visualization.py`, plus empty `test_enhanced_streaming.py`).

Full suite after Tier 2/3: **312 passed, 2 skipped**.

### Tier 4 — structure & hygiene (larger, schedule deliberately)

14. ✅ **500-line rule — Tier 4 targets split** July 7 2026: orchestrator (`orchestrator_tool_helpers.py`),
    WCCA (`wcca_*_mixin.py`), book writing (`book_writing_*`), advanced coding (`coding_*`),
    web server (`web/schemas.py`, `web/routes_mcp.py`, `web/routes_personality.py`).
    Other large files (core/tools, `llm_driven_orchestrator.py`) remain for a future pass.
15. ✅ **`self_repair_agent` pytest coverage** July 7 2026 — `tests/agents/test_self_repair_agent.py`
    (streams thinking/result, LLM passthrough). Smoke tests for `book_writing_agent` and
    `neural_orchestrator_agent` in `tests/agents/test_specialized_agent_smoke.py`.
16. ✅ **MCP submodule footprint** July 7 2026 — removed all git submodules (`Ollama-mcp`, `servers`,
    `supabase-mcp`); `mcp_servers/` is on-demand via `scripts/clone_mcp_servers.py` only
    (`mcp_servers/README.md`, `.gitignore` keeps clone dirs local).
17. ✅ **Supabase `sbp_...` token risk resolved** July 7 2026 — N/A, Richard
    confirmed no Supabase project exists anymore, so the leaked token (still
    in git history) has nothing to grant access to. Given that, worth
    reconsidering whether the vendored `supabase-mcp` submodule (see #16) is
    still needed at all.

---

## Merge recommendation

When manual tests A–F look good:

1. Merge `claude/tier2-tier3-cleanup-2026-07` → `fix/revive-2026-07` (or open a PR) — it's a superset of `composer/orchestrator-search-quality` (includes the Tier 1 CI commit) plus Tier 2/3.
2. Update `revival-2026-07.md` §4 — model-routing settings, Ollama-down UX, save-to-file are shipped on branch.
3. Keep Gmail MCP entry in `data/mcp_tools.json` only if you intend to connect it from `/mcp` — it does not auto-connect.
4. ✅ Tier 4 complete (July 7 2026): 500-line splits for orchestrator/WCCA/book/coding/web
   server; agent pytest coverage; MCP on-demand-only (no git submodules).

---

## Quick file index (Composer-only)

```
agents/base_orchestrator_agent.py   # REASONING_TEMPERATURE, search observation formatting
agents/llm_driven_orchestrator.py # search/document prompt rules
tools/web_search_tool.py          # Tavily+Brave concurrent merge
data/mcp_tools.json               # Gmail MCP config entry
web/user_errors.py                # NEW — friendly error formatting
web/server.py                     # settings + chat error enrichment + ollama status
web/static/settings.html          # model routing UI
web/static/app.js                 # error cards + amber dot
web/static/style.css              # warn dot + error hint styles
tests/web/test_user_errors.py     # NEW
tests/web/test_web_server.py      # routing + ollama-down tests
planning/roadmap/composer-orchestrator-search-quality-2026-07.md  # this file
```
