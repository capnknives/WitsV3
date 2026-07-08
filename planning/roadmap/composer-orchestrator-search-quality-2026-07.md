# Composer branch — changes & test guide

**Branch:** `composer/orchestrator-search-quality`  
**Base:** `fix/revive-2026-07` @ `856af3b` (Claude's July 7 revival work)  
**Last updated:** July 7, 2026  
**Author:** Composer (Cursor agent session)

This file is the handoff doc for Richard. It lists **what Composer changed on this branch**, what was already on the shared revival base, and **what to manually test** before merging anything.

---

## Branch isolation

| Branch | Purpose |
|--------|---------|
| `fix/revive-2026-07` | Claude's ongoing revival work — keep clean |
| `composer/orchestrator-search-quality` | Composer experiments + web UI polish — **work here** |

Composer commits (newest first):

```
ab0936b  Web UI: model routing settings and friendly Ollama-down errors
e2f538f  Improve ReAct search quality: lower reasoning temp, richer observations, merged providers
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
| — | Merge composer branch | PR into `fix/revive-2026-07` or `main` when tests A–F look good |
| — | Manual ops | Revoke leaked Supabase token; add `ANTHROPIC_API_KEY` if using ask-Claude |
| — | WCCA intent JSON repair | Same repair-reparse pattern as the orchestrator, applied to WCCA intent parsing |

**Recently closed** (were on this list): embedding truncation on large memory stores (shipped `7660664`), `BRAVE_SEARCH_API_KEY` support (shipped), and the `set_dependencies(tool_registry=...)` startup crash (shipped `28f3c5c`).

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

6. **Retire or redirect `TASK.md`.** Root `TASK.md` (last real update
   2025-06-12) still shows "Phase 2 Neural Web — ACTIVE" and duplicates
   `planning/tasks/task-management.md`. It contradicts the July 2026 revival
   docs. Make `planning/roadmap/revival-2026-07.md` the canonical status doc and
   leave `TASK.md` as a one-line redirect.
7. **Flag stale roadmaps.** `planning/roadmap/neural-web-roadmap.md` (2025-06)
   predates the revival; add a header noting it's historical.
8. **Note the doc footprint.** ~96 `.md` files repo-wide (many inside
   submodules). The `docs/` synthetic-brain set overlaps heavily
   (`IMPLEMENTATION_STATUS.md`, `IMPLEMENTATION_SUMMARY.md`, `REMAINING_TASKS.md`,
   `README_SYNTHETIC_BRAIN.md`) — consolidate to one.

### Tier 3 — dead / dormant code (decide: wire up, archive, or delete)

9. **Neural web tools are built but unreachable.** `tools/enhanced_reasoning.py`
   (831 lines), `tools/neural_web_nlp.py` (673), `tools/neural_web_visualization.py`
   (739) are **not auto-registered** — the registry only auto-discovers tools
   with zero required constructor args, and these need `(config, llm_interface)`.
   Also `NeuralWebVisualizationTool.__init__` calls `super().__init__(config)`
   but `BaseTool.__init__` expects `(name, description)` — it would fail if
   instantiated. Either add a DI path (like document/web-search tools use
   `set_dependencies`) or explicitly mark them experimental.
10. **`core/adaptive_llm_interface.py` (371 lines) is dormant.** Only loaded when
    `default_provider: adaptive`; superseded by `core/model_router.py`. Mark
    experimental or remove.
11. **Parallel `*_fixed.py` / `*_updated.py` files.** e.g.
    `memory_handler.py` + `_fixed.py` + `_updated.py`,
    `cognitive_architecture.py` + `_updated.py`,
    `neural_memory_backend.py` + `_fixed.py`. Keep the live one, delete the
    orphans (grep shows some `_fixed` variants are imported nowhere).
12. **`gui/` (29 files, `matrix_ui.py` 932 lines) is parked.** Web UI replaced
    it. Move to `planning/archive/` or a separate branch to shrink the tree.
13. **~14 root-level `test_*.py` are never collected.** `pyproject.toml` sets
    `testpaths = tests`, so `test_enhanced_reasoning.py`,
    `test_neural_web_nlp.py`, `test_authentication.py`, etc. at the repo root run
    zero times. Move the useful ones into `tests/`, delete the rest.

### Tier 4 — structure & hygiene (larger, schedule deliberately)

14. **500-line rule is widely violated.** 20+ files exceed it; worst offenders:
    `agents/advanced_coding_agent.py` (1,482), `agents/wits_control_center_agent.py`
    (905), `agents/base_orchestrator_agent.py` (909), `web/server.py` (877). The
    orchestrator especially would benefit from splitting the guardrail/observation
    helpers into a mixin.
15. **`self_repair_agent` test is a 3-line stub.** Add real coverage for the
    self-repair path; also no dedicated tests for `book_writing_agent` or
    `neural_orchestrator_agent`.
16. **MCP submodule footprint.** Three vendored submodules (`servers`,
    `supabase-mcp`, `Ollama-mcp`) add 266+ files. Consider on-demand clone via
    the existing `scripts/clone_mcp_servers.py` instead of vendoring, to shrink
    the working tree.
17. **Confirm the Supabase `sbp_...` token was revoked.** Still an open manual
    item in the revival doc; it may remain valid in git history.

---

## Merge recommendation

When manual tests A–F look good:

1. Merge `composer/orchestrator-search-quality` → `fix/revive-2026-07` (or open a PR).
2. Update `revival-2026-07.md` §4 — model-routing settings, Ollama-down UX, save-to-file are shipped on branch.
3. Keep Gmail MCP entry in `data/mcp_tools.json` only if you intend to connect it from `/mcp` — it does not auto-connect.

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
