# WitsV3 Revival — July 2026 Status & Plan

**Branch:** `fix/revive-2026-07` (July 6–7 2026, merged to `main` at 1a555f2; later work continues on the branch)
**Test suite:** 318 passed / 2 skipped (skips are external MCP-server integration tests)
**Last updated:** July 7, 2026

This document tracks the July 2026 revival effort: what was worked on, what
broke along the way, how each issue was fixed, what remains open, and the
suggested next steps.

### Git workflow (integration branch)

**All revival work lands on `fix/revive-2026-07` first.** Feature branches
(e.g. `cursor/*`, `composer/*`, `claude/*`) merge there when a chunk of work is
done; agents and sessions should **push to `fix/revive-2026-07`**, not directly
to `main`. Richard merges `fix/revive-2026-07` → `main` when manual testing
looks good — that promotion step stays a deliberate human decision.

---

## 1. Work completed (by theme)

### Core revival (July 6)
| Commit | What |
|---|---|
| `705ebf0` | Revived the project: 17 bugs fixed, test suite green (142 passed), secrets moved to gitignored `.env` (loaded via `core/config.py`), `config.yaml` scrubbed |
| `c3770fe` | Model refresh: qwen3:8b (orchestrator), qwen2.5-coder:7b (coding), llama3.2:3b (fast), nomic-embed-text (768-dim); fixed `vector_dim`; faster startup |
| `512d2c3` | README rewritten (was corrupted), documents July 2026 state |
| `3b1b36f` | MCP stdio handshake fixed: proper `initialize` + `notifications/initialized`, notification-skipping reader (was sending a fake "info" method) |

### Document RAG (July 6–7)
| Commit | What |
|---|---|
| `4169cb9` | `documents/` drop folder, auto-ingest at startup, semantic `document_search` tool, `delete_segments` API on memory backends |
| `89fd4ac` | **pypdf installed + in requirements** — PDFs were silently skipped at every ingest ("PDF support requires pip install pypdf") |
| `ed0ef6c` | `ingest_documents` result now self-describing: `searchable_files` per-file chunk counts + plain-language `message`, because qwen3 read `files_ingested: 0` (file already ingested at upload) as "no access" |

### Web UI (July 6–7)
| Commit | What |
|---|---|
| `9b37d8b` | FastAPI+SSE streaming chat on 0.0.0.0:8000, PWA for Android, bearer auth (`WITSV3_WEB_TOKEN`) |
| `59deb6f` `194d26b` | `start_web_ui.bat` launcher, friendly already-running message |
| `2da61a5` `5234387` | Magic login link (`?token=`), QR code in the startup banner, client/cache hardening |
| `78bfa04` `cface0f` | Token modal fixed — prompt ignored Enter (now a real form validating via `/api/status`), and `display:flex` beat `[hidden]` so the modal could never close |
| `0cd8d1b` | Same `[hidden]`-vs-`display` CSS bug on the mobile side panel; whole stylesheet swept |
| `e767cb6` | `/personality` questionnaire page → `config/personality_overrides.yaml`, deep-merged, applies live |
| `2b3326b` | `/settings` page (context window, temperature, models, escalation → `config.local.yaml`), `/mcp` server manager, ask-Claude escalation (approval card + cost estimate; needs `ANTHROPIC_API_KEY`) |

### Auth & security (July 6–7)
| Commit | What |
|---|---|
| `705ebf0` | Secrets out of git; `.env` + env-override loading |
| `21dfdcc` | `setup_auth.py` preserves `.env` on save; UTF-8 console |
| `1f530b5` | `.env` token/hash divergence fixed (every `verify_auth` was failing); `setup_initial_token` strips stale token lines; auth test suite repaired |

### Chat quality & routing (July 7)
| Commit | What |
|---|---|
| `8514e62` | "Amnesia" fix: direct-response window 5 → 20 messages (configurable via `/settings`) |
| `89fd4ac` | `validate_tool_call` now unwraps schemas nested under `"parameters"` — valid args were flagged "unknown" and required-param checks were silently skipped for `document_search`/`web_search`/`ask_claude` |
| `32f857f` | **Orchestrator JSON robustness** (roadmap #5): reasoning calls use Ollama `format=json`; parser strips `<think>` blocks, scans for balanced JSON, completes truncated objects, repairs common syntax slips; one-shot repair-reparse LLM retry before keyword fallback |
| `bddfe8f` | **Web search actually used for current events**: "who died June 14 2026?" refused ("training only goes to 2023") without searching. Fixed 4 defects — (1) WCCA `_needs_web_search()` routes current-info/lookup questions to the orchestrator before the casual heuristic (which flagged any <10-word question as small talk); (2) `ReliableOllamaInterface` dropped the `format` kwarg → `format=json` reasoning crashed ALL orchestration at runtime (now `**kwargs` pass-through); (3) ReAct prompt now tells the model to call web_search and never refuse on a cutoff; (4) Tavily `basic` depth gave a wrong answer (Ken Errair, d.1968) → default `advanced` returns Oliver Tree. Verified live end-to-end. Suite 251 passed / 2 skipped |
| `277960d` | **Control-center routing overhaul** (the "too rigid" fix): (a) casual-chat detector substring-matched "hi" inside "things"/"this" → real requests answered as small talk, now word-boundary matching; (b) intent analysis + direct-response prompts now include a live ingested-document inventory, and any message mentioning an ingested filename (or words from it, e.g. "audit") routes straight to the orchestrator, checked before the casual heuristic; (c) LLM `goal_defined` intents defaulted to "simple" and never reached the orchestrator — now they delegate. Regression tests in `tests/agents/test_wcca_routing.py` |

### Environment (no commits — machine/system state)
- Ollama model store moved to `D:\OllamaModels` (via the tray app's db.sqlite; it ignores `OLLAMA_MODELS`); GPU tuning env vars (`OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0`, `OLLAMA_KEEP_ALIVE=10m`); all 4 models 100% GPU on the RTX 3070
- npm cache junction bug fixed (`npm config set cache "D:\ClaudeData\npm-cache"`)
- Firewall rule for TCP 8000 in place; web UI reachable from the phone
- Gotcha: the Ollama tray app is sometimes not running — chat then fails with connection errors at `localhost:11434`; start `%LOCALAPPDATA%\Programs\Ollama\ollama app.exe`

---

## 2. Error-log triage (from `logs/witsv3.log`, July 6–7 chats)

| Error | Status |
|---|---|
| `ToolResult` import failure, missing matplotlib, `add_memory_segment` | ✅ Fixed July 6 (revival pass) |
| MCP clone/loader/npm-EEXIST cascade at startup | ✅ Fixed July 6 (`3b1b36f` + npm cache fix); MCP startup connect now off by default, managed via `/mcp` page |
| `verify_auth` failing on every request | ✅ Fixed (`1f530b5`) |
| "Unknown parameters for document_search/web_search: ['query']" | ✅ Fixed (`89fd4ac`) |
| PDFs skipped — pypdf missing | ✅ Fixed (`89fd4ac`) |
| "I don't have access to your report" despite successful search | ✅ Fixed (`ed0ef6c` + `277960d`) |
| Document questions misrouted to chat/clarification unless phrased exactly | ✅ Fixed (`277960d`) |
| `web_search` fails with DuckDuckGo status 202 (rate-limited/blocked) | ✅ Fixed July 7 — rewritten as a multi-provider tool: real DuckDuckGo HTML/Lite scraping (with a browser UA + 202/429 retry-backoff) instead of the useless Instant Answer API, plus optional Tavily/Brave via `.env` keys |
| Orchestrator "Failed to parse reasoning response" on save-to-file (×16, content embedded in JSON) | ✅ Fixed July 7 — save-to-file flow: WCCA routing, session injection, content auto-fill, strip large bodies from ReAct JSON |
| Ollama connection refused (service not running) | ⚠️ Environmental — start the tray app; consider a friendlier UI error |

---

## 3. What's left to do

### Richard's manual action items (browser/keys — can't be automated)
- [x] ~~**Revoke the leaked `sbp_...` Supabase personal access token**~~ N/A confirmed July 7 2026 — Richard has no Supabase project anymore, so the leaked token has nothing to grant access to. No action needed; left in git history as a dead credential.
- [ ] **Add `ANTHROPIC_API_KEY` to `.env`** if the ask-Claude escalation should work.
- [x] ~~**Get a Tavily web-search API key**~~ DONE July 7 2026 — key added to `.env` as `TAVILY_API_KEY`; verified live, `web_search` now routes through Tavily (returns synthesized answer + results).
- [ ] **Get a Brave Search API key** at brave.com/search/api (free tier: 2000/mo) and add `BRAVE_SEARCH_API_KEY=...` to `.env`. Good second option / fallback. Neither key is required — `web_search` already returns real results keyless via DuckDuckGo.

### Repo hygiene
- [x] ~~Push `fix/revive-2026-07` to GitHub and merge to `main`~~ DONE July 7 — branch pushed (with the mcp_servers repos registered as proper submodules) and merged to `main` (1a555f2) with Richard's authorization. Repo hygiene fully resolved.

### Roadmap (approved earlier)
4. ~~Smart model routing~~ SHIPPED July 7 2026: new `core/model_router.py` + `model_routing` config section (enabled by default; trivial → llama3.2:3b, code → qwen2.5-coder:7b, complex → default). Routes on the **raw user message/goal only** — never full prompt templates, which always look complex. Wired at three points: WCCA casual-chat response, WCCA fallback direct response, and the orchestrator ReAct loop (routes once per goal in `run()`; `allow_trivial=False` so the 3b model never handles ReAct JSON; code goals go to qwen2.5-coder, which should also reduce the malformed-JSON failures). `core/adaptive_llm_interface.py` was NOT activated — it routes to on-disk neural "modules" that don't exist, not to Ollama models; the lean router replaces that idea. Tests in `tests/core/test_model_router.py`; suite 207 passed / 2 skipped. Note: routing config is not in the /settings web page yet (config.local.yaml can override it).
5. ~~Orchestrator reasoning robustness~~ SHIPPED July 7 2026: three layers of defense against qwen3's malformed ReAct JSON. (a) **Ollama structured output**: reasoning calls now send `format=json` (new `format` param on `OllamaInterface.generate_text`/`_prepare_payload`, `response_format` on `BaseAgent.generate_response` — passed only when set, so other interfaces/test fakes are unaffected), which grammar-constrains generation to valid JSON. (b) **Robust parsing** in `LLMDrivenOrchestrator`: strips qwen3 `<think>` blocks, extracts candidates from markdown fences and string-aware balanced-brace scanning (the old greedy `\{.*\}` regex broke whenever a response had multiple `{...}` spans), completes truncated objects (closes open strings/braces), and applies conservative repairs (trailing commas, smart quotes, Python `True/False/None`). (c) **Repair-reparse round trip** in the ReAct loop: if parsing still fails, the fallback result is flagged `_parse_failed` and the model is asked once (also `format=json`) to rewrite its own output as valid JSON before keyword fallback is used. Tests in `tests/agents/test_orchestrator_json.py` (15 cases incl. the literal "Expecting ',' delimiter" failure); suite 222 passed / 2 skipped. Note: the related result-dismissing synthesis ("hallucinates instead of using document_search results") should also shrink — structured output stops think-block leakage into parsed thoughts — but watch the logs to confirm.

### Smaller quality items
6. ~~**web_search resilience**~~ SHIPPED July 7 2026: `tools/web_search_tool.py` rewritten from a single dead endpoint into a provider fallback chain. Root cause was deeper than the 202 — the old tool only hit the DuckDuckGo **Instant Answer API** (`api.duckduckgo.com`), which is not a web search engine (it returns Wikipedia-style disambiguation "RelatedTopics" for a tiny set of queries and 202s under load), so most searches returned empty even when not blocked. New chain: **Tavily** (if `TAVILY_API_KEY`) → **Brave** (if `BRAVE_SEARCH_API_KEY`) → **DuckDuckGo HTML** scrape → **DuckDuckGo Lite** scrape → Instant Answer API (last resort). The DDG scrape paths send a real browser User-Agent (the default aiohttp UA is what triggers the 202 bot wall), unwrap DDG's `/l/?uddg=` redirect links, and retry on 202/429 with exponential backoff. Keyless works out of the box (verified live — real organic results with snippets); keys just improve quality. New `web_search` config section (`provider`/`max_results`/`timeout`/`region`/`safesearch`); keys injected from `.env`, never config.yaml. Tool now takes `set_dependencies(config, ...)`. Tests in `tests/tools/test_web_search_tool.py` (8 cases: HTML parse, redirect-unwrap, max_results, 202→Lite fallthrough, Tavily-preferred, all-fail); suite 226 passed / 2 skipped.
7. ~~**MCP tool discovery / marketplace**~~ SHIPPED July 7 2026: WITS can now find and add *new* capabilities on demand from the official MCP registry (registry.modelcontextprotocol.io), instead of only using pre-configured servers. New `core/mcp_registry_search.py` queries the registry and derives a runnable stdio command per package (npm → `npx -y pkg@ver`, pypi → `uvx pkg==ver`), surfaces required env vars, dedupes by `isLatest`, and — because the registry matches `search` as a literal phrase — falls back from a multi-word query to individual keywords, merging + relevance-ranking the results. Two ways in: (a) agent-facing read-only tool `search_mcp_tools` (`tools/mcp_discovery_tool.py`, auto-registered) so WITS can look up a server when no installed tool fits; (b) `/mcp` web page "Discover servers" search box with one-click **Install** (writes the config entry + required-env inputs) then the existing **Connect** button runs it. New web routes `GET /api/mcp/registry/search` + `POST /api/mcp/registry/install`; new `tool_system.mcp_registry_url` config. **Security boundary:** searching is read-only and safe; *installing/connecting downloads and runs third-party code*, so that stays a deliberate human click in the web UI — the agent can discover but never self-installs. Tests in `tests/core/test_mcp_registry_search.py` (12: command derivation for npm/pypi/non-stdio/oci, env-var + isLatest handling, keyword fallback, HTTP-error). Verified live against the real registry (postgres/slack/gmail/spotify all return installable servers with correct commands). Suite 238 passed / 2 skipped.
8. ~~**Friendlier "Ollama is down" handling**~~ SHIPPED July 7 2026 (Composer branch): friendly errors in web UI.
9. ~~**Intent-handler cleanup**~~ SHIPPED July 7 2026: `direct_response`/`clarification_question` now use the intent JSON text instead of a second casual-chat LLM call; `_requires_orchestrator_for_input()` unifies doc + web-search guards; misclassified `direct_response` on tool-needed queries is overridden to orchestrator. Tests in `tests/agents/test_wcca_routing.py`.
10. ~~**WCCA intent JSON repair-reparse**~~ SHIPPED July 7 2026 (`bd5e22a`): shared `core/json_llm_parser.py`; WCCA intent calls use `format=json` + repair-reparse before heuristic fallback. Wired into `wcca_intent_mixin.py` after Tier 4 split. Tests in `tests/agents/test_wcca_intent_json.py`.
11. ~~**Save conversation/story to file**~~ SHIPPED July 7 2026 (Composer branch): log analysis showed `write_file` never ran — save requests hit direct-response or ReAct JSON broke when the model embedded file bodies in reasoning JSON. Fixes: WCCA `_needs_file_write()` routes save/export phrasing to orchestrator; orchestrator injects `conversation_history` into `read_conversation_history`; auto-fills `write_file` content from observations or session; strips >300-char `content` from ReAct JSON; `write_file` kwarg aliases (`path`→`file_path`, `text`/`body`→`content`); two-step prompt (read history → write file, never embed body in JSON). Tests in `tests/agents/test_orchestrator_save_file.py`, `tests/core/test_tool_registry_kwargs.py`, `tests/agents/test_wcca_routing.py`.
12. ~~**MCP discovery follow-ups**~~ SHIPPED July 7 2026: (a) **OCI/Docker packages** — `core/mcp_registry_search.py` now maps `registryType: oci` to a `docker run -i --rm` command (image `:version` tag added only if the identifier doesn't already carry one), and forwards each package's declared env vars as bare `-e NAME` flags so docker inherits them from the host process env the same way npx/uvx already do — without baking secrets into argv. (b) **Browse-before-install preview** — `normalize_server()` now also returns `install_options` (every locally-runnable package, not just the first); the `/mcp` Discover tab renders a package picker when a server offers more than one (e.g. npm vs. docker), plus a collapsible "Details" panel listing every package's registry type/version/transport and any remote-only endpoints, all before the Install button is touched. (c) **Deep-link** — `search_mcp_tools` now returns a `deep_link` (`/mcp?discover=<query>`) in its result and message; `/mcp` reads `?discover=` on load, switches to Discover, and re-runs the search automatically. Tests: 4 new cases in `tests/core/test_mcp_registry_search.py` (docker command derivation, tag handling, env forwarding, no duplicate base flags) and new `tests/tools/test_mcp_discovery_tool.py` (3 cases incl. the deep-link URL). Full suite 329 passed / 2 skipped.

### Parked (explicitly out of scope for now)
- Docker packaging, Supabase cloud sync
- PyQt6 GUI — archived July 7 2026 to `planning/archive/gui/` (was `gui/`); web UI is the replacement

---

## 4. Suggested next steps (in order)

1. Optionally add `ANTHROPIC_API_KEY` if the ask-Claude escalation should work. Brave key: DONE (`BRAVE_SEARCH_API_KEY` added; `web_search` merges Tavily+Brave). Supabase token: N/A, no project exists anymore.
2. **Re-test save-to-file** after restart: *"Save a log of our conversations as exports/chat_log.txt"* — expect orchestrator → `read_conversation_history` → `write_file`, not JSON parse loops.
3. ~~Merge `composer/orchestrator-search-quality` → `main`~~ DONE July 7 2026: `claude/tier2-tier3-cleanup-2026-07` (superset of `composer/orchestrator-search-quality` plus Tier 2/3 cleanup below) fast-forward merged into `fix/revive-2026-07`.
4. ~~Optional roadmap #11: MCP OCI/Docker install, browse-before-install preview, deep-link from `search_mcp_tools` to `/mcp`.~~ DONE (`86c513d`).
5. ~~WCCA intent JSON repair-reparse~~ SHIPPED July 7 2026 (`bd5e22a` on `main`; merged into `fix/revive-2026-07` with Tier 4 mixin split). Embedding input truncation is DONE (`7660664`).

### Whole-repo audit backlog (July 7, 2026)

A read-only pass over the entire codebase (beyond the search/MCP work) produced a
prioritized backlog of "obviously missing" items — CI, tooling-config
de-duplication, dead/dormant code (neural web tools, `adaptive_llm_interface`,
`gui/`, `*_fixed.py`/`*_updated.py`), uncollected root-level tests, and the
500-line-rule violations. **Tiers 1–3 shipped July 7 2026** (CI + tooling
config consolidation by the Cursor session; doc truth pass + neural-web-tools
wiring + dead-code archival/deletion by Claude — see the **"Codebase audit —
obviously missing / suggested additions"** section in
`planning/roadmap/composer-orchestrator-search-quality-2026-07.md` for the
full list). **Tier 4 complete (July 7 2026):** orchestrator/WCCA/book/coding/web
500-line splits, agent pytest coverage, MCP on-demand-only (all submodules removed
from git). Optional follow-up: other large core/tools modules not in the original
Tier 4 worst-offender list.
