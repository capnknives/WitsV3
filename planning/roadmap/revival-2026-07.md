# WitsV3 Revival — July 2026 Status & Plan

**Branch:** `fix/revive-2026-07` (July 6–7 2026, merged to `main` at 1a555f2; later work continues on the branch)
**Test suite:** 222 passed / 2 skipped (skips are external MCP-server integration tests)
**Last updated:** July 7, 2026

This document tracks the July 2026 revival effort: what was worked on, what
broke along the way, how each issue was fixed, what remains open, and the
suggested next steps.

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
| `web_search` fails with DuckDuckGo status 202 (rate-limited/blocked) | ⚠️ OPEN — external; needs fallback backend or backoff |
| Orchestrator "Failed to parse reasoning response: Expecting ',' delimiter" (×15 in one session) | ✅ Fixed July 7 (orchestrator JSON robustness: `format=json` + robust parse + repair-reparse) |
| Ollama connection refused (service not running) | ⚠️ Environmental — start the tray app; consider a friendlier UI error |

---

## 3. What's left to do

### Richard's manual action items (browser/keys — can't be automated)
1. **Revoke the leaked `sbp_...` Supabase personal access token** at supabase.com/dashboard/account/tokens (leaked in git history before the secrets cleanup).
2. **Add `ANTHROPIC_API_KEY` to `.env`** if the ask-Claude escalation should work.

### Repo hygiene
3. ~~Push `fix/revive-2026-07` to GitHub and merge to `main`~~ DONE July 7 — branch pushed (with the mcp_servers repos registered as proper submodules) and merged to `main` (1a555f2) with Richard's authorization. Repo hygiene fully resolved.

### Roadmap (approved earlier)
4. ~~Smart model routing~~ SHIPPED July 7 2026: new `core/model_router.py` + `model_routing` config section (enabled by default; trivial → llama3.2:3b, code → qwen2.5-coder:7b, complex → default). Routes on the **raw user message/goal only** — never full prompt templates, which always look complex. Wired at three points: WCCA casual-chat response, WCCA fallback direct response, and the orchestrator ReAct loop (routes once per goal in `run()`; `allow_trivial=False` so the 3b model never handles ReAct JSON; code goals go to qwen2.5-coder, which should also reduce the malformed-JSON failures). `core/adaptive_llm_interface.py` was NOT activated — it routes to on-disk neural "modules" that don't exist, not to Ollama models; the lean router replaces that idea. Tests in `tests/core/test_model_router.py`; suite 207 passed / 2 skipped. Note: routing config is not in the /settings web page yet (config.local.yaml can override it).
5. ~~Orchestrator reasoning robustness~~ SHIPPED July 7 2026: three layers of defense against qwen3's malformed ReAct JSON. (a) **Ollama structured output**: reasoning calls now send `format=json` (new `format` param on `OllamaInterface.generate_text`/`_prepare_payload`, `response_format` on `BaseAgent.generate_response` — passed only when set, so other interfaces/test fakes are unaffected), which grammar-constrains generation to valid JSON. (b) **Robust parsing** in `LLMDrivenOrchestrator`: strips qwen3 `<think>` blocks, extracts candidates from markdown fences and string-aware balanced-brace scanning (the old greedy `\{.*\}` regex broke whenever a response had multiple `{...}` spans), completes truncated objects (closes open strings/braces), and applies conservative repairs (trailing commas, smart quotes, Python `True/False/None`). (c) **Repair-reparse round trip** in the ReAct loop: if parsing still fails, the fallback result is flagged `_parse_failed` and the model is asked once (also `format=json`) to rewrite its own output as valid JSON before keyword fallback is used. Tests in `tests/agents/test_orchestrator_json.py` (15 cases incl. the literal "Expecting ',' delimiter" failure); suite 222 passed / 2 skipped. Note: the related result-dismissing synthesis ("hallucinates instead of using document_search results") should also shrink — structured output stops think-block leakage into parsed thoughts — but watch the logs to confirm.

### Smaller quality items
6. **web_search resilience**: DuckDuckGo HTML endpoint rate-limits (status 202); add retry/backoff and/or an alternative backend.
7. **Friendlier "Ollama is down" handling** in the web UI (currently three retries then a raw error in chat).
8. **Intent-handler cleanup**: `clarification_question`/`direct_response` LLM intents still funnel through the casual-chat prompt instead of using their own generated text; the enhanced/meta-reasoning path duplicates routing logic. Worth unifying when touching WCCA next.

### Parked (explicitly out of scope for now)
- Docker packaging, Supabase cloud sync
- PyQt6 GUI (`gui/`) — works but Richard dislikes it; the web UI is the replacement

---

## 4. Suggested next steps (in order)

1. Do the two manual items (revoke Supabase token, add Anthropic key).
2. web_search fallback (#6) as a standalone small task.
3. Expose model_routing in the /settings web page.
4. Friendlier "Ollama is down" error in the web UI (#7).
