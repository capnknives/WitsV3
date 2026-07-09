# WitsV3

A local-first LLM orchestration system. Talk to it in a browser (or the CLI); it plans with a ReAct loop, calls real tools, remembers context, and can edit its own code safely when asked.

**Primary interface:** Web UI (`python run_web.py`) — desktop and phone (PWA).  
**LLM backend:** [Ollama](https://ollama.com) on your machine.  
**Status:** Actively maintained personal assistant stack — not a hosted SaaS product.

**Local folders on this PC:** personal runtime = `WitsV3`, Cursor agent = `WitsV3-cursor`, Claude agent = `WitsV3-claude` — see [`WORKTREES.md`](WORKTREES.md).

---

## Current status (2026-07-08)

| Area | State |
|------|--------|
| Test suite | **665 passed, 2 skipped** — `pytest tests/ -q --no-cov` |
| Release branch | **`main`** promoted from `fix/revive-2026-07` @ `4c676c3` (Phase 0 complete) |
| Models (default) | `qwen3:8b` (general + routing), `qwen2.5-coder:7b` (coding), `nomic-embed-text` (embeddings) |
| GPU target | Configured models fit in ~8 GB VRAM when fully on-GPU |
| Built-in tools | **26** auto-discovered (files, search, docs, MCP discovery, self-repair, neural-web helpers, …) |
| Agents | Control center → orchestrator + book / coding / self-repair specialists |
| Safe code edits | Verified-edit pipeline: write → pytest → commit, or revert to original bytes |
| Guest / family testers | Opt-in LAN access via `/join` (invite code; no owner token) — see below |
| CI | GitHub Actions on Python 3.10 / 3.11 (`.github/workflows/ci.yml`) |
| Secrets | Gitignored `.env` only — never in `config.yaml` |

**Recently shipped (July 2026):** Web UI + PWA, Document RAG, multi-provider web search, smart model routing, MCP registry discover/install, orchestrator JSON robustness, synthesis guard, one-click chat export, coding + self-repair, daily self-repair schedule, guest / family-tester access (full Phase 3–4), **Phase 1** (follow-up intent, hybrid doc search, memory flush, evidence gate, multi-session Chats panel), guest profile fact editor, **chat slash commands** (type `/` in the composer), **Ollama model pull/status panel** in Settings.

**What's next:** **Phase 2.2** — MCP server health panel. See [`docs/roadmap/suggested-features-2026-07.md`](docs/roadmap/suggested-features-2026-07.md).

---

## Quick start

### Prerequisites

- **Python 3.10+**
- **Ollama** installed and running ([ollama.com](https://ollama.com))
- ~8 GB VRAM recommended (CPU works; slower)

### 1. Clone and install

```bash
git clone https://github.com/capnknives/WitsV3.git
cd WitsV3

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
python scripts/setup_local_data.py
```

Or use the installer (deps + local data + auth + model pulls):

```bash
python install.py
```

### 2. Models

```bash
ollama pull qwen3:8b
ollama pull qwen2.5-coder:7b
ollama pull nomic-embed-text
```

Optional GPU tips (set as user environment variables, then restart Ollama):

- `OLLAMA_FLASH_ATTENTION=1`
- `OLLAMA_KV_CACHE_TYPE=q8_0`
- `OLLAMA_KEEP_ALIVE=10m`

Model store location is controlled in the Ollama app settings on Windows (tray app), not only by `OLLAMA_MODELS`.

### 3. Secrets

```bash
# Windows
copy .env.example .env

# macOS / Linux
# cp .env.example .env
```

Minimum for the web UI: set a strong `WITSV3_WEB_TOKEN` in `.env`.

Optional:

| Variable | Purpose |
|----------|---------|
| `WITSV3_AUTH_TOKEN_HASH` | Admin token hash — generate with `python setup_auth.py` |
| `WITSV3_GUEST_INVITE` | Short invite code for family testers on `/join` (requires `web_ui.guest_access.enabled`) |
| `WITSV3_GUEST_SECRET` | Signs guest session tokens (recommended random string; else derived from web token + invite) |
| `TAVILY_API_KEY` / `BRAVE_SEARCH_API_KEY` | Better `web_search` (DuckDuckGo is the keyless fallback) |
| `ANTHROPIC_API_KEY` | Ask-Claude escalation (per-request approval in the UI; never automatic) |
| `WITSV3_SUPABASE_*` | Optional Supabase memory backend — skip if unused |

### 4. Run

```bash
# Web UI (recommended) — uses project venv packages
.venv\Scripts\python.exe run_web.py          # Windows
# .venv/bin/python run_web.py                # macOS / Linux

# Or double-click start_web_ui.bat on Windows
```

Open `http://localhost:8000`, enter `WITSV3_WEB_TOKEN`, chat.

Phone (same Wi-Fi): open the LAN URL printed at startup → browser menu → **Add to Home screen** (PWA). Away from home: [Tailscale](https://tailscale.com) on PC and phone. One-time Windows firewall (Admin PowerShell):

```powershell
New-NetFirewallRule -DisplayName "WitsV3 Web UI" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow
```

### Guest / family testers (optional)

Let someone on your LAN (e.g. a nephew) chat **without** your owner token:

1. Set `WITSV3_GUEST_INVITE` (and ideally `WITSV3_GUEST_SECRET`) in `.env`
2. Set `web_ui.guest_access.enabled: true` in `config.yaml`
3. They open `http://<your-lan-ip>:8000/join`, enter the invite code + their name

Guests get chat-only UI, a filtered tool allowlist (no file write / self-repair / MCP / settings), and a remembered identity per browser/device. Owner manages testers on **`/settings`** (age band, profile view, revoke, merge duplicates). Content blocklists live in **`config/guest_policy.yaml`**. Full design: [`docs/roadmap/guest-tester-access-2026-07.md`](docs/roadmap/guest-tester-access-2026-07.md).

Other entry points:

```bash
python run.py              # CLI
python run.py --test       # Non-interactive self-check (init + LLM + tools + memory + agents)
pytest tests/ -q --no-cov  # Full test suite
```

`run.py` and `run_web.py` both schedule the daily autonomous self-repair job in-process when enabled in config (no separate Docker service required).

---

## What you can do

| Capability | How |
|------------|-----|
| Chat with streaming thinking / tools | Web UI or CLI |
| Ask about your files | Drop into `var/documents/` (auto-ingest) or upload in the UI; ask in plain English |
| Search the web | `web_search` — Tavily → Brave → DuckDuckGo |
| Edit project code safely | “Fix the bug in `tools/foo.py`” → verified-edit pipeline |
| Autonomous maintenance | Daily self-repair scan (config: `self_repair.*`) |
| Guest / family tester on LAN | `/join` with invite code (opt-in; no owner token) |
| Extra tools | MCP discover/install on `/mcp` |

### Document RAG

Files in `var/documents/` are chunked, embedded (`nomic-embed-text`), and searchable. Built-in: `.txt` `.md` `.py` `.json` `.csv` `.html` `.log` (`.pdf` needs `pypdf`, already in requirements). Changed files re-ingest; deletes clean up. Tune under `document_rag:` in `config.yaml`.

### Self-repair & coding agent

Shared pipeline in `core/safe_code_editor.py`:

1. Snapshot original bytes  
2. Write candidate  
3. Run pytest  
4. Pass → git commit · Fail → restore exact original (nothing broken left on disk)

- Named file in the request → that file is targeted  
- No file named → self-repair scans `logs/witsv3.log` (then failing tests if needed)  
- New projects from the coding agent land in `var/workspace/<name>/` with `py_compile` checks  
- Edits stay inside the project tree (`resolve_within_project()`)

Defaults: `self_repair.enabled: true`, daily cron `0 3 * * *`, `restart_after_fix: false` (scheduled restarts never surprise an active session).

---

## Configuration

Main file: `config.yaml` (schema/defaults in `core/config.py`).

| Setting | Default | Purpose |
|---------|---------|---------|
| `ollama_settings.default_model` | `qwen3:8b` | General / orchestration |
| `ollama_settings.coding_agent_model` | `qwen2.5-coder:7b` | Coding agent |
| `ollama_settings.embedding_model` | `nomic-embed-text` | Embeddings (768-dim) |
| `memory_manager.backend` | `basic` | `basic`, `faiss_cpu`, `faiss_gpu`, or `neural` |
| `memory_manager.vector_dim` | `768` | Must match embedding model |
| `model_routing.enabled` | `false` (toggle in `/settings`) | Size models by query complexity when enabled |
| `tool_system.mcp_connect_on_startup` | `false` | Skip slow MCP boot unless you want it |
| `self_repair.enabled` | `true` | Diagnose / apply verified fixes |
| `self_repair.daily_schedule_enabled` | `true` | In-process daily scan |
| `self_repair.restart_after_fix` | `false` | Restart after a verified fix |

---

## Architecture (short)

```
User (Web UI / CLI)
        │
        ▼
WitsControlCenterAgent  ── intent, routing, conversation
        │
        ├──► LLMDrivenOrchestrator  (ReAct + tools + synthesis guard)
        ├──► AdvancedCodingAgent    (scaffolds + verified edits)
        ├──► SelfRepairAgent        (logs / tests → verified fixes)
        └──► BookWritingAgent
                │
                ▼
     Memory Manager · Tool Registry · Ollama · optional Neural Web / MCP
```

- **Verified edits:** `core/safe_code_editor.py` + `apply_code_fix` / related tools  
- **Model routing:** `core/model_router.py` (prefer this over the dormant adaptive-LLM stack)  
- **Docker / Supabase / PyQt6 GUI:** parked or optional — see roadmap “Parked”

---

## Project layout

```
WitsV3/
├── agents/           # Control center, orchestrator, specialists
├── core/             # Config, LLM, memory, safe_code_editor, schemas
├── tools/            # Auto-discovered tools (26 built-in)
├── web/              # FastAPI + SSE UI
├── tests/            # Pytest suite (~406 collected)
├── var/              # Runtime data (memory, documents, exports, logs, workspace, cache)
├── docs/             # Architecture, roadmap, historical notes
├── config.yaml       # Main config
├── run_web.py        # Web entry
├── run.py            # CLI + --test + daily self-repair schedule
└── .env.example      # Secrets template
```

---

## Documentation map

| Doc | Role |
|-----|------|
| **This README** | Install, run, capabilities, config |
| [`WORKTREES.md`](WORKTREES.md) | Personal vs Cursor vs Claude local folders |
| [`AGENTS.md`](AGENTS.md) | Agent hierarchy and conventions |
| [`FILE_STRUCTURE.md`](FILE_STRUCTURE.md) | Directory reference |
| [`docs/roadmap/suggested-features-2026-07.md`](docs/roadmap/suggested-features-2026-07.md) | **What's next** (canonical) |
| [`docs/roadmap/revival-2026-07.md`](docs/roadmap/revival-2026-07.md) | July 2026 shipped work log |
| [`docs/README.md`](docs/README.md) | Index of all planning docs |
| [`TASK.md`](TASK.md) / [`PLANNING.md`](PLANNING.md) | Redirects to the docs above |
| [`docs/archive/historical-docs/SYNTHETIC_BRAIN.md`](docs/archive/historical-docs/SYNTHETIC_BRAIN.md) | Historical “synthetic brain” initiative (not current product) |
| [`DOCKER_INSTRUCTIONS.md`](DOCKER_INSTRUCTIONS.md) | Parked background-agent container notes |

Changelog detail lives in the revival log and git history — this README stays focused on how to run and use the system today.

---

## Contributing

- Tests for new behavior (`tests/` mirrors packages); async + mocks for Ollama/IO  
- Type hints, Google-style docstrings, black / pep8  
- Prefer config over hardcoded values; keep edits inside the project tree for tool safety  
- Forward work: update `docs/roadmap/suggested-features-2026-07.md`, not the retired `TASK.md` body  

## License

MIT — see [LICENSE](LICENSE).
