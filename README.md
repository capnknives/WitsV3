# WitsV3

A streamlined, LLM-wrapper based AI orchestration system that runs entirely on local models via Ollama.

## Overview

WitsV3 is designed for maximum flexibility and LLM-driven decision making. It focuses on a CLI-first approach with a modular design: a control-center agent clarifies goals, a ReAct-style orchestrator decomposes and executes tasks with tools, and a persistent memory system (with optional vector search and a neural-web knowledge graph) carries context across sessions.

## ✅ Current Status — Fully Operational (Updated: 2026-07-06)

- **🎯 Test Suite**: **142 passed, 2 skipped** (skips are external MCP-server integration tests), 0 failures
- **⚡ 100% GPU inference**: all configured models fit fully in 8 GB VRAM — no CPU spillover
- **🤖 Models**: Qwen3 8B (general/orchestration), Qwen2.5-Coder 7B (coding), Llama 3.2 3B (fast fallback), nomic-embed-text (embeddings)
- **⚙️ Tool Registry**: 14 tools auto-discovered and registered
- **🔐 Secrets hygiene**: credentials live in a gitignored `.env`, never in `config.yaml`
- **🧠 Agent System**: LLM-driven orchestrator with ReAct pattern, control center, and specialized agents (book writing, coding, self-repair)

### 📋 Changelog

**2026-07-06 — Web UI (desktop + phone)**

- New browser interface: streaming chat (SSE) over the full agent system, with side panels for tools, memory search, and document upload/list
- Mobile-first + installable as a PWA on Android ("Add to Home screen")
- Runs with `python run_web.py`; binds to the LAN so your phone can connect; bearer-token auth via `WITSV3_WEB_TOKEN` in `.env`
- Control center now routes document/file/memory questions to the tool-equipped orchestrator instead of answering directly

**2026-07-06 — Document RAG**

- New `documents/` drop folder: files are chunked, embedded with `nomic-embed-text`, and stored as searchable memory segments (auto-ingest at startup + `ingest_documents` tool)
- New `document_search` tool: agents answer questions from your documents via semantic search
- Changed files are re-ingested, deleted files are cleaned up (new `delete_segments` API on all memory backends)
- Supports .txt/.md/.py/.json/.csv/.html/.log out of the box; .pdf with `pip install pypdf`

**2026-07-06 — Model upgrade & performance pass** (`c3770fe`)

- Replaced the model lineup: `llama3` → **`qwen3:8b`** (default/orchestrator/control center), `deepseek-coder-v2:16b` (10 GB, spilled to CPU) → **`qwen2.5-coder:7b`** (4.7 GB, fully GPU); added `llama3.2:3b` as fast fallback
- Fixed `memory_manager.vector_dim` to 768 to match nomic-embed-text (was 384 in config and 4096 in code defaults — neither correct)
- New `tool_system.mcp_connect_on_startup` flag (default `false`) — startup no longer wastes ~20 s attempting to reach unbuilt external MCP node servers
- `auto_restart_on_file_change` now defaults off (no surprise restarts while editing)
- Ollama tuning applied at the environment level: flash attention, `q8_0` KV cache, 10-minute keep-alive; model store relocated to a dedicated data drive
- Measured result: control-center responses dropped from ~14–21 s to **~2–3 s**

**2026-07-06 — Revival & hardening pass** (`705ebf0`)

- Fixed 17 bugs across the codebase, including: broken stdout handling that crashed `run.py` at launch on Windows, tool auto-discovery failing on a missing `ToolResult` export, datetime-serialization failures in the FAISS memory backend and memory export, the cross-domain-learning module targeting a NeuralWeb API that didn't exist, missing `import os` in the JSON tool, and Windows `npm`/`npx` resolution in the MCP adapters
- Restored conversation-summarization features (agent-interaction summaries, typed transcripts, manual topic extraction)
- Moved all secrets (Supabase key, auth token hash) out of `config.yaml` into a gitignored `.env` with environment-variable overrides in `core/config.py`
- Rebuilt the test suite health: **98 passed / 36 failed / 8 errors → 142 passed / 0 failed** (updated stale mocks to httpx 0.28 and the retry-aware LLM interface contract, fixed a nondeterministic embedding fixture)
- Pinned a reproducible dependency set in `requirements.lock` (Python 3.10, pydantic 2.13, httpx 0.28, faiss-cpu 1.14, supabase 2.31)

## Getting Started

### Prerequisites

- **Python 3.10+**
- **Ollama** ([ollama.com](https://ollama.com)) with the models below
- An NVIDIA GPU with ~8 GB VRAM is recommended — every configured model fits fully on-GPU at that size

### Models

```bash
ollama pull qwen3:8b            # general / orchestration (~5.2 GB)
ollama pull qwen2.5-coder:7b    # coding agent (~4.7 GB)
ollama pull llama3.2:3b         # fast fallback (~2.0 GB)
ollama pull nomic-embed-text    # embeddings, 768-dim (~274 MB)
```

> **Tip — model storage location:** the Ollama desktop app stores models under `C:\Users\<you>\.ollama\models` by default and manages the location in its own settings (the `OLLAMA_MODELS` environment variable alone is not enough when using the tray app). Change it in the Ollama app settings if you want models on another drive.

> **Tip — GPU performance:** setting `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0`, and `OLLAMA_KEEP_ALIVE=10m` (user environment variables, then restart Ollama) speeds up attention, halves KV-cache VRAM, and avoids model reloads between agent calls.

### Installation

#### Option 1: Automated

```bash
git clone https://github.com/capnknives/WitsV3.git
cd WitsV3
python install.py
```

#### Option 2: Manual

```bash
git clone https://github.com/capnknives/WitsV3.git
cd WitsV3

# Recommended: use a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

pip install -r requirements.txt

# Set up local data files (memory, neural web)
python scripts/setup_local_data.py

# Set up secrets + authentication
copy .env.example .env        # then fill in your values
python setup_auth.py          # generates an auth token (hash stored in .env)

# Verify
pytest tests/ -q --no-cov
```

### First Run Setup

1. **Secrets**: copy `.env.example` to `.env` and fill in your values. Secrets (Supabase key, auth token hash) live in the gitignored `.env` — never in `config.yaml`.
2. **Authentication**: run `python setup_auth.py` to generate your authentication token. Save the token securely; only its hash is stored.
3. **Local data**: run `python scripts/setup_local_data.py` to initialize memory files.
4. **Ollama**: make sure Ollama is running and the four models above are pulled.

### Running WitsV3

```bash
# Start the CLI interface
python run.py

# Non-interactive system self-test (init + LLM + tools + memory + agents)
python run.py --test

# Run the background agent
python run_background_agent.py

# Run the test suite
pytest tests/ -q --no-cov
```

### Quick Start with Make

```bash
make install-dev   # install development dependencies
make dev           # run the main application
make test-cov      # run tests with coverage
make build         # clean and build package
```

## Configuration

Main configuration is `config.yaml`; the pydantic models (and defaults) live in `core/config.py`. Highlights:

| Setting | Default | Purpose |
| --- | --- | --- |
| `ollama_settings.default_model` | `qwen3:8b` | General/orchestration model |
| `ollama_settings.coding_agent_model` | `qwen2.5-coder:7b` | Coding agent model |
| `ollama_settings.embedding_model` | `nomic-embed-text` | Embeddings (768-dim) |
| `ollama_settings.fallback_models` | qwen3:8b → llama3.2:3b → qwen2.5-coder:7b | Automatic fallback order |
| `memory_manager.backend` | `basic` | `basic`, `faiss_cpu`, `faiss_gpu`, or `neural` |
| `memory_manager.vector_dim` | `768` | Must match the embedding model |
| `tool_system.mcp_connect_on_startup` | `false` | Connect to external MCP servers at boot |
| `auto_restart_on_file_change` | `false` | Watchdog-based auto-restart for development |

Secrets are supplied via `.env` / environment variables (loaded in `core/config.py`):

- `WITSV3_SUPABASE_URL`, `WITSV3_SUPABASE_KEY` — optional Supabase memory backend
- `WITSV3_AUTH_TOKEN_HASH` — SHA-256 hash of the admin token (written by `setup_auth.py`)

## Architecture

- **WitsControlCenterAgent (WCCA)** — user interaction, intent parsing, and goal clarification
- **LLMDrivenOrchestrator** — ReAct-style reasoning loop that plans, calls tools, and synthesizes results
- **Specialized agents** — book writing, coding, and self-repair agents invoked by the control center
- **Memory Manager** — persistent memory segments with embeddings; backends for JSON (basic), FAISS, neural-web, and Supabase
- **Neural Web** — concept graph with activation propagation and cross-domain learning
- **Tool Registry** — auto-discovers tools (14 built-in: file ops, calculator, math, JSON, Python execution, web search, datetime, conversation analysis, network control, thinking, intent analysis)
- **Adaptive LLM System** — complexity analyzer + dynamic module loader + semantic cache for routing queries to appropriately-sized models (`llm_interface.default_provider: adaptive` to enable)
- **MCP integration** — Model Context Protocol adapter for external tool servers (opt-in at startup)

## Web UI

**Easiest:** double-click `start_web_ui.bat` (or the "WITS Web UI" desktop shortcut) — it uses the
project's `.venv` automatically, from any location.

Manual equivalent (note: plain `python` from a random prompt won't work — the web packages live in
the project venv):

```bash
cd C:\path\to\WitsV3
.venv\Scripts\python.exe run_web.py
```

Open `http://localhost:8000` on the PC — or from your **Android phone** on the same Wi-Fi,
open the `http://<PC-IP>:8000` URL the server prints at startup, enter your
`WITSV3_WEB_TOKEN`, and use the browser menu → **Add to Home screen** for an app-like
fullscreen experience (PWA).

- Streaming chat with live thinking/tool-call events from the agent system
- ☰ panel: tool list, semantic memory search, document list + upload (feeds Document RAG)
- Auth: set `WITSV3_WEB_TOKEN` in `.env` (any strong random string); the UI prompts once and remembers it
- One-time firewall rule (run PowerShell **as Administrator**):
  `New-NetFirewallRule -DisplayName "WitsV3 Web UI" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow`
- Away from home? Install [Tailscale](https://tailscale.com) on the PC and phone — the same URL works
  over its private VPN with zero port forwarding.

## Document RAG

Drop files into the `documents/` folder and WitsV3 ingests them automatically at startup
(chunked → embedded → stored in memory). Then just ask about them in the CLI — the
orchestrator uses the `document_search` tool to find relevant passages.

- Supported out of the box: `.txt`, `.md`, `.py`, `.json`, `.csv`, `.html`, `.log`
  (`.pdf` too, with `pip install pypdf`)
- Changed files are re-ingested; deleted files have their chunks cleaned up
- Manual rescan: ask the agent to "ingest documents", or call the `ingest_documents` tool
- Tuning in `config.yaml` under `document_rag:` (folder path, chunk size/overlap, startup ingest)

## Roadmap

1. ~~**Web UI**~~ — ✅ shipped 2026-07-06 (see above), including Android/PWA access
2. ~~**Document RAG**~~ — ✅ shipped 2026-07-06 (see above)
3. **Smart model routing** — wire the adaptive-LLM stack so trivial queries hit `llama3.2:3b`, code hits `qwen2.5-coder:7b`, and complex reasoning hits `qwen3:8b`
4. PyQt6 desktop GUI and background-agent revival (in progress on a separate branch)

## Documentation

- [FILE_STRUCTURE.md](FILE_STRUCTURE.md) — file structure reference
- [AGENTS.md](AGENTS.md) — agent architecture
- [PATH_MIGRATION_GUIDE.md](PATH_MIGRATION_GUIDE.md) — path changes from previous versions
- [TASK.md](TASK.md) — task tracking
- [planning/](planning/README.md) — architecture, implementation notes, roadmap, and technical notes
  - [System Architecture](planning/architecture/system-architecture.md)
  - [Neural Web Roadmap](planning/roadmap/neural-web-roadmap.md)
  - [Consolidated System Fixes](planning/technical-notes/consolidated-system-fixes.md)

## Project Structure

```
WitsV3/
├── agents/              # Agent implementations (control center, orchestrator, specialized)
├── core/                # Core systems (config, LLM interface, memory, schemas, adapters)
├── config/              # Additional YAML configs (personality, ethics, background agent)
├── data/                # Local data (memory files, MCP tool definitions) — personal data gitignored
├── gui/                 # PyQt6 desktop GUI (work in progress)
├── planning/            # Design docs, roadmaps, technical notes
├── scripts/             # Setup and maintenance scripts
├── tests/               # Test suite (142 passing)
├── tools/               # Tool implementations
├── config.yaml          # Main configuration
├── run.py               # Main entry point (CLI + --test self-check)
├── requirements.txt     # Python dependencies (requirements.lock = pinned set)
└── README.md            # This file
```

## Contributing

WitsV3 follows strict development standards:

- **Testing required** — all new features must include tests
- **Async patterns** — all I/O operations must be async
- **Type hints** — full type annotation coverage
- **Documentation** — Google-style docstrings
- **Code quality** — PEP8 with black formatting

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
