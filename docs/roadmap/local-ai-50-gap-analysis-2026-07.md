# Top 50 Local AI Features — WitsV3 Gap Analysis

**Created:** July 8, 2026  
**Source doc:** `FEATURE_IDEAS/Top Local AI System Features.docx`  
**Compared against:** `suggested-features-2026-07.md`, `revival-2026-07.md`, `guest-tester-access-2026-07.md`, `neural-web-roadmap.md`

Legend: ✅ Has · 🟡 Partial · ❌ Gap · ➖ N/A (hardware/OS — out of WitsV3 scope)

---

## 1. Core hardware & processing (1–6)

| # | Feature | Status | WitsV3 today | Gap / notes |
|---|---------|--------|--------------|-------------|
| 1 | Dedicated NPU (40+ TOPS) | ➖ | — | Consumer hardware; WitsV3 runs on whatever host provides Ollama |
| 2 | Unified memory 32GB+ | ➖ | — | Host requirement; README recommends ~8 GB VRAM for models |
| 3 | Hybrid CPU/GPU/NPU routing | 🟡 | Ollama owns GPU; `model_router.py` routes by task complexity, not chip type | No NPU path; no explicit GPU layer controls in app |
| 4 | Hardware security enclaves | ➖ | — | OS/firmware (Pluton, Secure Enclave) |
| 5 | ARM/Metal API optimizations | 🟡 | Ollama/MLX on Mac is environmental | App does not select MLX vs CUDA |
| 6 | Air-gapped sovereignty | 🟡 | Local-first; Ollama + optional keyless DDG search | `web_search` and MCP can reach network; no “offline-only” mode flag → **Phase 2.6** |

---

## 2. Foundation models & inference (7–13)

| # | Feature | Status | WitsV3 today | Gap / notes |
|---|---------|--------|--------------|-------------|
| 7 | GGUF format support | 🟡 | Via Ollama model pulls (GGUF under the hood) | No in-app GGUF management or quant picker |
| 8 | Daemon CLI inference (Ollama) | ✅ | `ollama_settings.url`, all agents use `OllamaInterface` | Shipped |
| 9 | GUI management layer | 🟡 | Web `/settings` for models; Ollama pull/status panel (Phase 2.1, July 9) | Quick win shipped |
| 10 | MoE base architectures | 🟡 | Can run MoE models if pulled in Ollama | No MoE-aware routing |
| 11 | Native multimodal backbones | ❌ | Text-only orchestration path | No vision/audio in chat |
| 12 | Multi-token prediction (MTP) | ➖ | Ollama/runtime concern | — |
| 13 | Advanced quantization (SpinQuant etc.) | 🟡 | Archived `adaptive_llm` had quant stubs | Dormant; Ollama handles quant at pull time |

---

## 3. Agentic orchestration & MCP (14–20)

| # | Feature | Status | WitsV3 today | Gap / notes |
|---|---------|--------|--------------|-------------|
| 14 | MCP client integration | ✅ | `core/mcp_adapter.py`, `tools/mcp_tool.py`, `/mcp` UI | Shipped; `mcp_connect_on_startup: false` by default |
| 15 | Windows ODR (agent registry) | ❌ | Custom registry search + local config | Windows-only OS feature |
| 16 | SKILL.md / reusable skill packages | 🟡 | Cursor skills external; no WitsV3 skill installer | Could adopt SKILL.md for orchestrator playbooks |
| 17 | Multi-agent collaboration | 🟡 | WCCA → orchestrator **or** specialist (coding, self-repair, book) | No parallel agent swarm or inter-agent debate |
| 18 | Dynamic cross-provider routing | 🟡 | `model_router.py` (local models); `ask_claude` escalation with UI approval | No automatic cloud fallback; Claude is opt-in |
| 19 | Strict tool call validation | ✅ | `tool_registry.validate_tool_call`, schema unwrap fix | Shipped |
| 20 | Execution telemetry / observability | 🟡 | SSE `StreamData`; `core/metrics.py` in background agent only | No OpenTelemetry, trace IDs, or tool analytics dashboard |

---

## 4. Memory & cognitive persistence (21–27)

| # | Feature | Status | WitsV3 today | Gap / notes |
|---|---------|--------|--------------|-------------|
| 21 | Tri-layer memory (working / episodic / semantic) | 🟡 | Single `MemoryManager` segment store; `config/wits_core.yaml` aspirational; `working_memory.py` + KG exist but not on default path | Layers not enforced in live `run_web.py` |
| 22 | Pre-compaction memory flushes | ✅ | `maybe_flush_conversation_memory` before agent runs | Shipped Phase 1.4 (July 8) |
| 23 | Autonomous “dreaming” / consolidation | ❌ | Synthetic brain archived | Explicitly out of scope unless revived |
| 24 | Frozen prompt memory snapshots | 🟡 | Personality + guest profiles load at session start | No session-frozen MEMORY.md injection pattern |
| 25 | Active memory sub-agent retrieval | 🟡 | `memory_manager.search_memory` in orchestrator context; document inventory in WCCA | Not a blocking pre-turn retrieval agent |
| 26 | Action-sensitive memory triggers | ❌ | — | No temporal/trigger-based memory rules |
| 27 | Unified multi-tenant memory DB | 🟡 | Guest profiles + audit isolated; owner memory global | No per-user memory isolation for owner; Supabase optional |

---

## 5. RAG & knowledge synthesis (28–33)

| # | Feature | Status | WitsV3 today | Gap / notes |
|---|---------|--------|--------------|-------------|
| 28 | GraphRAG / knowledge-graph retrieval | 🟡 | `core/knowledge_graph.py` + neural web; **not wired to document RAG** | GraphRAG pipeline missing |
| 29 | Local dense embeddings | ✅ | `nomic-embed-text` via Ollama on ingest/search | Shipped |
| 30 | Agentic retry + drift-guarded reformulation | 🟡 | Orchestrator can retry tools; synthesis guard on final answer | No dedicated RAG query reformulation loop |
| 31 | Hybrid lexical + semantic fusion | ✅ | BM25 + vector fuse in `document_search` | Shipped Phase 1.3 (July 8) |
| 32 | Incremental RAG indexing (hashing) | ✅ | SHA-256 per file in `document_tools.py` | Shipped |
| 33 | Evidence sufficiency scoring | 🟡 | Synthesis guard checks term overlap with doc/web observations | No explicit “insufficient evidence → refuse” rubric |

---

## 6. Voice, ambient & physical (34–39)

| # | Feature | Status | WitsV3 today | Gap / notes |
|---|---------|--------|--------------|-------------|
| 34 | Local GPU STT (Whisper) | ❌ | — | Not planned in current roadmaps |
| 35 | Neural TTS (Piper) | ❌ | — | — |
| 36 | Streaming pipeline + VAD | ❌ | SSE text streaming only | — |
| 37 | Wake-word recognition | ❌ | — | — |
| 38 | Smart home (Home Assistant) | ❌ | — | — |
| 39 | Digital persona emulation | 🟡 | Personality questionnaire + guest profiles; book agent | No “colleague-skill” lifelog ingestion |

---

## 7. Security, sandboxing & guardrails (40–45)

| # | Feature | Status | WitsV3 today | Gap / notes |
|---|---------|--------|--------------|-------------|
| 40 | Hardware microVM sandboxes | ❌ | `python_execution_tool` restricted; pytest gate on code edits | No Firecracker/gVisor |
| 41 | Sovereign execution brokers | ❌ | — | Research-grade; overkill for personal LAN |
| 42 | Zero-trust ephemeral filesystems | 🟡 | Verified edit reverts on test fail | Not full ephemeral sandbox |
| 43 | Gateway callback approvals | ✅ | Ask-Claude approval card; owner-only shutdown; guest tool allowlist | Shipped for high-impact paths |
| 44 | Input prompt-injection defenses | 🟡 | Guest content policy; orchestrator context hygiene | No dedicated injection classifier |
| 45 | Output schema + toxicity filtering | 🟡 | Pydantic tool schemas; guest content bands | No PII redaction layer on owner path |

---

## 8. Continuous learning & fine-tuning (46–50)

| # | Feature | Status | WitsV3 today | Gap / notes |
|---|---------|--------|--------------|-------------|
| 46 | QLoRA / PEFT fine-tuning | ❌ | — | Out of scope unless requirements change |
| 47 | Memory-optimized training kernels (Unsloth) | ❌ | — | — |
| 48 | Synthetic data generation pipelines | ❌ | — | — |
| 49 | Target-aware data sampling | ❌ | — | — |
| 50 | Unstructured model pruning | ❌ | Memory **segment** pruning only | Not model weight pruning |

---

## Scorecard

| Category | ✅ | 🟡 | ❌ | ➖ |
|----------|---:|---:|---:|---:|
| Hardware (1–6) | 0 | 3 | 0 | 3 |
| Models & inference (7–13) | 1 | 5 | 1 | 1 |
| Orchestration & MCP (14–20) | 2 | 4 | 1 | 0 |
| Memory (21–27) | 0 | 6 | 2 | 0 |
| RAG (28–33) | 2 | 3 | 1 | 0 |
| Voice & ambient (34–39) | 0 | 1 | 5 | 0 |
| Security (40–45) | 1 | 4 | 2 | 0 |
| Fine-tuning (46–50) | 0 | 0 | 5 | 0 |
| **Total** | **6** | **26** | **17** | **4** |

**Interpretation:** WitsV3 is strong on **local orchestration, MCP, document RAG core, tool validation, guest safety, and verified code edits**. It is weak on **voice/ambient, fine-tuning, GraphRAG/hybrid search, formal memory layers, and enterprise-grade sandboxing** — which matches its identity as a personal LAN assistant, not a full “AI PC OS.”

---

## Alignment with existing roadmaps

| Existing item | Maps to feature # | Notes |
|---------------|-------------------|-------|
| Guest access MVP (Phase 3–4) | 27, 43–45 | ✅ Shipped July 8 |
| Conversation-history-aware intent | 21–22 | ✅ Shipped Phase 1.1 |
| Multi-session chat history | 21, UI | ✅ Shipped Phase 1.6 |
| Memory browser | 21 | ✅ Shipped |
| Hybrid RAG / reranking | 28, 31, 33 | ✅ Shipped Phase 1.3 |
| Neural web | 21, 28 | Parked as research; KG exists |
| MCP health dashboard | 14, 20 | **→ Phase 2.2** (next) |
| Ollama pull/status helper | 8–9 | ✅ Shipped Phase 2.1 (July 9) |
| Streaming tool progress | 20 | **→ Phase 2.4** |
| Clutter / file splits | — | Hygiene, not feature gaps |

**Revival backlog:** closed. **Phase 2.2** (MCP health panel) is the next canonical item.

---

## Recommended product stance

Do **not** chase all 50 features. The source doc describes a **full sovereign AI OS**; WitsV3 is a **personal orchestration stack** (Richard + family testers on LAN). Prioritize gaps that compound daily use:

1. **Trust & grounding** — RAG quality, memory across sessions, intent routing  
2. **Household UX** — guest admin, multi-session, simpler model management  
3. **Safety without enterprise weight** — content policy completion, injection hardening lite  
4. **Park** — voice, Home Assistant, fine-tuning, microVM sandboxes, Windows ODR  

Forward phases live in [`suggested-features-2026-07.md`](suggested-features-2026-07.md) § “Post–Top-50 roadmap.”
