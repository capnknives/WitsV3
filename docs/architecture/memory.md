---
title: "WitsV3 Memory Architecture"
created: "2026-07-09"
last_updated: "2026-07-09"
status: "reference"
---

# WitsV3 Memory Architecture

Single source for how durable state works in WitsV3 (July 2026).  
Install/run context: [`README.md`](../../README.md).

## Storage layers

| Layer | Location | Purpose |
|-------|----------|---------|
| **Episodic segments** | `var/data/wits_memory.json` (+ FAISS index when `backend: faiss_cpu`) | Embeddings + metadata for semantic search |
| **Sessions** | `var/sessions/*.json` | Full chat transcripts for the Web UI |
| **Knowledge log** | `var/data/knowledge_log.json` | Recurring errors + durable owner facts |
| **Documents** | `var/documents/` | RAG corpus (hybrid BM25 + vector via `document_search`) |
| **Guest profiles** | `var/data/guest_profiles/` | Isolated tester facts (no global memory writes) |

## Default backend (Phase 3a)

`config.yaml` sets `memory_manager.backend: faiss_cpu` after validation in
`tests/core/test_faiss_memory_backend.py`. Roll back to `basic` if FAISS or
`faiss-cpu` causes issues on your machine.

**Live validation (July 9, 2026):** `scripts/a42ee2e0_live_smoke.py --quick`
passed on RTX 3070 stack — FAISS initialized 226 segments with `nomic-embed-text`
(no rollback required).

Embeddings use Ollama `nomic-embed-text` (768-dim). Auto-pruning runs on add
for both `basic` and FAISS backends when segment count exceeds
`max_memory_segments`.

## Hot-path behavior

1. **Orchestrator** searches global memory for the goal, plus **session-filtered**
   memory when `session_id` is present (`agents/base_orchestrator_agent.py`).
2. **Pre-compaction flush** (owner only) writes `CONVERSATION_FLUSH` segments before
   history window drops old turns (`core/conversation_compaction.py`).
3. **Guests** skip global memory writes; they use guest tools and profiles only.
4. **Self-repair** records recurring errors into the knowledge log.
5. **Fact promotion:** segments with `importance >= 0.9` and `USER_FACT` /
   `metadata.remember: true`, plus heuristic owner-path extraction
   (`core/fact_extraction.py`) after chat turns.

## Dormant / research paths

| Component | Status |
|-----------|--------|
| `core/knowledge_graph.py` | Not wired on hot path |
| `core/working_memory.py` | In-process only; tests/archive |
| Neural web (`backend: neural`) | **Research-only** — tools gated unless backend is `neural`; see [`neural-web-roadmap.md`](../roadmap/neural-web-roadmap.md) |
| Supabase sync | Optional / parked |

## Owner tools

- `knowledge_log_add_fact`, `knowledge_log_list_facts`, etc. (see tool registry)
- Memory browser in Web UI: search, recent, gated prune

## What this is not (yet)

WitsV3 has **durable episodic + semantic memory** suitable for a personal assistant.
Automatic continuous learning from every turn is limited to heuristic owner-path
fact promotion; deeper learning remains in the Phase 3b+ backlog
([`suggested-features-2026-07.md`](../roadmap/suggested-features-2026-07.md)).
