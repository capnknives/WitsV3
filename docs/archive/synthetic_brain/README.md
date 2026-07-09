# Synthetic brain / cognitive cluster (dormant)

**Status:** Archived July 2026. Not used on the normal startup path.

The live stack uses `core/memory_manager.py` with the `basic` backend (default),
WCCA intent routing, and the ReAct orchestrator. These modules were an
experimental Phase 1 "synthetic brain" layer that was never wired into
`run.py` / `run_web.py`.

## Why archived

- No production importer — only standalone tests and cross-imports within this cluster
- APIs drifted from live `KnowledgeGraph` / `MemoryManager` (wrong constructors, stub procedural memory)
- Superseded by the July 2026 revival: document RAG, model routing, orchestrator synthesis guard

## Source archived here (July 8, 2026)

| Archived path (under this folder) | Role |
|-----------------------------------|------|
| `core/memory_handler.py` | Unified memory facade (working/episodic/semantic/procedural) |
| `core/cognitive_architecture.py` | Perception → reasoning → action loop |
| `core/synthetic_brain_stubs.py` | Stub LLM/KG/WM for integration tests |
| `core/synthetic_brain_integration.py` | Compatibility wrappers |
| `tests/test_memory_handler.py` | Unit tests (removed from pytest collection) |
| `tests/test_cognitive_architecture.py` | Unit tests (removed from pytest collection) |

Historical docs: [`SYNTHETIC_BRAIN.md`](../historical-docs/SYNTHETIC_BRAIN.md) and siblings under `docs/archive/historical-docs/`.

Re-enable only for research — do not wire back into production without reconciling
APIs with the current `MemoryManager` and orchestrator stack.
