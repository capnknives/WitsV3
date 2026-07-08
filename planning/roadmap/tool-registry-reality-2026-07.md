# Tool Registry Reality Check — July 2026

**Created:** July 8, 2026  
**Companion:** [`clutter-catalog-2026-07.md`](clutter-catalog-2026-07.md)  
**Hot path:** WCCA → `LLMDrivenOrchestrator` + auto-discovered `tools/*.py` + optional MCP `mcp_*`

---

## Status legend

| Status | Meaning |
|--------|---------|
| **LIVE** | Used on default hot path (prompts, agents, or web UI) |
| **BLOCKED** | Registered but deliberately excluded from orchestrator |
| **NEVER_PROMPTED** | Discoverable / dumped into AVAILABLE TOOLS, but never guided in ReAct Important bullets or WCCA |
| **DEAD** | Superseded or no real callers — delete candidates |
| **SCRIPT_NOT_TOOL** | File under `tools/` that is not a `BaseTool` |

---

## Discovery rules (from `core/tool_registry.py`)

1. **Builtins** — always: `think`, `calculator` (defined in the registry module).
2. **Explicit file tools** — `read_file`, `write_file`, `list_directory`, `datetime` from `tools/file_tools.py`.
3. **Auto-discover** — every `tools/*.py` except `__*` and `file_tools.py`; register zero-arg `BaseTool` subclasses.
4. **MCP** — *not* file discovery. `MCPToolRegistry` / `/mcp` connect registers `mcp_{name}` wrappers at runtime.
5. **DI** — `run.py` calls `set_dependencies(...)` on tools that define it.

**Orchestrator filters (identical sets):**

- Prompt strip: `_ORCHESTRATOR_TOOL_EXCLUDE` in `llm_driven_orchestrator.py`
- Runtime preflight: `ORCHESTRATOR_BLOCKED_TOOLS` in `orchestrator_tool_helpers.py`
- Members: `{intent_analysis, json_manipulate}`

> Every *non-blocked* tool still appears in the ReAct **AVAILABLE TOOLS** dump. “Prompted” below means explicit Important / routing guidance, not passive listing.

---

## Full table

| Tool name | Source | Discovery | Status | Evidence | Recommendation |
|-----------|--------|-----------|--------|----------|----------------|
| `think` | `core/tool_registry.py` | builtin | LIVE | Fallback / parse path | keep |
| `calculator` | `core/tool_registry.py` | builtin | LIVE | Dump + builtins | keep (prefer over `math_operations`) |
| `read_file` | `file_tools.py` | explicit | LIVE | ReAct Important; doc preflight | keep |
| `write_file` | `file_tools.py` | explicit | LIVE | Save/export + auto-write | keep |
| `list_directory` | `file_tools.py` | explicit | LIVE | ReAct Important | keep |
| `datetime` | `file_tools.py` | explicit | NEVER_PROMPTED | Dump only | keep or add time guidance |
| `web_search` | `web_search_tool.py` | auto | LIVE | Heavy ReAct + WCCA | keep |
| `document_search` | `document_tools.py` | auto | LIVE | Heavy ReAct + WCCA | keep |
| `ingest_documents` | `document_tools.py` | auto | LIVE | ReAct + startup ingest | keep |
| `read_conversation_history` | `conversation_history_tool.py` | auto | LIVE | Save/export flow | keep |
| `analyze_conversation` | `conversation_history_tool.py` | auto | NEVER_PROMPTED | No Important/WCCA | prompt or hide from dump |
| `list_mcp_tools` | `list_mcp_tools.py` | auto | LIVE | MCP Important + `/mcp` | keep |
| `search_mcp_tools` | `mcp_discovery_tool.py` | auto | LIVE | Discovery path | keep |
| `ask_claude` | `ask_claude_tool.py` | auto | LIVE | Escalation UI / key | keep (niche) |
| `diagnose_log_errors` | `self_repair_tools.py` | auto | LIVE | `SelfRepairAgent` | keep; optional dump exclude |
| `apply_code_fix` | `self_repair_tools.py` | auto | LIVE | SelfRepair + coding handlers | keep; optional dump exclude |
| `restart_app` | `self_repair_tools.py` | auto | LIVE | SelfRepair (gated) | keep; optional dump exclude |
| `run_test_suite` | `self_repair_tools.py` | auto | LIVE | **Correction (July 8):** wired into `SelfRepairAgent.run()`'s whole-codebase fallback (no file named, no log errors → run the suite, parse failures) — not unused | keep |
| `python_execute` | `python_execution_tool.py` | auto | NEVER_PROMPTED | Dump only | prompt or demote |
| `math_operations` | `math_tool.py` | auto | NEVER_PROMPTED | Overlaps calculator | prune or prefer calculator |
| `network_control` | `network_control_tool.py` | auto | NEVER_PROMPTED | Auth niche; setup_auth | keep niche; exclude from dump |
| `enhanced_reasoning` | `enhanced_reasoning.py` | auto | NEVER_PROMPTED | DI tests; rare ReAct use | agent-only / exclude dump |
| `neural_web_nlp_extract` | `neural_web_nlp.py` | auto | NEVER_PROMPTED | DI tests | same |
| `neural_web_visualize` | `neural_web_visualization.py` | auto | NEVER_PROMPTED | DI tests | same |
| `intent_analysis` | `intent_analysis_tool.py` | auto | BLOCKED ≈ DEAD | Blocked + WCCA mixin owns intent | **delete** |
| `json_manipulate` | `json_tool.py` | auto | BLOCKED | Blocked; composer string remnant | **delete or unblock** |
| `mcp_*` | dynamic `MCPTool` | MCP attach | LIVE (dynamic) | Prompt + `/mcp` | keep as class |
| — | `ollama_probe.py` | not BaseTool | SCRIPT_NOT_TOOL | HTTP probe script | **move → `scripts/`** |
| — | `mcp_tool_registry.py` | not BaseTool | SCRIPT_NOT_TOOL | Infra helper | keep / move to `core/` |
| — | `enhanced_reasoning_{models,patterns}.py` | not BaseTool | SCRIPT_NOT_TOOL | Support modules | keep |

---

## Counts

| Status | Count | Notes |
|--------|------:|-------|
| LIVE (static) | 16 | Core chat/RAG/MCP/repair agents |
| BLOCKED | 2 | intent + json |
| NEVER_PROMPTED | 10 | Noise in AVAILABLE TOOLS |
| SCRIPT_NOT_TOOL | 4 | Misfiled / support |
| DYNAMIC MCP | N | Runtime-connected |

README’s “22 built-in tools” is **stale** (self-repair + neural + MCP helpers pushed the count higher).

---

## Mental model

```
WCCA intent/routing
        │
        ▼
Orchestrator AVAILABLE TOOLS  ← all non-blocked registry tools
        │
        ├─ Explicit Important bullets → real hot path
        ├─ Dump-only                  → NEVER_PROMPTED (still callable)
        └─ ORCHESTRATOR_BLOCKED       → filtered + preflight reject

Agents (SelfRepair / Coding) ──► get_tool(...) directly
Web /mcp + ask_claude        ──► mcp_* + escalation
```

---

## Cleanup waves

### Wave A — Safe / high leverage

1. **Delete** `tools/intent_analysis_tool.py` (and any exports) — superseded by `wcca_intent_mixin`.
2. **Delete or unblock** `tools/json_tool.py` — if deleting, remove WCCA composer `"data_analysis": "json_manipulate"` remnant.
3. **Move** `tools/ollama_probe.py` → `scripts/ollama_probe.py`.
4. **Exclude agent-owned tools from orchestrator dump:** `diagnose_log_errors`, `apply_code_fix`, `restart_app`, `run_test_suite` (still callable via `get_tool`).
5. ~~**Wire or demote** `run_test_suite`~~ — done; `SelfRepairAgent.run()`'s whole-codebase fallback calls it directly (see correction in the table above).
6. **Update README** tool count / list.

### Wave B — Dump hygiene (optional)

Exclude from default ReAct dump (keep registered for direct/agent use):

- `network_control`
- `enhanced_reasoning`, `neural_web_nlp_extract`, `neural_web_visualize`
- `math_operations` (keep `calculator`)

### Wave C — Prompt guidance (if keeping)

Add short Important bullets for: `python_execute`, `datetime`, `analyze_conversation` — or hide them.

---

## Test coverage notes

| Well covered | Thin |
|--------------|------|
| web_search, documents, file tools, json, math, python_execute, self-repair×4, neural DI, MCP discovery | ask_claude (web), network_control (manual), intent (guardrail only), think/calculator (smoke) |

---

*Next: reconcile Wave A with [`clutter-catalog-2026-07.md`](clutter-catalog-2026-07.md) deletes; re-count tools after any registry exclude changes.*
