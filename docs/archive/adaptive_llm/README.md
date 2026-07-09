# Adaptive LLM system (dormant)

**Status:** Deprecated July 2026. Not used on the normal startup path.

`llm_interface.default_provider` defaults to `ollama`. If set to `adaptive`,
`get_llm_interface()` and `get_enhanced_llm_interface()` now log a warning and
return the standard Ollama interface instead.

## Why deprecated

`core/adaptive_llm_interface.py` routed queries to on-disk neural "modules" that
were never shipped in the revival stack. Smart model routing lives in
`core/model_router.py` and is exposed on `/settings`.

## Source archived here (July 8, 2026)

The modules were moved out of the live `core/` package into this folder so they
no longer sat on the import path. They were preserved for research revival only.

**July 8 2026 (Phase 2b):** The Python tree under `core/` was pruned from the
working copy (~10 files). This README remains as the index.

| Former path (under this folder) | Role |
|---------------------------------|------|
| `core/adaptive_llm_interface.py` | Main adaptive interface (experimental) |
| `core/adaptive_llm_config.py` | Settings dataclasses |
| `core/complexity_analyzer.py` | Query complexity scoring |
| `core/dynamic_module_loader.py` | Module loader (the only `torch` consumer) |
| `core/semantic_cache.py` | Semantic cache |
| `core/adaptive/` | Placeholder tokenizer / response generator / performance tracker |

The former `tests/test_adaptive_llm.py` integration smoke and its `tests/config.yaml`
were deleted (recoverable via git history) since their subjects are archived.

## Recover the snapshot

Full code tree is preserved in git at tag **`archive-pre-prune-2b-2026-07`**:

```powershell
git ls-tree -r --name-only archive-pre-prune-2b-2026-07 docs/archive/adaptive_llm/

# Restore locally (optional)
git checkout archive-pre-prune-2b-2026-07 -- docs/archive/adaptive_llm/core/
```

Re-enable only for research — do not wire back into production without replacing
the module-loader design with real Ollama model routing. If you revive these,
restore the root `torch.py` shim expectation or add a real `torch` dependency.
