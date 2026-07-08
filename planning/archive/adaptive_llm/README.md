# Adaptive LLM system (dormant)

**Status:** Deprecated July 2026. Not used on the normal startup path.

`llm_interface.default_provider` defaults to `ollama`. If set to `adaptive`,
`get_llm_interface()` and `get_enhanced_llm_interface()` now log a warning and
return the standard Ollama interface instead.

## Why deprecated

`core/adaptive_llm_interface.py` routed queries to on-disk neural "modules" that
were never shipped in the revival stack. Smart model routing lives in
`core/model_router.py` and is exposed on `/settings`.

## Source still in repo

| File | Role |
|------|------|
| `core/adaptive_llm_interface.py` | Main adaptive interface (experimental) |
| `core/adaptive_llm_config.py` | Settings dataclasses |
| `core/complexity_analyzer.py` | Query complexity scoring |
| `core/dynamic_module_loader.py` | Module loader |
| `core/semantic_cache.py` | Semantic cache |
| `tests/test_adaptive_llm.py` | Manual/integration smoke (optional) |

Re-enable only for research — do not wire back into production without replacing
the module-loader design with real Ollama model routing.
