# Manual test scripts

Standalone smoke-test scripts moved out of the repo root (they cluttered the
top level and `pytest.ini`'s `testpaths = tests` never collected them anyway
— see `planning/roadmap/composer-orchestrator-search-quality-2026-07.md`
Tier 3 #13). Not pytest suites: run directly with `python
scripts/manual_tests/<script>.py`. Several require a live Ollama instance
and/or `config.yaml` to be present.
