# Manual test scripts

Standalone smoke scripts — **not** collected by pytest. Run with:

```bash
python scripts/manual_tests/<script>.py
```

Many need a live Ollama instance and a filled-in `config.yaml` / `.env`.

Canonical product install and day-to-day usage: root [`README.md`](../../README.md).  
Formal suite: `pytest tests/ -q --no-cov`.
