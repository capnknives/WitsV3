# Docker notes (parked)

> **Not the recommended way to run WitsV3.** Day-to-day use is local:
> `python run_web.py` (or `run.py`). Docker packaging is explicitly **parked** on the
> roadmap — see [`docs/roadmap/suggested-features-2026-07.md`](docs/roadmap/suggested-features-2026-07.md) §4.

What exists for experimental / background use:

| Artifact | Purpose |
|----------|---------|
| `Dockerfile` | Image for the main app (not actively productized) |
| `Dockerfile.background` | Background agent image |
| `Dockerfile.sandbox` | Optional sandbox image for `security.sandbox_mode: docker` |
| `docker-compose.background.yml` | Compose file for the background agent path |

There is **no** root `docker-compose.yml` for a full stack. Prefer local venv + Ollama.

### Background agent (optional)

```bash
DOCKER_BUILDKIT=1 docker build -f Dockerfile.background -t witsv3-background:latest .
docker compose -f docker-compose.background.yml up -d
```

Daily self-repair does **not** require Docker: `run_web.py` / `run.py` schedule it
in-process when `self_repair.daily_schedule_enabled` is true.

### Troubleshooting

1. Rebuild with `--no-cache` if dependency layers go stale  
2. Keep `requirements.lock` aligned with the Python version you use  
3. Mount a volume over `/app/data` if you need persistence across container restarts  
