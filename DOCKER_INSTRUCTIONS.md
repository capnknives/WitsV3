# Docker notes (parked)

> **Not the recommended way to run WitsV3.** Day-to-day use is local:
> `python run_web.py` (or `run.py`). Docker packaging is explicitly **parked** on the
> roadmap — see [`docs/roadmap/suggested-features-2026-07.md`](docs/roadmap/suggested-features-2026-07.md) §4.

What exists today:

| Artifact | Purpose |
|----------|---------|
| `Dockerfile` | Image for the main app (not actively productized) |
| `Dockerfile.background` | Background agent image |
| `Dockerfile.sandbox` | **Sandbox image** for `security.sandbox_mode: docker` |
| `docker-compose.background.yml` | Compose file for the background agent path |

There is **no** root `docker-compose.yml` for a full stack. Prefer local venv + Ollama.

### Docker sandbox (`security.sandbox_mode: docker`)

When enabled in `config.yaml`, Docker isolates **only** these paths:

1. **`python_execute`** — user script in a temp dir, `--network=none`, ephemeral container
2. **Verified edits** (coding + self-repair) — pytest gate runs inside the container

Wits itself (Web UI, agents, memory, Ollama) still runs on the host. Docker does **not** containerize the full stack.

**Startup**

- `start_web_ui.bat` runs `scripts/ensure_docker_desktop.ps1` before `run_web.py`
- `run.py` / `run_web.py` call `ensure_docker_sandbox_ready()` at init — Wits fails fast if Docker is down
- Log line to confirm: `Docker sandbox ready` in `logs/witsv3.log`

**Verify**

```powershell
# Preflight
.\.venv\Scripts\python.exe -c "import asyncio; from core.config import load_config; from core.docker_sandbox import ensure_docker_sandbox_ready; c=load_config(); print(asyncio.run(ensure_docker_sandbox_ready(c)))"

# Image present
docker images witsv3-sandbox

# Smoke (no Ollama)
python scripts/conversation_task_smoke.py --quick --only sandbox-docker-ready,sandbox-docker-exec
```

**Rebuild sandbox image**

```powershell
docker build -f Dockerfile.sandbox -t witsv3-sandbox .
```

The image installs `requirements.txt` plus pytest so the verified-edit gate can collect and run project tests. If you previously built a pytest-only image, rebuild once so `import pydantic` succeeds inside the container (`ensure_sandbox_image` auto-rebuilds when the health check fails).

**pytest in Docker (read-only mount)**

The project is mounted read-only at `/workspace`. Pytest cache and basetemp are redirected to `/tmp` inside the container so collection does not fail. Tests that **write under the repo** (outside pytest `tmp_path`) may still fail in docker mode — host pytest was the previous behavior. For self-repair, the candidate file is written on the host **before** the container runs; the read-only mount still sees that change.

**Troubleshooting**

1. `Docker sandbox preflight failed` — start Docker Desktop or set `sandbox_mode: "off"`
2. `docker-credential-desktop` not found — ensure `%ProgramFiles%\Docker\Docker\resources\bin` is on PATH (handled in `core/docker_sandbox.py` on Windows)
3. Build fails from a bare shell — use full path to `docker.exe` or run via `start_web_ui.bat`
4. Rebuild with `--no-cache` if the sandbox image layers go stale

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
