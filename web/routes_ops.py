"""Operator UX routes: tool metrics, scheduled tasks, offline flag (Phase 2.3–2.6)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from core.config import save_local_overrides
from core.tool_metrics import tool_metrics
from web.schemas import OfflineModeUpdate

logger = logging.getLogger("WitsV3.WebUI")


def register_ops_routes(app: FastAPI, system) -> None:
    """Register operator observability routes."""

    @app.get("/api/metrics/tools")
    async def metrics_tools():
        return {"tools": tool_metrics.snapshot()}

    @app.get("/api/tasks")
    async def list_scheduled_tasks():
        return {"tasks": _load_background_tasks(system)}

    @app.get("/api/security/offline")
    async def get_offline_mode():
        return {"offline_mode": system.config.security.offline_mode}

    @app.post("/api/security/offline")
    async def set_offline_mode(request: Request, body: OfflineModeUpdate):
        if getattr(request.state, "auth_role", None) != "owner":
            return JSONResponse({"detail": "Only the owner can change offline mode."}, status_code=403)
        system.config.security.offline_mode = body.offline_mode
        save_local_overrides({"security": {"offline_mode": body.offline_mode}})
        logger.info("Offline mode set to %s via web UI", body.offline_mode)
        return {"offline_mode": body.offline_mode}


def _load_background_tasks(system) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    bg_path = Path("config/background_agent.yaml")
    if bg_path.exists():
        try:
            data = yaml.safe_load(bg_path.read_text(encoding="utf-8")) or {}
            for name, spec in (data.get("tasks") or {}).items():
                if not isinstance(spec, dict):
                    continue
                tasks.append(
                    {
                        "id": name,
                        "source": "background_agent",
                        "enabled": bool(spec.get("enabled")),
                        "schedule": spec.get("schedule"),
                        "description": spec.get("description", ""),
                    }
                )
        except Exception as e:
            logger.warning("Could not parse background_agent.yaml: %s", e)

    sr = system.config.self_repair
    tasks.append(
        {
            "id": "daily_self_repair",
            "source": "run.py / run_web.py",
            "enabled": bool(sr.daily_schedule_enabled and sr.enabled),
            "schedule": sr.daily_schedule_cron,
            "description": "In-process self-repair scan (logs → verified fixes)",
        }
    )
    return tasks
