"""Ollama model status and pull helpers for the web settings UI."""

from __future__ import annotations

import re
from typing import Any

import httpx

_MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")

_ROLE_LABELS = {
    "default": "Default",
    "orchestrator": "Orchestrator",
    "embedding": "Embeddings",
    "routing_trivial": "Routing · casual",
    "routing_code": "Routing · code",
    "routing_complex": "Routing · complex",
}


def validate_model_name(model: str) -> str:
    """Return trimmed model name or raise ValueError."""
    name = (model or "").strip()
    if not name or not _MODEL_NAME_RE.match(name):
        raise ValueError("invalid model name")
    return name


def model_is_available(name: str, available: set[str]) -> bool:
    """True if Ollama reports this model (exact or base name match)."""
    if name in available:
        return True
    base = name.split(":")[0]
    if base in available:
        return True
    return any(av.split(":")[0] == base for av in available)


def collect_configured_models(config: Any) -> list[dict[str, str]]:
    """Unique configured Ollama models with role labels for the status panel."""
    ollama = config.ollama_settings
    mr = config.model_routing
    entries: list[tuple[str, str]] = [
        ("default", ollama.default_model),
        ("orchestrator", ollama.orchestrator_model),
        ("embedding", ollama.embedding_model),
    ]
    if mr.enabled:
        entries.extend(
            [
                ("routing_trivial", mr.trivial_model),
                ("routing_code", mr.code_model),
                ("routing_complex", mr.complex_model),
            ]
        )
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for role, name in entries:
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(
            {
                "role": role,
                "label": _ROLE_LABELS.get(role, role),
                "name": name,
            }
        )
    return out


async def probe_ollama(base_url: str, timeout: float = 5.0) -> tuple[bool, set[str]]:
    """Return (reachable, available_model_names). Empty set when unreachable."""
    url = base_url.rstrip("/") + "/api/tags"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
    except Exception:
        return False, set()

    names: set[str] = set()
    for item in data.get("models", []):
        name = item.get("name") or item.get("model") or ""
        if not name:
            continue
        names.add(name)
        names.add(name.split(":")[0])
    return True, names


async def build_ollama_status(config: Any) -> dict[str, Any]:
    """Structured status for configured vs installed models."""
    base_url = config.ollama_settings.url
    reachable, available = await probe_ollama(base_url)
    configured = collect_configured_models(config)
    models = []
    for entry in configured:
        installed = reachable and model_is_available(entry["name"], available)
        models.append({**entry, "installed": installed})
    return {
        "url": base_url,
        "reachable": reachable,
        "available_models": sorted(available) if reachable else [],
        "configured_models": models,
        "all_installed": reachable and all(m["installed"] for m in models),
    }


async def pull_ollama_model(base_url: str, model: str, timeout: float = 600.0) -> dict[str, Any]:
    """Pull a model via Ollama POST /api/pull (non-streaming)."""
    name = validate_model_name(model)
    url = base_url.rstrip("/") + "/api/pull"
    payload = {"name": name, "stream": False}
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
    except httpx.TimeoutException as e:
        raise TimeoutError(f"pull timed out for {name}") from e
    except httpx.HTTPStatusError as e:
        detail = e.response.text[:500] if e.response is not None else str(e)
        raise RuntimeError(detail or f"pull failed for {name}") from e
    except Exception as e:
        raise RuntimeError(str(e)) from e

    status = (data.get("status") or "").lower()
    if status and status not in ("success", "complete", "completed"):
        raise RuntimeError(data.get("status") or f"pull did not complete for {name}")
    return {"model": name, "status": data.get("status") or "success", "detail": data}
