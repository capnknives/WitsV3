#!/usr/bin/env python3
"""One-shot Docker sandbox verification (preflight, log line, python_execute)."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import load_config
from core.docker_sandbox import ensure_docker_sandbox_ready
from core.runtime_paths import ensure_runtime_layout, main_log_path
from core.sandbox_runner import run_python_sandboxed, sandbox_mode


async def main() -> int:
    ensure_runtime_layout()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(main_log_path()), encoding="utf-8"),
        ],
    )
    config = load_config()
    logger = logging.getLogger("WitsV3")

    if sandbox_mode(config) != "docker":
        print("SKIP: sandbox_mode is not docker")
        return 0

    ok, detail = await ensure_docker_sandbox_ready(config)
    logger.info("Docker sandbox ready: %s", detail[:200])
    print(f"preflight: {ok} {detail}")
    if not ok:
        return 1

    result = await run_python_sandboxed("print(sum(range(1, 11)))", config=config)
    out = (result.output or "").strip()
    print(f"python_execute: success={result.success} output={out!r}")
    return 0 if result.success and out == "55" else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
