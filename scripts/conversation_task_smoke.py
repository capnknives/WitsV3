#!/usr/bin/env python3
"""Full-spread conversational + task smoke test for WitsV3.

Run from repo root:
  python scripts/conversation_task_smoke.py --quick     # operator + routing (no Ollama)
  python scripts/conversation_task_smoke.py --live        # full spread (requires Ollama)
  python scripts/conversation_task_smoke.py --live --only orch-sqrt,orch-codebase
  python scripts/conversation_task_smoke.py --live --metrics --tool-mode ollama_native
  python scripts/smoke_ab_compare.py --report json
  python scripts/conversation_task_smoke.py --quick --report json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from core.runtime_paths import ensure_runtime_layout, exports_dir, sessions_dir, workspace_dir
from core.smoke_metrics import dump_report_json, reset_report, smoke_metrics_enabled
from scripts.smoke_harness import filter_scenarios, load_scenarios, run_scenarios


async def main() -> int:
    parser = argparse.ArgumentParser(description="WitsV3 conversation + task smoke test")
    parser.add_argument("--quick", action="store_true", help="operator + routing only (no LLM turns)")
    parser.add_argument("--live", action="store_true", help="include live Ollama scenarios")
    parser.add_argument("--only", metavar="IDS", help="comma-separated scenario ids")
    parser.add_argument(
        "--report",
        choices=("text", "json"),
        default="text",
        help="output format for metrics report",
    )
    parser.add_argument(
        "--metrics",
        action="store_true",
        help="enable WITS_SMOKE_METRICS instrumentation (also set env WITS_SMOKE_METRICS=1)",
    )
    parser.add_argument(
        "--tool-mode",
        choices=("json_react", "ollama_native"),
        help="override orchestrator.tool_calling_mode for this run",
    )
    args = parser.parse_args()

    if args.metrics:
        os.environ["WITS_SMOKE_METRICS"] = "1"

    ensure_runtime_layout()
    exports_dir().mkdir(parents=True, exist_ok=True)
    sessions_dir().mkdir(parents=True, exist_ok=True)
    workspace_dir().mkdir(parents=True, exist_ok=True)

    only = {s.strip() for s in args.only.split(",")} if args.only else None
    scenarios = load_scenarios()
    if only:
        scenarios = [s for s in scenarios if s["id"] in only]
    elif args.quick:
        scenarios = filter_scenarios(scenarios, live=False)

    print("=== conversation_task_smoke ===")
    reset_report()

    from run import WitsV3System
    from core.config import load_config

    config = load_config("config.yaml")
    if args.tool_mode:
        config.orchestrator.tool_calling_mode = args.tool_mode
        print(f"Tool mode override: {args.tool_mode}")
    system = WitsV3System(config)
    print("Initializing WitsV3...")
    await system.initialize()
    if not system.control_center:
        print("FATAL: control center not initialized")
        return 1

    live = (not args.quick) and (args.live or not args.only)
    state = await run_scenarios(system, scenarios, live=live)

    print(
        f"\n=== Summary: {state.pass_count} passed, "
        f"{state.fail_count} failed, {state.skip_count} skipped ==="
    )

    if smoke_metrics_enabled() and args.report == "json":
        print(dump_report_json())
    elif smoke_metrics_enabled():
        for entry in json.loads(dump_report_json()):
            print(f"  metrics {entry['scenario_id']}: llm={entry['llm_calls']} "
                  f"react={entry['react_iterations']} ms={entry['wall_ms']}")

    await system.shutdown()
    return 1 if state.fail_count else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
