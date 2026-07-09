#!/usr/bin/env python3
"""A/B compare json_react vs ollama_native on hot-path smoke scenarios."""

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
from core.smoke_metrics import get_report, reset_report, smoke_metrics_enabled
from scripts.smoke_harness import filter_scenarios, load_scenarios, run_scenarios


AB_SCENARIOS = ("orch-sqrt", "orch-codebase", "orch-edge")


async def run_mode(mode: str, scenario_ids: set[str]) -> dict[str, dict]:
    from run import WitsV3System
    from core.config import load_config

    config = load_config("config.yaml")
    config.orchestrator.tool_calling_mode = mode
    system = WitsV3System(config)
    await system.initialize()
    if not system.control_center:
        raise RuntimeError("control center not initialized")

    scenarios = [s for s in load_scenarios() if s["id"] in scenario_ids]
    reset_report()
    state = await run_scenarios(system, scenarios, live=True)
    await system.shutdown()

    by_id = {entry["scenario_id"]: entry for entry in get_report()}
    return {
        "mode": mode,
        "pass_count": state.pass_count,
        "fail_count": state.fail_count,
        "metrics": by_id,
    }


def _gate_passes(json_run: dict, native_run: dict) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if native_run["fail_count"] > json_run["fail_count"]:
        reasons.append(
            f"native failures ({native_run['fail_count']}) > json ({json_run['fail_count']})"
        )

    for sid in AB_SCENARIOS:
        jm = json_run["metrics"].get(sid, {})
        nm = native_run["metrics"].get(sid, {})
        if nm.get("llm_calls", 99) > jm.get("llm_calls", 0) + 1:
            reasons.append(f"{sid}: native llm_calls ({nm.get('llm_calls')}) worse than json")

    return len(reasons) == 0, reasons


async def main() -> int:
    parser = argparse.ArgumentParser(description="A/B smoke: json_react vs ollama_native")
    parser.add_argument(
        "--scenarios",
        default=",".join(AB_SCENARIOS),
        help="comma-separated scenario ids",
    )
    parser.add_argument("--report", choices=("text", "json"), default="text")
    args = parser.parse_args()

    os.environ["WITS_SMOKE_METRICS"] = "1"
    ensure_runtime_layout()
    exports_dir().mkdir(parents=True, exist_ok=True)
    sessions_dir().mkdir(parents=True, exist_ok=True)
    workspace_dir().mkdir(parents=True, exist_ok=True)

    ids = {s.strip() for s in args.scenarios.split(",") if s.strip()}
    print("=== smoke_ab_compare ===")
    print(f"Scenarios: {', '.join(sorted(ids))}")

    json_run = await run_mode("json_react", ids)
    native_run = await run_mode("ollama_native", ids)
    gate_ok, gate_reasons = _gate_passes(json_run, native_run)

    result = {
        "json_react": json_run,
        "ollama_native": native_run,
        "gate_passes": gate_ok,
        "gate_reasons": gate_reasons,
    }

    if args.report == "json":
        print(json.dumps(result, indent=2))
    else:
        print("\n--- json_react ---")
        for sid, m in sorted(json_run["metrics"].items()):
            print(f"  {sid}: llm={m.get('llm_calls')} react={m.get('react_iterations')} ms={m.get('wall_ms')}")
        print("\n--- ollama_native ---")
        for sid, m in sorted(native_run["metrics"].items()):
            print(f"  {sid}: llm={m.get('llm_calls')} react={m.get('react_iterations')} ms={m.get('wall_ms')}")
        print(f"\nGate: {'PASS' if gate_ok else 'FAIL'}")
        for r in gate_reasons:
            print(f"  - {r}")

    return 0 if gate_ok else 1


if __name__ == "__main__":
    if not smoke_metrics_enabled():
        os.environ["WITS_SMOKE_METRICS"] = "1"
    raise SystemExit(asyncio.run(main()))
