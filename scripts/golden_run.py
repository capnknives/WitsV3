"""Run the golden turn set against the live system and append a results table.

Usage:
    python scripts/golden_run.py --runs 3 --tag chat --out docs/benchmarks/golden.md
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.golden import GoldenCase, evaluate, load_cases, tools_seen  # noqa: E402

try:  # C1a may not be merged yet
    from core import turn_metrics
except ImportError:
    turn_metrics = None


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    p = argparse.ArgumentParser(description="Run the WitsV3 golden turn set.")
    p.add_argument("--runs", type=int, default=1, help="runs per case (default 1)")
    p.add_argument("--only", default="", help="comma-separated case ids")
    p.add_argument("--tag", default="", help="only cases with this tag")
    p.add_argument("--cases", default=str(ROOT / "tests/golden/turns.yaml"))
    p.add_argument("--out", default=str(ROOT / "docs/benchmarks/golden.md"))
    return p


async def run_once(
    control_center, case: GoldenCase
) -> tuple[bool, float | None, list[str], list[str]]:
    """Run one case; return (passed, first_result_ms, tools seen, unobserved)."""
    turn = None
    if turn_metrics is not None:
        turn = turn_metrics.start_turn("golden", None, len(case.message))
    start = time.perf_counter()
    first_result: float | None = None
    events = []
    try:
        history = None
        if case.history:
            from core.schemas import ConversationHistory

            history = ConversationHistory(session_id="golden")
            for turn_ in case.history:
                history.add_message(turn_["role"], turn_["content"])
            history.add_message("user", case.message)
        async for ev in control_center.run(case.message, conversation_history=history):
            events.append(ev)
            if first_result is None and ev.type in ("result", "error"):
                first_result = (time.perf_counter() - start) * 1000
    finally:
        if turn is not None:
            errored = any(e.type == "error" for e in events)
            await turn_metrics.finish_turn(turn, "error" if errored else "ok")
    names = [c["name"] for c in turn.tool_calls] if turn is not None else None
    res = evaluate(case, events, names)
    return (
        res.passed,
        first_result,
        names if names is not None else tools_seen(events),
        res.unobserved,
    )


async def main_async(args: argparse.Namespace) -> int:
    """Run selected cases and write the table; return the exit code."""
    from core.config import load_config  # live system, imported lazily
    from run import WitsV3System

    cases = load_cases(args.cases)
    only = {s for s in args.only.split(",") if s}
    cases = [
        c for c in cases if (not only or c.id in only) and (not args.tag or args.tag in c.tags)
    ]
    system = WitsV3System(load_config())
    await system.initialize()
    control_center = system.control_center
    rows, failed_all = [], False
    for case in cases:
        results = [await run_once(control_center, case) for _ in range(args.runs)]
        passes = sum(r[0] for r in results)
        times = [r[1] for r in results if r[1] is not None]
        med = f"{statistics.median(times):.0f}" if times else "-"
        tools = sorted({t for r in results for t in r[2]})
        unobs = sorted({u for r in results for u in r[3]})
        rows.append(
            f"| {case.id} | {passes}/{args.runs} | {med} | {', '.join(tools) or '-'} "
            f"| {', '.join(unobs) or '-'} |"
        )
        failed_all = failed_all or passes == 0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M")
    header = "| case | pass | median first_result_ms | tools seen | unobserved |\n|---|---|---|---|---|\n"
    with out.open("a", encoding="utf-8") as fh:
        fh.write(f"\n## Golden run {stamp}\n\n{header}" + "\n".join(rows) + "\n")
    return 1 if failed_all else 0


def main() -> int:
    """CLI entry point."""
    return asyncio.run(main_async(build_parser().parse_args()))


if __name__ == "__main__":
    sys.exit(main())
