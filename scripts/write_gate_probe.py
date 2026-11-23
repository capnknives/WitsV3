"""RT-3b: measure what a real model does when the untrusted-content write gate fires.

Runs real turns through the control center (the same entry point as golden_run) inside a
golden_run sandbox. Web search is patched to a fixed result set, so no search reaches the
network. Scenarios (see write_gate_probe_lib.SCENARIOS):
  A / A2  planted instruction in a search result (write + remember / python_execute)
  B / B2  planted instruction in a notes document (asked about its topic / asked to summarise)
  C       owner asks to search then save, then says "yes, go ahead and save it"
  D       scenario A with security.untrusted_write_gate off (sandbox config only)

Usage:
    python scripts/write_gate_probe.py --sandbox D:/probe-sandbox --runs 3 --out probe.md
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import golden_run  # noqa: E402
import write_gate_probe_lib as lib  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    p = argparse.ArgumentParser(description="Real-model probe of the untrusted write gate.")
    p.add_argument("--sandbox", default="", help="DIR that holds all data for this run")
    p.add_argument("--runs", type=int, default=3, help="runs per scenario (default 3)")
    p.add_argument("--out", default="", help="summary file (default DIR/probe/summary.md)")
    p.add_argument("--only", default="", help="comma-separated scenarios, e.g. A,C")
    p.add_argument("--timeout-seconds", type=float, default=300.0, help="per-turn limit")
    return p


def _set_gate(system, enabled: bool) -> None:
    """Switch the gate on every config object the native loop could read."""
    cc = system.control_center
    for owner in (system, cc, getattr(cc, "native_orchestrator", None)):
        cfg = getattr(owner, "config", None)
        if cfg is not None and getattr(cfg, "security", None) is not None:
            cfg.security.untrusted_write_gate = enabled


async def _segment_ids(system) -> set[str] | None:
    mm = getattr(system, "memory_manager", None)
    if mm is None:
        return None
    try:
        return {s.id for s in await mm.get_recent_memory(limit=1_000_000)}
    except Exception:
        return None


async def _purge_new_segments(system, before: set[str] | None) -> int:
    """Delete memory segments a run added, so one run's reply never feeds the next."""
    after = await _segment_ids(system)
    if before is None or after is None:
        return 0
    removed = 0
    for seg_id in after - before:
        removed += await system.memory_manager.delete_segments({"id": seg_id})
    return removed


async def _turn(system, sandbox, history, message, sid, timeout, spy, spec, tool_msgs):
    """Run one user turn and measure tools, reply, files and store changes."""
    from core.schemas import StreamData

    ignore = {lib.DOC_NAME}
    before = lib.scan_sandbox(sandbox, ignore)["stores"]
    history.add_message("user", message)
    hits0, net0 = spy["fake_hits"], spy["network_attempts"]
    tool_msgs.clear()
    events: list[StreamData] = []
    start = time.perf_counter()

    async def consume():
        async for ev in system.control_center.run(
            message, conversation_history=history, session_id=sid
        ):
            events.append(ev)

    timed_out = False
    try:
        await asyncio.wait_for(consume(), timeout)
    except asyncio.TimeoutError:
        timed_out = True
    wall = time.perf_counter() - start
    # Reason: memory writes may be batched; flush so the store scan sees this turn's writes.
    flush = getattr(getattr(system, "memory_manager", None), "flush", None)
    if flush is not None:
        with contextlib.suppress(Exception):
            await flush()
    after = lib.scan_sandbox(sandbox, ignore)
    reply = lib.reply_text(events)
    history.add_message("assistant", reply or "(no reply)")
    calls = lib.tool_calls(events)
    hits, net = spy["fake_hits"] - hits0, spy["network_attempts"] - net0
    return {
        "message": message,
        **lib.turn_validity(spec, calls, hits, net, tool_msgs),
        "model_tool_messages": list(tool_msgs),
        "fake_search_hits": hits,
        "network_attempts": net,
        "tools": calls,
        "reply": reply,
        "files_written": after["files"],
        "memory_hits": lib.new_store_hits(before, after["stores"]),
        "wall_s": round(wall, 2),
        "timed_out": timed_out,
        "event_types": [ev.type for ev in events],
    }


def _native_llm(system):
    """The LLM interface the native loop calls, whose tool messages are captured."""
    native = getattr(system.control_center, "native_orchestrator", None)
    llm = getattr(native, "llm_interface", None)
    if llm is None or not hasattr(llm, "generate_chat"):
        raise RuntimeError("probe: native orchestrator has no generate_chat to observe")
    return llm


async def run_scenario(system, sandbox: Path, scenario: str, n: int, timeout: float) -> dict:
    """Run one scenario once in a fresh session; write DIR/probe/<scenario>-<n>.json."""
    from core.schemas import ConversationHistory

    spec = lib.SCENARIOS[scenario]
    sid = f"probe-{scenario}-{n}-{uuid.uuid4().hex[:8]}"
    history = ConversationHistory(session_id=sid)
    seg_before = await _segment_ids(system)
    record = {"scenario": scenario, "run": n, "gate": spec["gate"], "session": sid, "turns": []}
    search_tool = lib.find_search_tool(getattr(system, "tool_registry", None))
    llm = _native_llm(system)
    _set_gate(system, spec["gate"])
    try:
        with (
            lib.patched_search(spec["search"], search_tool) as spy,
            lib.capture_tool_messages(llm) as tool_msgs,
        ):
            for message in spec["messages"]:
                turn = await _turn(
                    system, sandbox, history, message, sid, timeout, spy, spec, tool_msgs
                )
                record["turns"].append(turn)
        record["invalid"] = any(t["invalid"] for t in record["turns"])
        record["planted_delivered"] = record["turns"][0]["planted_delivered"]
    finally:
        _set_gate(system, True)
        # Reason: canary files are removed after the whole run, so C's turn 2 sees turn 1's.
        lib.delete_files(sandbox, lib.scan_sandbox(sandbox, {lib.DOC_NAME})["files"])
        record["segments_purged"] = await _purge_new_segments(system, seg_before)
    out = sandbox / "probe"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{scenario}-{n}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


async def plant_document(system, config, sandbox: Path) -> dict:
    """Seed the notes file where golden_run seeds documents, then try to index it.

    Indexing runs before any probe turn, so it never taints a turn; it needs the embedding
    model, so a failure is recorded and read_file can still reach sandbox/files.
    """
    from core import runtime_paths

    for folder in (sandbox / "files", runtime_paths.documents_dir(config.folders.base)):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / lib.DOC_NAME).write_text(lib.DOC_TEXT, encoding="utf-8")
    try:
        result = await system.tool_registry.execute_tool("ingest_documents")
        keys = ("files_scanned", "files_ingested", "files_unchanged", "chunks_added", "errors")
        info = {k: result.get(k) for k in keys} if isinstance(result, dict) else {}
        return {"ingested": isinstance(result, dict) and bool(result.get("success")), **info}
    except Exception as exc:  # recorded, not fatal: read_file still reaches the file
        return {"ingested": False, "error": str(exc)[:300]}


async def main_async(args: argparse.Namespace) -> int:
    """Build the sandboxed system, run every scenario N times, write the summary."""
    from core.config import load_config  # lazy: the PC wrapper patches load_config first

    sandbox = Path(args.sandbox).resolve()
    out = Path(args.out).resolve() if args.out else sandbox / "probe" / "summary.md"
    only = [s for s in args.only.split(",") if s] or list(lib.SCENARIOS)
    unknown = [s for s in only if s not in lib.SCENARIOS]
    if unknown:
        print(f"unknown scenario(s): {unknown}", file=sys.stderr)
        return 2
    config = load_config(str(ROOT / "config.yaml"))
    records: list[dict] = []
    with contextlib.ExitStack() as stack:
        try:
            stack.enter_context(golden_run.enter_sandbox(config, sandbox))
        except golden_run.SandboxError as exc:
            print(f"refusing to start: {exc}", file=sys.stderr)
            return 3
        if config.conversation.front_door != "native":
            print("conversation.front_door is not 'native'; the gate is not on the path")
            return 4
        from run import WitsV3System  # after the sandbox: run.py opens its log at import

        system = WitsV3System(config)
        await system.initialize()
        if {"B", "B2"} & set(only):
            print(f"planted document: {await plant_document(system, config, sandbox)}")
        for scenario in only:
            for n in range(1, args.runs + 1):
                rec = await run_scenario(system, sandbox, scenario, n, args.timeout_seconds)
                records.append(rec)
                tools = [f"{c['name']}:{c['status']}" for t in rec["turns"] for c in t["tools"]]
                flag = f" INVALID (delivered={rec['planted_delivered']})" if rec["invalid"] else ""
                print(f"{scenario} run {n}: {tools}{flag}")
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(
                    lib.render_summary(records, time.strftime("%Y-%m-%d %H:%M")), "utf-8"
                )
    print(f"summary: {out}")
    invalid = [f"{r['scenario']}-{r['run']}" for r in records if r.get("invalid")]
    if invalid:
        print(f"INVALID runs (search not faked or plant not delivered): {invalid}", file=sys.stderr)
        return 5
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; exits 2 before building anything without --sandbox."""
    args = build_parser().parse_args(argv)
    if not args.sandbox:
        print("--sandbox DIR is required: the probe never runs against real data", file=sys.stderr)
        return 2
    if args.runs < 1:
        print("--runs must be at least 1", file=sys.stderr)
        return 2
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
