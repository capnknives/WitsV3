#!/usr/bin/env python3
"""Live smoke test for chat_export_a42ee2e0 fixes (real Ollama + WitsV3System).

Run from repo root:
  python scripts/a42ee2e0_live_smoke.py
  python scripts/a42ee2e0_live_smoke.py --quick   # routing/files only, no LLM turns
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from core.runtime_paths import ensure_runtime_layout, exports_dir, sessions_dir, workspace_dir
from core.schemas import ConversationHistory, StreamData

PASS = 0
FAIL = 0
SKIP = 0


def ok(name: str, detail: str = "") -> None:
    global PASS
    PASS += 1
    print(f"  PASS  {name}" + (f" — {detail}" if detail else ""))


def fail(name: str, detail: str) -> None:
    global FAIL
    FAIL += 1
    print(f"  FAIL  {name} — {detail}")


def skip(name: str, detail: str) -> None:
    global SKIP
    SKIP += 1
    print(f"  SKIP  {name} — {detail}")


async def collect_streams(agent, user_input: str, conversation: ConversationHistory, session_id: str):
    """Yield all stream chunks; return (final_text, all_content, tool_names)."""
    parts: list[str] = []
    tool_hits: list[str] = []
    final = ""
    async for stream in agent.run(
        user_input=user_input,
        conversation_history=conversation,
        session_id=session_id,
        user_role="owner",
    ):
        if stream.content:
            parts.append(stream.content)
        if stream.type == "tool_call" and stream.content:
            tool_hits.append(stream.content)
        if stream.type in ("result", "error"):
            final = stream.content or final
    return final, "\n".join(parts), tool_hits


async def operator_checks(system) -> None:
    """July 9 operator UX + memory 3a checks (no LLM)."""
    print("\n=== Operator UX + memory (no LLM) ===")
    backend = system.config.memory_manager.backend
    if backend == "faiss_cpu":
        ok("memory backend", backend)
    else:
        fail("memory backend", f"expected faiss_cpu, got {backend}")

    if system.memory_manager:
        try:
            await system.memory_manager.initialize()
            ok("memory manager init")
        except Exception as e:
            fail("memory manager init", str(e))
    else:
        fail("memory manager", "not configured")

    from core.mcp_health import clear_server_health, get_server_health, record_connect_failure

    clear_server_health()
    record_connect_failure("smoke-test", "simulated connect failure")
    health = get_server_health("smoke-test")
    if health.get("last_error"):
        ok("mcp health tracking")
    else:
        fail("mcp health tracking", "missing last_error")
    clear_server_health()

    from core.tool_metrics import ToolMetricsRecorder

    rec = ToolMetricsRecorder()
    rec.record("smoke_tool", 10.0, success=True)
    if rec.snapshot():
        ok("tool metrics recorder")
    else:
        fail("tool metrics recorder", "empty snapshot")

    if hasattr(system.config.security, "offline_mode"):
        ok("offline_mode config", str(system.config.security.offline_mode))
    else:
        fail("offline_mode config", "missing on SecuritySettings")


async def routing_checks(wcca) -> None:
    print("\n=== Routing (deterministic, no LLM) ===")
    cases = [
        (
            "save debugthisoneplz → orchestrator",
            "save our conversation as debugthisoneplz",
            lambda i: i.get("suggested_response") == "orchestrator"
            and i.get("specialized_agent") != "self_repair",
        ),
        (
            "save a copy → orchestrator",
            "Save a copy of our conversation as importantissues01",
            lambda i: i.get("suggested_response") == "orchestrator",
        ),
        (
            "codebase tour playbook",
            "What can you tell me about your codebase wits?",
            lambda i: i.get("playbook_id") == "codebase_tour",
        ),
        (
            "introspection skips coding agent",
            "No, I want you to actually look at your own files, the code.",
            None,
        ),
    ]
    for label, msg, pred in cases:
        if label.startswith("introspection"):
            agent = await wcca._select_specialized_agent(msg)
            if agent is None:
                ok(label)
            else:
                fail(label, f"got specialized agent {agent}")
            continue
        intent = await wcca._analyze_user_intent(msg, None)
        if pred(intent):
            ok(label, str(intent.get("suggested_response") or intent.get("routing_destination")))
        else:
            fail(label, json.dumps(intent, default=str)[:200])


async def live_turn(
    system,
    session_id: str,
    message: str,
    *,
    max_seconds: float = 180,
) -> tuple[str, str, list[str]]:
    if session_id not in system.session_histories:
        system.session_histories[session_id] = ConversationHistory(session_id=session_id)
    conv = system.session_histories[session_id]
    conv.add_message("user", message)
    root = system.config.runtime_paths.root
    try:
        final, blob, tools = await asyncio.wait_for(
            collect_streams(system.control_center, message, conv, session_id),
            timeout=max_seconds,
        )
    except asyncio.TimeoutError:
        final, blob, tools = "", "TIMEOUT", []

    assistant_text = final or (blob[:4000] if blob else "(no response)")
    conv.add_message("assistant", assistant_text)
    try:
        from core.session_store import persist_session

        persist_session(conv, root)
    except Exception:
        pass
    return final, blob, tools


async def live_checks(system, only: set[str] | None = None) -> None:
    print("\n=== Live LLM turns (Ollama) ===")
    session = f"smoke-{uuid.uuid4().hex[:8]}"
    root = system.config.runtime_paths.root
    steps = only or {
        "sqrt",
        "codebase",
        "pong",
        "edge",
        "save",
        "save2",
        "session",
        "water",
    }

    if "sqrt" in steps:
        print("  … sqrt(75231)")
        final, blob, tools = await live_turn(
            system, session, "what is the square-root of 75231", max_seconds=120
        )
        if "web_search" in blob.lower():
            fail("sqrt no web_search", "web_search appeared in stream")
        elif re.search(r"274\.2\d*", final + blob):
            ok("sqrt answer", final[:80])
        else:
            fail("sqrt answer", f"final={final[:120]!r}")

    if "codebase" in steps:
        print("  … codebase question")
        final, blob, tools = await live_turn(
            system,
            session,
            "What can you tell me about your codebase wits?",
            max_seconds=180,
        )
        if "read_file" in blob or "list_directory" in blob or "README" in blob:
            ok("codebase tools used", "filesystem bootstrap/reads in stream")
        else:
            fail("codebase tools used", "no read_file/list_directory in stream")
        if "BLUEBIRD" in final and "read_file" not in blob:
            fail("codebase no hallucinated secrets", "BLUEBIRD without file reads")
        elif "witwatersrand" in final.lower() or "university of the wits" in final.lower():
            fail("codebase grounded answer", f"off-topic: {final[:100]}")
        elif final.strip():
            ok("codebase response", final[:100])

    if "pong" in steps:
        print("  … pong script write")
        final, blob, tools = await live_turn(
            system,
            session,
            "Create a python script that recreates the game pong in your allowed file area.",
            max_seconds=360,
        )
        pong_paths = list(workspace_dir(root).rglob("pong.py"))
        if pong_paths:
            ok("pong on disk", str(pong_paths[0].relative_to(ROOT)))
        else:
            fail(
                "pong on disk",
                f"no pong.py under {workspace_dir(root)}; final={final[:80]!r}",
            )

    if "edge" in steps:
        print("  … open microsoft edge")
        final, blob, tools = await live_turn(
            system, session, "Please open microsoft edge", max_seconds=120
        )
        mcp_calls = blob.lower().count("list_mcp_tools")
        if mcp_calls >= 5:
            fail("edge no MCP loop", f"list_mcp_tools seen ~{mcp_calls} times")
        elif mcp_calls <= 3 and (
            "can't open" in final.lower()
            or "cannot open" in final.lower()
            or "mcp" in final.lower()
            or "capability" in final.lower()
            or final.strip()
        ):
            ok("edge handled", f"mcp_calls≈{mcp_calls}; {final[:90]}")
        else:
            fail("edge handled", f"mcp_calls={mcp_calls}; final={final[:120]!r}")

    if "save" in steps:
        print("  … save as debugthisoneplz")
        final, blob, tools = await live_turn(
            system, session, "save our conversation as debugthisoneplz", max_seconds=120
        )
        export_path = exports_dir(root) / "debugthisoneplz.txt"
        routed_self_repair = bool(
            re.search(r"Selected self-repair agent|Selected coding agent.*self", blob, re.I)
        )
        msg_count = len(system.session_histories[session].messages)
        min_export = max(40, msg_count * 20)
        if routed_self_repair:
            fail("save not self-repair", "routed to self-repair agent")
        elif export_path.is_file() and export_path.stat().st_size >= min_export:
            ok("debugthisoneplz export", f"{export_path.stat().st_size} bytes")
        else:
            fail(
                "debugthisoneplz export",
                f"exists={export_path.is_file()} size={export_path.stat().st_size if export_path.is_file() else 0}",
            )

    if "save2" in steps:
        print("  … save a copy importantissues01")
        final, blob, tools = await live_turn(
            system,
            session,
            "Save a copy of our conversation as importantissues01",
            max_seconds=120,
        )
        imp_path = exports_dir(root) / "importantissues01.txt"
        msg_count = len(system.session_histories[session].messages)
        min_export = max(40, msg_count * 20)
        if imp_path.is_file() and imp_path.stat().st_size >= min_export:
            ok("importantissues01 export", f"{imp_path.stat().st_size} bytes")
        else:
            fail(
                "importantissues01 export",
                f"exists={imp_path.is_file()} size={imp_path.stat().st_size if imp_path.is_file() else 0}",
            )

    if "session" in steps:
        print("  … session persistence")
        from core.session_store import load_persisted_sessions_into

        reloaded: dict = {}
        n = load_persisted_sessions_into(reloaded, root)
        msg_count = len(reloaded[session].messages) if session in reloaded else 0
        if session in reloaded and msg_count >= 2:
            ok("session persisted", f"{msg_count} messages on disk")
        else:
            fail("session persisted", f"loaded={n} session_msgs={msg_count}")

    if "water" in steps:
        print("  … is water wet (sanity)")
        final, blob, tools = await live_turn(system, session, "Is water wet?", max_seconds=60)
        if final.strip():
            ok("water wet reply", final[:80])
        else:
            fail("water wet reply", "empty")


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="routing only")
    parser.add_argument(
        "--only",
        metavar="STEPS",
        help="comma-separated live steps: sqrt,codebase,pong,edge,save,save2,session,water",
    )
    args = parser.parse_args()

    ensure_runtime_layout()
    exports_dir().mkdir(parents=True, exist_ok=True)
    sessions_dir().mkdir(parents=True, exist_ok=True)
    workspace_dir().mkdir(parents=True, exist_ok=True)

    print("=== a42ee2e0 live smoke ===")
    from run import WitsV3System
    from core.config import load_config

    config = load_config("config.yaml")
    system = WitsV3System(config)
    print("Initializing WitsV3 (Ollama required)...")
    await system.initialize()
    if not system.control_center:
        print("FATAL: control center not initialized")
        return 1

    await operator_checks(system)
    await routing_checks(system.control_center)

    if args.quick:
        print(f"\nDone: {PASS} passed, {FAIL} failed, {SKIP} skipped (quick mode)")
        await system.shutdown()
        return 1 if FAIL else 0

    only = {s.strip() for s in args.only.split(",")} if args.only else None
    await live_checks(system, only=only)

    print(f"\n=== Summary: {PASS} passed, {FAIL} failed, {SKIP} skipped ===")
    await system.shutdown()
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
