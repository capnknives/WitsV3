"""Shared runner for conversation + task smoke scenarios."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from core.runtime_paths import exports_dir
from core.schemas import ConversationHistory, StreamData
from core.smoke_metrics import get_current, scenario_metrics, smoke_metrics_enabled

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS_PATH = Path(__file__).resolve().parent / "smoke_scenarios.yaml"


@dataclass
class SmokeResult:
    scenario_id: str
    tier: str
    passed: bool
    detail: str = ""
    skipped: bool = False


@dataclass
class SmokeRunState:
    results: list[SmokeResult] = field(default_factory=list)
    live_session_id: str = ""
    system: Any = None

    @property
    def pass_count(self) -> int:
        return sum(1 for r in self.results if r.passed and not r.skipped)

    @property
    def fail_count(self) -> int:
        return sum(1 for r in self.results if not r.passed and not r.skipped)

    @property
    def skip_count(self) -> int:
        return sum(1 for r in self.results if r.skipped)


def load_scenarios(path: Path | None = None) -> list[dict[str, Any]]:
    data = yaml.safe_load((path or SCENARIOS_PATH).read_text(encoding="utf-8"))
    return list(data.get("scenarios", []))


def filter_scenarios(
    scenarios: list[dict[str, Any]],
    *,
    tiers: set[str] | None = None,
    only: set[str] | None = None,
    live: bool = False,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for sc in scenarios:
        if only and sc["id"] not in only:
            continue
        if tiers and sc.get("tier") not in tiers:
            continue
        if live and not sc.get("requires_live"):
            continue
        if not live and sc.get("requires_live") and tiers is None and only is None:
            # quick mode skips live-only unless explicitly requested
            continue
        out.append(sc)
    return out


def _record(state: SmokeRunState, scenario_id: str, tier: str, passed: bool, detail: str = "") -> None:
    if smoke_metrics_enabled() and get_current():
        get_current().passed = passed
    state.results.append(SmokeResult(scenario_id=scenario_id, tier=tier, passed=passed, detail=detail))
    label = "PASS" if passed else "FAIL"
    print(f"  {label}  {scenario_id}" + (f" — {detail}" if detail else ""))


def _skip(state: SmokeRunState, scenario_id: str, tier: str, detail: str) -> None:
    state.results.append(
        SmokeResult(scenario_id=scenario_id, tier=tier, passed=False, detail=detail, skipped=True)
    )
    print(f"  SKIP  {scenario_id} — {detail}")


async def collect_streams(agent, user_input: str, conversation: ConversationHistory, session_id: str, **kwargs):
    parts: list[str] = []
    tool_hits: list[str] = []
    final = ""
    stream_types: list[str] = []
    async for stream in agent.run(
        user_input=user_input,
        conversation_history=conversation,
        session_id=session_id,
        **kwargs,
    ):
        if stream.content:
            parts.append(stream.content)
        stream_types.append(stream.type)
        if stream.type == "tool_call" and stream.content:
            tool_hits.append(stream.content)
        if stream.type in ("result", "error"):
            final = stream.content or final
    return final, "\n".join(parts), tool_hits, stream_types


def _build_history(scenario: dict[str, Any], session_id: str) -> ConversationHistory | None:
    hist_spec = scenario.get("history")
    if not hist_spec:
        return None
    conv = ConversationHistory(session_id=session_id)
    for msg in hist_spec:
        conv.add_message(msg["role"], msg["content"])
    return conv


async def run_operator_check(scenario: dict[str, Any], system, state: SmokeRunState) -> None:
    sid = scenario["id"]
    tier = scenario["tier"]
    check = scenario.get("check", "")

    with scenario_metrics(sid, route="operator") if smoke_metrics_enabled() else _null_ctx():
        if check == "memory_backend":
            backend = system.config.memory_manager.backend
            _record(state, sid, tier, backend == scenario.get("expect", "faiss_cpu"), backend)
        elif check == "memory_init":
            ok = False
            if system.memory_manager:
                try:
                    await system.memory_manager.initialize()
                    ok = True
                except Exception as e:
                    _record(state, sid, tier, False, str(e))
                    return
            _record(state, sid, tier, ok)
        elif check == "mcp_health":
            from core.mcp_health import clear_server_health, get_server_health, record_connect_failure

            clear_server_health()
            record_connect_failure("smoke-test", "simulated connect failure")
            health = get_server_health("smoke-test")
            ok = bool(health.get("last_error"))
            clear_server_health()
            _record(state, sid, tier, ok)
        elif check == "tool_metrics":
            from core.tool_metrics import ToolMetricsRecorder

            rec = ToolMetricsRecorder()
            rec.record("smoke_tool", 10.0, success=True)
            _record(state, sid, tier, bool(rec.snapshot()))
        elif check == "offline_mode_config":
            ok = hasattr(system.config.security, "offline_mode")
            _record(state, sid, tier, ok)
        elif check == "docker_sandbox_ready":
            from core.sandbox_runner import sandbox_mode

            if sandbox_mode(system.config) != "docker":
                _skip(state, sid, tier, "sandbox_mode is not docker")
                return
            from core.docker_sandbox import ensure_docker_sandbox_ready

            ok, detail = await ensure_docker_sandbox_ready(system.config)
            _record(state, sid, tier, ok, detail[:120])
        elif check == "docker_sandbox_exec":
            from core.sandbox_runner import run_python_sandboxed, sandbox_mode

            if sandbox_mode(system.config) != "docker":
                _skip(state, sid, tier, "sandbox_mode is not docker")
                return
            code = scenario.get("code", "print(sum(range(1, 11)))")
            expected = str(scenario.get("expect_output", "55")).strip()
            result = await run_python_sandboxed(code, config=system.config, timeout=60.0)
            out = (result.output or "").strip()
            ok = result.success and out == expected
            detail = out if ok else f"success={result.success} out={out!r} err={result.error!r}"
            _record(state, sid, tier, ok, detail)
        else:
            _record(state, sid, tier, False, f"unknown operator check: {check}")


class _null_ctx:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False

    async def __aenter__(self):
        return None

    async def __aexit__(self, *args):
        return False


async def run_routing_scenario(scenario: dict[str, Any], wcca, state: SmokeRunState) -> None:
    sid = scenario["id"]
    tier = scenario["tier"]
    message = scenario.get("message", "")
    check = scenario.get("check", "")
    assertions = scenario.get("assert", {})
    session_id = "smoke-routing"

    with scenario_metrics(sid, route="routing") if smoke_metrics_enabled() else _null_ctx():
        if check == "specialist_none":
            agent = await wcca._select_specialized_agent(message)
            _record(state, sid, tier, agent is None, str(agent) if agent else "")
            return

        if check == "orchestrator_guard":
            history = _build_history(scenario, session_id)
            from agents import routing_classifier as rc

            doc_inventory = scenario.get("doc_inventory")
            if doc_inventory:
                original = wcca._get_document_inventory

                async def _fake_inventory():
                    return doc_inventory

                wcca._get_document_inventory = _fake_inventory  # type: ignore[method-assign]
                try:
                    routing_msg = rc.follow_up_routing_message(message, history)
                    needs = await wcca._requires_orchestrator_for_input(routing_msg)
                finally:
                    wcca._get_document_inventory = original  # type: ignore[method-assign]
            else:
                routing_msg = rc.follow_up_routing_message(message, history)
                needs = await wcca._requires_orchestrator_for_input(routing_msg)
            _record(state, sid, tier, needs, f"guard={needs} routing_msg={routing_msg[:60]}")
            return

        if check == "guest_no_specialist":
            wcca._request_user_role = "guest"
            try:
                intent = await wcca._analyze_user_intent(message, None)
            finally:
                wcca._request_user_role = "owner"
            agent = intent.get("specialized_agent")
            _record(state, sid, tier, agent is None, f"agent={agent}")
            return

        history = _build_history(scenario, session_id)
        role = scenario.get("role", "owner")
        prev_role = getattr(wcca, "_request_user_role", "owner")
        wcca._request_user_role = role

        last_task_route = scenario.get("last_task_route")
        if last_task_route:
            wcca._last_task_context[session_id] = {
                "route": last_task_route,
                "goal": scenario.get("last_task_goal", ""),
            }

        doc_inventory = scenario.get("doc_inventory")
        if doc_inventory:
            original = wcca._get_document_inventory

            async def _fake_inventory():
                return doc_inventory or {}

            wcca._get_document_inventory = _fake_inventory  # type: ignore[method-assign]
            try:
                intent = await wcca._analyze_user_intent(message, history, session_id=session_id)
            finally:
                wcca._get_document_inventory = original  # type: ignore[method-assign]
                wcca._request_user_role = prev_role
                if last_task_route:
                    wcca._last_task_context.pop(session_id, None)
        else:
            try:
                intent = await wcca._analyze_user_intent(message, history, session_id=session_id)
            finally:
                wcca._request_user_role = prev_role
                if last_task_route:
                    wcca._last_task_context.pop(session_id, None)

        passed = _evaluate_intent_assertions(intent, assertions)
        detail = ""
        if not passed:
            detail = str({k: intent.get(k) for k in sorted(intent.keys()) if not k.startswith("_")})[:200]
        _record(state, sid, tier, passed, detail)


def _evaluate_intent_assertions(intent: dict[str, Any], assertions: dict[str, Any]) -> bool:
    for key, expected in assertions.items():
        if key in (
            "llm_calls_max",
            "react_iterations_max",
            "wall_ms_max",
        ):
            continue
        if key == "specialized_agent_not":
            if intent.get("specialized_agent") == expected:
                return False
        elif key == "preferred_tool":
            if intent.get("preferred_tool") != expected:
                return False
        elif key == "playbook_id":
            if intent.get("playbook_id") != expected:
                return False
        elif intent.get(key) != expected:
            return False
    return True


def _evaluate_performance_budgets(assertions: dict[str, Any]) -> tuple[bool, str]:
    """Compare smoke metrics against optional budget caps."""
    if not smoke_metrics_enabled():
        return True, ""
    metrics = get_current()
    if metrics is None:
        return True, ""

    if "llm_calls_max" in assertions and metrics.llm_calls > assertions["llm_calls_max"]:
        return False, f"llm_calls={metrics.llm_calls} > max {assertions['llm_calls_max']}"
    if "react_iterations_max" in assertions and metrics.react_iterations > assertions["react_iterations_max"]:
        return (
            False,
            f"react_iterations={metrics.react_iterations} > max {assertions['react_iterations_max']}",
        )
    if "wall_ms_max" in assertions and metrics.wall_ms > assertions["wall_ms_max"]:
        return False, f"wall_ms={metrics.wall_ms:.0f} > max {assertions['wall_ms_max']}"
    return True, ""


async def run_multiturn_scenario(scenario: dict[str, Any], system, state: SmokeRunState) -> None:
    """Run a scripted multi-turn live conversation."""
    sid = scenario["id"]
    tier = scenario["tier"]
    turns = scenario.get("turns", [])
    timeout = float(scenario.get("timeout", 180))
    role = scenario.get("role", "owner")

    if not state.live_session_id:
        import uuid

        state.live_session_id = f"smoke-{uuid.uuid4().hex[:8]}"
    session_id = state.live_session_id

    with scenario_metrics(sid, route=tier) if smoke_metrics_enabled() else _null_ctx():
        if session_id not in system.session_histories:
            system.session_histories[session_id] = ConversationHistory(session_id=session_id)
        conv = system.session_histories[session_id]

        import asyncio

        for idx, turn in enumerate(turns):
            message = turn.get("message", "")
            turn_assert = turn.get("assert", {})
            conv.add_message("user", message)
            try:
                final, blob, tools, stream_types = await asyncio.wait_for(
                    collect_streams(
                        system.control_center,
                        message,
                        conv,
                        session_id,
                        user_role=role,
                    ),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                _record(state, sid, tier, False, f"TIMEOUT turn {idx + 1}")
                return

            assistant_text = final or (blob[:4000] if blob else "(no response)")
            conv.add_message("assistant", assistant_text)

            if turn.get("expect_clarification"):
                pending = getattr(system.control_center, "_pending_clarifications", {})
                if session_id not in pending:
                    _record(state, sid, tier, False, f"turn {idx + 1}: expected pending clarification")
                    return

            if turn_assert:
                passed, detail = _evaluate_live_assertions(
                    turn_assert,
                    final=final,
                    blob=blob,
                    tools=tools,
                    stream_types=stream_types,
                    session_id=session_id,
                    system=system,
                    message=message,
                )
                if not passed:
                    _record(state, sid, tier, False, f"turn {idx + 1}: {detail}")
                    return

            if turn.get("check_intent"):
                intent = await system.control_center._analyze_user_intent(message, conv, session_id)
                if not _evaluate_intent_assertions(intent, turn["check_intent"]):
                    _record(state, sid, tier, False, f"turn {idx + 1} intent: {intent}")
                    return

        try:
            from core.session_store import persist_session

            persist_session(conv, system.config.runtime_paths.root)
        except Exception:
            pass

        perf_ok, perf_detail = _evaluate_performance_budgets(scenario.get("assert", {}))
        if not perf_ok:
            _record(state, sid, tier, False, perf_detail)
            return
        _record(state, sid, tier, True)


async def run_live_scenario(scenario: dict[str, Any], system, state: SmokeRunState) -> None:
    sid = scenario["id"]
    tier = scenario["tier"]
    message = scenario.get("message", "")
    assertions = scenario.get("assert", {})
    timeout = float(scenario.get("timeout", 180))
    check = scenario.get("check", "")
    role = scenario.get("role", "owner")

    if not state.live_session_id:
        import uuid

        state.live_session_id = f"smoke-{uuid.uuid4().hex[:8]}"

    session_id = state.live_session_id

    with scenario_metrics(sid, route=tier) if smoke_metrics_enabled() else _null_ctx():
        if check == "session_persisted":
            from core.session_store import load_persisted_sessions_into

            root = system.config.runtime_paths.root
            reloaded: dict = {}
            load_persisted_sessions_into(reloaded, root)
            msg_count = len(reloaded[session_id].messages) if session_id in reloaded else 0
            min_msgs = int(scenario.get("min_messages", 2))
            _record(
                state,
                sid,
                tier,
                session_id in reloaded and msg_count >= min_msgs,
                f"msgs={msg_count}",
            )
            return

        if session_id not in system.session_histories:
            system.session_histories[session_id] = ConversationHistory(session_id=session_id)
        conv = system.session_histories[session_id]
        conv.add_message("user", message)

        import asyncio

        try:
            final, blob, tools, stream_types = await asyncio.wait_for(
                collect_streams(
                    system.control_center,
                    message,
                    conv,
                    session_id,
                    user_role=role,
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            _record(state, sid, tier, False, "TIMEOUT")
            return

        assistant_text = final or (blob[:4000] if blob else "(no response)")
        conv.add_message("assistant", assistant_text)
        try:
            from core.session_store import persist_session

            persist_session(conv, system.config.runtime_paths.root)
        except Exception:
            pass

        passed, detail = _evaluate_live_assertions(
            assertions, final=final, blob=blob, tools=tools, stream_types=stream_types,
            session_id=session_id, system=system, message=message,
        )
        if passed:
            perf_ok, perf_detail = _evaluate_performance_budgets(assertions)
            if not perf_ok:
                passed = False
                detail = perf_detail
        if smoke_metrics_enabled() and get_current():
            m = get_current()
            m.passed = passed
            for t in tools:
                m.record_tool(t)
        _record(state, sid, tier, passed, detail)


def _evaluate_live_assertions(
    assertions: dict[str, Any],
    *,
    final: str,
    blob: str,
    tools: list[str],
    stream_types: list[str],
    session_id: str,
    system: Any,
    message: str,
) -> tuple[bool, str]:
    combined = (final or "") + blob
    lowered = combined.lower()

    if assertions.get("no_orchestrator") and "Delegating to orchestrator" in blob:
        return False, "routed to orchestrator"

    if assertions.get("no_tool_calls") and "tool_call" in stream_types:
        return False, "tool_call in stream"

    if "stream_contains" in assertions:
        if assertions["stream_contains"].lower() not in lowered:
            return False, f"missing {assertions['stream_contains']}"

    for needle in assertions.get("stream_contains_any", []):
        if needle.lower() in lowered:
            break
    else:
        if assertions.get("stream_contains_any"):
            return False, "none of stream_contains_any found"

    for needle in assertions.get("stream_not_contains_any", []):
        if needle.lower() in lowered:
            return False, f"found forbidden {needle}"

    if "stream_not_contains" in assertions:
        if assertions["stream_not_contains"].lower() in lowered:
            return False, f"found {assertions['stream_not_contains']}"

    cond = assertions.get("stream_not_contains_with_unless")
    if cond:
        for forbidden, unless in cond.items():
            if forbidden in combined and unless.lower() not in lowered:
                return False, f"{forbidden} without {unless}"

    if "regex" in assertions:
        if not re.search(assertions["regex"], combined):
            return False, f"regex miss: {final[:80]!r}"

    if assertions.get("final_non_empty") and not (final or "").strip():
        return False, "empty final"

    if "export_file" in assertions:
        root = system.config.runtime_paths.root
        path = exports_dir(root) / assertions["export_file"]
        msg_count = len(system.session_histories.get(session_id, ConversationHistory(session_id=session_id)).messages)
        min_export = max(40, msg_count * 20)
        if not path.is_file() or path.stat().st_size < min_export:
            return False, f"export size={path.stat().st_size if path.is_file() else 0}"

    if "mcp_list_tools_max" in assertions:
        count = lowered.count("list_mcp_tools")
        if count > assertions["mcp_list_tools_max"]:
            return False, f"mcp_calls={count}"

    if assertions.get("stream_type") == "result" and "result" not in stream_types:
        return False, "no result stream"

    return True, (final or "")[:80]


async def run_scenarios(
    system,
    scenarios: list[dict[str, Any]],
    *,
    live: bool = False,
) -> SmokeRunState:
    state = SmokeRunState(system=system)
    wcca = system.control_center

    by_tier: dict[str, list[dict]] = {}
    for sc in scenarios:
        by_tier.setdefault(sc.get("tier", "other"), []).append(sc)

    if "operator" in by_tier:
        print("\n=== Operator (no LLM) ===")
        for sc in by_tier["operator"]:
            await run_operator_check(sc, system, state)

    if wcca and "routing" in by_tier:
        print("\n=== Routing (deterministic) ===")
        for sc in by_tier["routing"]:
            await run_routing_scenario(sc, wcca, state)

    if wcca and "guest" in by_tier:
        print("\n=== Guest routing ===")
        for sc in by_tier["guest"]:
            await run_routing_scenario(sc, wcca, state)

    if live and wcca:
        live_tiers = ("direct", "orchestrator", "multiturn", "performance")
        for tier in live_tiers:
            if tier not in by_tier:
                continue
            print(f"\n=== Live: {tier} ===")
            for sc in by_tier[tier]:
                if sc.get("turns"):
                    await run_multiturn_scenario(sc, system, state)
                else:
                    await run_live_scenario(sc, system, state)

    return state
