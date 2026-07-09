"""
ReAct tool guardrails, observation formatting, and save-to-file helpers.

Extracted from BaseOrchestratorAgent to keep the orchestrator module under
the 500-line project limit while preserving test harness compatibility.
"""

from __future__ import annotations

import re
from collections.abc import AsyncGenerator
from typing import Any

from core.runtime_paths import exports_subpath, upgrade_runtime_path
from core.schemas import ConversationHistory, StreamData

from agents.orchestrator_codebase import OrchestratorCodebaseMixin
from agents.orchestrator_observations import OrchestratorObservationsMixin
from agents.orchestrator_preflight import OrchestratorPreflightMixin
from agents.orchestrator_synthesis import OrchestratorSynthesisMixin
from agents.routing_classifier import FILE_SAVE_SIGNALS


class OrchestratorToolHelpersMixin(
    OrchestratorSynthesisMixin,
    OrchestratorObservationsMixin,
    OrchestratorPreflightMixin,
    OrchestratorCodebaseMixin,
):
    """Mixin: save/export helpers; synthesis and observations in sibling mixins."""

    @staticmethod
    def _has_read_history_observation(observations: list[str]) -> bool:
        prefix = "Tool read_conversation_history result:"
        return any(o.startswith(prefix) for o in observations)

    @staticmethod
    def _save_file_path_from_goal(goal: str) -> str | None:
        """Extract a target path from save/export phrasing (e.g. exports/chat.txt)."""
        if not goal:
            return None
        patterns_with_ext = (
            r"(?:as|to|into)\s+([^\s?\"']+\.(?:txt|md|log|json))",
            r"\b([\w./-]+\.(?:txt|md|log|json))\b",
        )
        for pattern in patterns_with_ext:
            match = re.search(pattern, goal, re.IGNORECASE)
            if match:
                return upgrade_runtime_path(match.group(1).replace("\\", "/"))

        extless = re.search(
            r"(?:as|to|into)\s+([\w][\w.-]*)",
            goal,
            re.IGNORECASE,
        )
        if extless:
            name = extless.group(1).strip().rstrip(".")
            if name and "/" not in name and "\\" not in name:
                return exports_subpath(f"{name}.txt")
        return None

    @staticmethod
    def _save_export_min_bytes(state: dict[str, Any]) -> int:
        """Minimum plausible export size for the current session."""
        conv = state.get("conversation_history")
        if conv is not None and getattr(conv, "messages", None):
            return max(40, len(conv.messages) * 20)
        return 40

    @staticmethod
    def _save_already_verified(state: dict[str, Any]) -> bool:
        """True when a save goal already produced a non-trivial export on disk."""
        goal = state.get("goal", "")
        if not OrchestratorToolHelpersMixin._goal_saves_conversation(goal):
            return False
        path = OrchestratorToolHelpersMixin._save_file_path_from_goal(goal)
        if not path:
            return False
        min_bytes = OrchestratorToolHelpersMixin._save_export_min_bytes(state)
        for obs in reversed(state.get("observations", [])):
            if "write_file" not in obs or OrchestratorToolHelpersMixin._observation_indicates_failure(
                obs
            ):
                continue
            if OrchestratorToolHelpersMixin._write_result_verified(
                path, obs, min_bytes=min_bytes
            ):
                return True
        return False

    @staticmethod
    def _write_result_verified(
        file_path: str, write_obs: str, *, min_bytes: int = 1
    ) -> bool:
        """True when write_file observation indicates a non-empty file on disk."""
        if not write_obs or "failed" in write_obs.lower() or write_obs.startswith("Blocked "):
            return False
        if "Successfully" not in write_obs and "written to" not in write_obs.lower():
            return False
        from core.safe_code_editor import resolve_within_project

        try:
            resolved = resolve_within_project(file_path)
            return resolved.is_file() and resolved.stat().st_size >= min_bytes
        except (OSError, PermissionError):
            return False

    async def _auto_write_saved_conversation(
        self,
        file_path: str,
        state: dict[str, Any],
        session_id: str | None,
    ) -> AsyncGenerator[StreamData, None]:
        """Write session transcript after read_conversation_history for save goals."""
        yield self.stream_action(
            f"Auto-saving conversation to {file_path} (read_conversation_history complete)"
        )
        try:
            write_result = await self._call_tool("write_file", {"file_path": file_path}, state)
            write_obs = self._format_tool_observation("write_file", write_result)
        except Exception as e:
            write_obs = f"Tool write_file failed: {e}"
            yield self.stream_error(f"Auto-save failed: {e}")

        state["observations"].append(write_obs)
        yield self.stream_observation(write_obs)
        await self.store_memory(
            content=write_obs,
            segment_type="OBSERVATION",
            importance=0.8,
            metadata={"tool_name": "write_file", "session_id": session_id, "auto_save": True},
        )

        if not self._observation_indicates_failure(write_obs):
            min_bytes = self._save_export_min_bytes(state)
            verified = self._write_result_verified(file_path, write_obs, min_bytes=min_bytes)
            if not verified:
                err = (
                    f"Auto-save failed: {file_path} is missing or empty after write_file. "
                    "Conversation was not saved."
                )
                yield self.stream_error(err)
                state["observations"].append(f"Tool write_file failed: {err}")
                return
            final = f"Saved conversation log to {file_path}."
            state["completed"] = True
            state["final_answer"] = final
            yield self.stream_result(final)
            await self.store_memory(
                content=f"Final Answer: {final}",
                segment_type="FINAL_ANSWER",
                importance=1.0,
                metadata={"session_id": session_id},
            )

    @staticmethod
    def _tool_result_is_failure(result: Any) -> bool:
        if not isinstance(result, dict):
            return False
        if result.get("success") is False:
            return True
        if result.get("status") == "error":
            return True
        return bool(result.get("error"))

    @staticmethod
    def _observation_indicates_failure(observation: str) -> bool:
        lowered = observation.lower()
        return (
            (observation.startswith("Tool ") and " failed:" in observation)
            or "(search failed:" in lowered
            or observation.startswith("Blocked ")
        )

    def _record_tool_failure(
        self, tool_name: str, tool_args: dict[str, Any], state: dict[str, Any]
    ) -> None:
        sig = self._tool_call_signature(tool_name, tool_args)
        repeat = state.setdefault("tool_repeat_failures", {})
        repeat[sig] = repeat.get(sig, 0) + 1
        total = state.setdefault("tool_total_failures", {})
        total[tool_name] = total.get(tool_name, 0) + 1

    async def _prepare_tool_args(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """Inject session context and resolve save-to-file bodies before execute."""
        args = dict(tool_args)
        conversation_history = state.get("conversation_history")

        if tool_name in ("read_conversation_history", "analyze_conversation"):
            if conversation_history is not None:
                args.setdefault("conversation_history", conversation_history)
            if tool_name == "read_conversation_history" and self._goal_saves_conversation(
                state.get("goal", "")
            ):
                args.setdefault("max_messages", 0)

        if tool_name == "write_file":
            goal = state.get("goal", "")
            if self._goal_saves_conversation(goal) or self._should_inject_conversation_content(
                goal, state.get("observations", [])
            ):
                args.pop("content", None)
            content = args.get("content")
            if not content or not str(content).strip():
                from_obs = self._conversation_text_from_observations(state.get("observations", []))
                if from_obs:
                    args["content"] = from_obs
                elif conversation_history is not None and self._should_inject_conversation_content(
                    state.get("goal", ""),
                    state.get("observations", []),
                ):
                    args["content"] = self._format_conversation_for_file(conversation_history)
            if not args.get("file_path"):
                path = self._save_file_path_from_goal(state.get("goal", ""))
                if path:
                    args["file_path"] = path
            if args.get("file_path"):
                root = self.config.runtime_paths.root if getattr(self, "config", None) else "var"
                fp = str(args["file_path"]).replace("\\", "/")
                if self._should_inject_conversation_content(
                    state.get("goal", ""), state.get("observations", [])
                ):
                    if "/" not in fp and not fp.startswith(f"{root}/"):
                        basename = fp if fp.endswith((".txt", ".md", ".log", ".json")) else f"{fp}.txt"
                        fp = exports_subpath(basename, root)
                args["file_path"] = upgrade_runtime_path(fp, root)

        if tool_name == "calculator":
            args = self._normalize_calculator_args(args)

        if tool_name in (
            "guest_audit_summary",
            "guest_accounts_list",
            "guest_set_age_band",
            "guest_user_profile_summary",
        ):
            args.setdefault("user_role", state.get("user_role", "owner"))

        if tool_name == "web_search":
            args.setdefault("user_role", state.get("user_role", "owner"))

        return args

    @staticmethod
    def _has_web_search_observation(observations: list[str]) -> bool:
        return any("web_search results" in obs for obs in observations)

    @staticmethod
    def _goal_is_web_lookup(goal: str) -> bool:
        """True when GOAL asks for an online lookup / report (not user documents)."""
        lowered = goal.lower()
        signals = (
            "look up",
            "look it up",
            "search for",
            "search the web",
            "report on",
            "tell me about",
            "what is",
            "who is",
            "give me a small report",
            "give me a report",
            "find out",
        )
        if any(sig in lowered for sig in signals):
            return True
        if "?" in goal and any(
            w in lowered for w in ("game", "news", "weather", "price", "who won", "who died")
        ):
            return True
        return False

    @staticmethod
    def _goal_saves_conversation(goal: str) -> bool:
        """True when the user wants chat/story content written to disk."""
        lowered = goal.lower()
        return any(sig in lowered for sig in FILE_SAVE_SIGNALS)

    @staticmethod
    def _should_inject_conversation_content(goal: str, observations: list[str]) -> bool:
        if OrchestratorToolHelpersMixin._goal_saves_conversation(goal):
            return True
        if OrchestratorToolHelpersMixin._has_read_history_observation(observations):
            return True
        lowered = goal.lower()
        return bool(
            re.search(r"save\s+(a\s+)?(copy|log)|export\s+(this\s+)?(chat|conversation)", lowered)
        )

    @staticmethod
    def _extract_calculator_expression(goal: str) -> str:
        """Pull a calculator-safe expression from natural-language math goals."""
        lowered = goal.lower()
        m = re.search(r"square[\s-]?root\s+of\s+([\d.,]+)", lowered)
        if m:
            return f"sqrt({m.group(1).replace(',', '')})"
        m = re.search(r"\bsqrt\s*\(?\s*([\d.,]+)\s*\)?", lowered)
        if m:
            return f"sqrt({m.group(1).replace(',', '')})"
        m = re.search(r"what(?:'s| is)\s+([\d.,]+)\s*([\+\-\*\/\^])\s*([\d.,]+)", lowered)
        if m:
            return f"{m.group(1).replace(',', '')}{m.group(2)}{m.group(3).replace(',', '')}"
        nums = re.findall(r"[\d.,]+", goal)
        if nums and re.search(r"\b(sqrt|square[\s-]?root|calculate|compute)\b", lowered):
            return f"sqrt({nums[-1].replace(',', '')})"
        return goal.strip()

    @staticmethod
    def _normalize_calculator_args(args: dict[str, Any]) -> dict[str, Any]:
        """Map common LLM calculator arg shapes to a single expression string."""
        normalized = dict(args)
        expr = str(normalized.get("expression") or normalized.get("query") or "").strip()
        if not expr:
            op = str(normalized.get("operation") or "").lower()
            value = normalized.get("value")
            if op in ("square_root", "sqrt") and value is not None:
                expr = f"sqrt({value})"
            elif op and value is not None:
                expr = f"{value}"
        if expr.startswith("sqrt(") or "sqrt(" in expr:
            normalized["expression"] = expr
        elif expr:
            normalized["expression"] = expr
        return normalized

    @staticmethod
    def _goal_is_pure_math(goal: str) -> bool:
        lowered = goal.lower()
        if not re.search(r"\b(sqrt|square[\s-]?root|calculate|compute|math|\+|\-|\*|/)\b", lowered):
            return False
        if re.search(r"\b(search|news|weather|who|when|where|web)\b", lowered):
            return False
        return bool(re.search(r"\d", goal))

    @staticmethod
    def _format_conversation_for_file(
        conversation_history: ConversationHistory,
        max_messages: int = 0,
    ) -> str:
        """Format session messages for write_file content."""
        if not conversation_history or not conversation_history.messages:
            return "Conversation history is empty."

        messages = conversation_history.messages
        if max_messages > 0:
            messages = messages[-max_messages:]

        lines = []
        for msg in messages:
            lines.append(f"{msg.role.upper()}: {msg.content}")
        return "\n\n".join(lines)

    @staticmethod
    def _conversation_text_from_observations(observations: list[str]) -> str | None:
        """Pull formatted transcript from a prior read_conversation_history observation."""
        prefix = "Tool read_conversation_history result:"
        for obs in reversed(observations):
            if obs.startswith(prefix):
                text = obs[len(prefix) :].strip()
                if text and text not in (
                    "Starting new conversation.",
                    "Conversation history is empty.",
                ):
                    return text
        return None
