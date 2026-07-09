"""Preflight guardrails before orchestrator tool execution."""

from __future__ import annotations

import json
from typing import Any

from core.runtime_paths import exports_subpath

from agents.routing_classifier import needs_codebase_intro


class OrchestratorPreflightMixin:
    """Mixin: block doomed, repeat, or unsafe tool calls before execution."""

    REPEAT_TOOL_FAILURE_LIMIT = 2
    TOOL_TOTAL_FAILURE_LIMIT = 3
    ORCHESTRATOR_BLOCKED_TOOLS = frozenset({"intent_analysis", "json_manipulate"})

    def _guest_allowed_tools(self, state: dict[str, Any] | None = None) -> frozenset[str]:
        from core.guest_access import guest_tools_for_age_band, resolve_effective_role, tools_for_role

        react_state = state or getattr(self, "_react_state_for_tools", None) or {}
        user_role = react_state.get("user_role", "owner")
        guest_profile = react_state.get("guest_profile")
        effective = resolve_effective_role(user_role, guest_profile)
        role_tools = tools_for_role(effective, getattr(self, "config", None))
        if role_tools is not None:
            return role_tools
        if user_role == "guest":
            age_band = react_state.get("guest_age_band", "teen")
            return guest_tools_for_age_band(age_band, getattr(self, "config", None))
        return frozenset()

    @staticmethod
    def _has_ingested_documents(state: dict[str, Any]) -> bool:
        """True when USER DOCUMENTS lists at least one ingested file."""
        ctx = state.get("documents_context", "")
        return bool(ctx) and "No user documents are currently ingested" not in ctx

    @staticmethod
    def _tool_call_signature(tool_name: str, tool_args: dict[str, Any]) -> str:
        return f"{tool_name}:{json.dumps(tool_args, sort_keys=True, default=str)}"

    def _preflight_tool_call(
        self, tool_name: str, tool_args: dict[str, Any], state: dict[str, Any]
    ) -> str | None:
        """Return a block message to skip doomed/repeat tool calls, or None to proceed."""
        if getattr(self, "config", None) and self.config.security.offline_mode:
            if tool_name == "web_search" or tool_name.startswith("mcp_"):
                return (
                    f"Blocked {tool_name}: offline mode is enabled — "
                    "web search and MCP tools are disabled."
                )
        if state.get("user_role") == "owner":
            from core.injection_guard import check_tool_injection

            injection = check_tool_injection(tool_name, tool_args)
            if injection:
                return injection
        if state.get("user_role") == "guest" and tool_name not in self._guest_allowed_tools(state):
            return (
                f"Blocked {tool_name}: not available for guest users. "
                f"Use web_search, math_operations, or datetime, or final_answer to respond."
            )
        goal = state.get("goal", "")
        from agents.wcca_routing_mixin import OrchestratorRoutingMixin

        if state.get(
            "user_role"
        ) == "owner" and OrchestratorRoutingMixin._needs_guest_profile_review(
            OrchestratorRoutingMixin(), goal
        ):
            if tool_name == "web_search":
                return (
                    "Blocked web_search: guest profile questions must use "
                    "guest_user_profile_summary (saved facts only, no online lookup)."
                )
            if tool_name not in (
                "guest_user_profile_summary",
                "guest_accounts_list",
                "guest_audit_summary",
                "guest_set_age_band",
            ):
                return (
                    f"Blocked {tool_name}: for guest profile/interest questions use "
                    "guest_user_profile_summary only."
                )

        if tool_name in self.ORCHESTRATOR_BLOCKED_TOOLS:
            return (
                f"Blocked {tool_name}: not available in the orchestrator. "
                f"Use web_search for online lookups, document_search for uploaded files, "
                f"or final_answer to respond."
            )

        goal = state.get("goal", "")

        if state.get("lookup_search_done"):
            return (
                f"Blocked {tool_name}: web_search already returned results for this lookup. "
                f"Use action_type final_answer now — write a short report that answers GOAL "
                f"using only the web_search SOURCES in observations. Do not discuss unrelated "
                f"games, card lists, or uploaded documents."
            )

        if self._goal_is_web_lookup(goal) and tool_name == "document_search":
            return (
                "Blocked document_search: GOAL is a public web lookup — use web_search only, "
                "not the user's private uploaded files."
            )

        if (
            self._goal_is_web_lookup(goal)
            and tool_name == "web_search"
            and self._has_web_search_observation(state.get("observations", []))
        ):
            return (
                "Skipped repeat web_search: results are already in observations. "
                "Use final_answer to summarize for GOAL."
            )

        if (
            tool_name == "read_conversation_history"
            and self._goal_saves_conversation(goal)
            and self._has_read_history_observation(state.get("observations", []))
        ):
            root = self.config.runtime_paths.root if getattr(self, "config", None) else "var"
            path = self._save_file_path_from_goal(goal) or exports_subpath(
                "conversation_log.txt", root
            )
            return (
                f"Skipped repeat read_conversation_history: transcript already in observations. "
                f'Call write_file with {{"file_path": "{path}"}} (omit content) or final_answer.'
            )

        if (
            tool_name == "write_file"
            and self._goal_saves_conversation(goal)
            and self._save_already_verified(state)
        ):
            return (
                "Skipped repeat write_file: conversation already saved to disk. "
                "Use final_answer to confirm the path."
            )

        observations = state.get("observations", [])
        if tool_name == "list_mcp_tools":
            prior = sum(1 for o in observations if "list_mcp_tools" in o)
            if prior >= 2:
                return (
                    "Skipped repeat list_mcp_tools: tools already listed in observations. "
                    "Use a matching mcp_* tool or final_answer explaining the limitation."
                )

        if tool_name == "web_search" and self._goal_is_pure_math(goal):
            return (
                "Blocked web_search: GOAL is a pure math calculation — use calculator or "
                "math_operations only."
            )

        if self._has_ingested_documents(state) and tool_name in (
            "read_file",
            "list_directory",
        ):
            if not needs_codebase_intro(goal):
                return (
                    f"Blocked {tool_name}: ingested USER DOCUMENTS must be read with "
                    f"document_search (query + optional file_name), not filesystem tools. "
                    f"If you already have excerpts in observations, use final_answer."
                )

        sig = self._tool_call_signature(tool_name, tool_args)
        repeat = state.get("tool_repeat_failures", {}).get(sig, 0)
        if repeat >= self.REPEAT_TOOL_FAILURE_LIMIT:
            return (
                f"Skipped repeat {tool_name} call with identical tool_args "
                f"(already failed {repeat} times). Change args, pick another tool, "
                f"or use final_answer from existing observations."
            )

        total = state.get("tool_total_failures", {}).get(tool_name, 0)
        if total >= self.TOOL_TOTAL_FAILURE_LIMIT:
            return (
                f"Skipped {tool_name}: it failed {total} times this session. "
                f"Do not call it again — answer from observations or explain the blocker."
            )

        return None
