"""
ReAct tool guardrails, observation formatting, and save-to-file helpers.

Extracted from BaseOrchestratorAgent to keep the orchestrator module under
the 500-line project limit while preserving test harness compatibility.
"""

from __future__ import annotations

import json
import re
from typing import Any, AsyncGenerator, Dict, List, Optional

from core.schemas import ConversationHistory, StreamData


class OrchestratorToolHelpersMixin:
    """Mixin: preflight guardrails, observation layout, and export helpers."""

    REPEAT_TOOL_FAILURE_LIMIT = 2
    TOOL_TOTAL_FAILURE_LIMIT = 3
    ORCHESTRATOR_BLOCKED_TOOLS = frozenset({"intent_analysis", "json_manipulate"})

    @staticmethod
    def _has_ingested_documents(state: Dict[str, Any]) -> bool:
        """True when USER DOCUMENTS lists at least one ingested file."""
        ctx = state.get("documents_context", "")
        return bool(ctx) and "No user documents are currently ingested" not in ctx

    @staticmethod
    def _tool_call_signature(tool_name: str, tool_args: Dict[str, Any]) -> str:
        return f"{tool_name}:{json.dumps(tool_args, sort_keys=True, default=str)}"

    def _preflight_tool_call(
        self, tool_name: str, tool_args: Dict[str, Any], state: Dict[str, Any]
    ) -> Optional[str]:
        """Return a block message to skip doomed/repeat tool calls, or None to proceed."""
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
            path = self._save_file_path_from_goal(goal) or "exports/conversation_log.txt"
            return (
                f"Skipped repeat read_conversation_history: transcript already in observations. "
                f'Call write_file with {{"file_path": "{path}"}} (omit content) or final_answer.'
            )

        if self._has_ingested_documents(state) and tool_name in (
            "read_file",
            "list_directory",
        ):
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

    @staticmethod
    def _has_read_history_observation(observations: List[str]) -> bool:
        prefix = "Tool read_conversation_history result:"
        return any(o.startswith(prefix) for o in observations)

    @staticmethod
    def _save_file_path_from_goal(goal: str) -> Optional[str]:
        """Extract a target path from save/export phrasing (e.g. exports/chat.txt)."""
        if not goal:
            return None
        patterns = (
            r"(?:as|to|into)\s+([^\s?\"']+\.(?:txt|md|log|json))",
            r"\b([\w./-]+\.(?:txt|md|log|json))\b",
        )
        for pattern in patterns:
            match = re.search(pattern, goal, re.IGNORECASE)
            if match:
                return match.group(1).replace("\\", "/")
        return None

    async def _auto_write_saved_conversation(
        self,
        file_path: str,
        state: Dict[str, Any],
        session_id: Optional[str],
    ) -> AsyncGenerator[StreamData, None]:
        """Write session transcript after read_conversation_history for save goals."""
        yield self.stream_action(
            f"Auto-saving conversation to {file_path} (read_conversation_history complete)"
        )
        try:
            write_result = await self._call_tool(
                "write_file", {"file_path": file_path}, state
            )
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
            observation.startswith("Tool ")
            and " failed:" in observation
        ) or "(search failed:" in lowered or observation.startswith("Blocked ")

    def _record_tool_failure(
        self, tool_name: str, tool_args: Dict[str, Any], state: Dict[str, Any]
    ) -> None:
        sig = self._tool_call_signature(tool_name, tool_args)
        repeat = state.setdefault("tool_repeat_failures", {})
        repeat[sig] = repeat.get(sig, 0) + 1
        total = state.setdefault("tool_total_failures", {})
        total[tool_name] = total.get(tool_name, 0) + 1

    def _format_tool_observation(self, tool_name: str, result: Any) -> str:
        """Render a tool result for the ReAct observation."""
        if isinstance(result, dict) and self._is_web_search_result(tool_name, result):
            return self._format_search_observation(tool_name, result)
        if isinstance(result, dict) and self._is_document_search_result(tool_name, result):
            return self._format_document_observation(tool_name, result)
        return f"Tool {tool_name} result: {result}"

    @staticmethod
    def _is_web_search_result(tool_name: str, result: Dict[str, Any]) -> bool:
        """True when *result* is from web_search, not another results-list tool."""
        results = result.get("results")
        if not isinstance(results, list):
            return False

        if result.get("provider") or result.get("answer_provider"):
            return True

        if results and isinstance(results[0], dict):
            sample = results[0]
            if "link" in sample or "snippet" in sample:
                return tool_name == "web_search"
            if "file" in sample or "text" in sample or "installable" in sample:
                return False

        return False

    @staticmethod
    def _is_document_search_result(tool_name: str, result: Dict[str, Any]) -> bool:
        """True when *result* is from document_search."""
        if tool_name != "document_search" or not isinstance(result, dict):
            return False
        if "query" in result or result.get("success") is not None:
            return True
        results = result.get("results")
        return isinstance(results, list) and (
            not results or isinstance(results[0], dict) and "text" in results[0]
        )

    @staticmethod
    def _format_document_observation(tool_name: str, result: Dict[str, Any]) -> str:
        lines = [f"{tool_name} results (base your answer on the EXCERPTS below):"]
        if result.get("success") is False:
            lines.append(f"(search failed: {result.get('error', 'unknown error')})")
            return "\n".join(lines)
        excerpts = result.get("results") or []
        if not excerpts:
            lines.append("(no matching passages — try a broader query before giving up)")
            return "\n".join(lines)
        for i, r in enumerate(excerpts, 1):
            file_name = (r.get("file") or "unknown").strip()
            chunk = (r.get("chunk") or "").strip()
            text = (r.get("text") or "").strip()
            rel = r.get("relevance")
            header = f"[{i}] {file_name}"
            if chunk:
                header += f" ({chunk})"
            if rel is not None:
                header += f" relevance={rel}"
            lines.append(f"{header}\n    {text}")
        return "\n".join(lines)

    @staticmethod
    def _format_search_observation(tool_name: str, result: Dict[str, Any]) -> str:
        lines = [f"{tool_name} results (base your answer on the SOURCES below):"]
        answer = result.get("answer")
        if answer:
            provider = result.get("answer_provider", "search engine")
            lines.append(
                f"{provider} summary (usually accurate — use it, but trust the "
                f"sources below if any clearly contradicts it): {answer}"
            )
        sources = result.get("results") or []
        max_sources = 5
        max_snippet = 220
        for i, r in enumerate(sources[:max_sources], 1):
            title = (r.get("title") or "").strip()
            snippet = (r.get("snippet") or "").strip()
            if len(snippet) > max_snippet:
                snippet = snippet[:max_snippet] + "…"
            link = (r.get("link") or "").strip()
            lines.append(f"[{i}] {title}\n    {snippet}\n    source: {link}")
        if len(sources) > max_sources:
            lines.append(f"(+ {len(sources) - max_sources} more sources omitted)")
        if not sources and not answer:
            lines.append("(no results found)")
        return "\n".join(lines)

    async def _prepare_tool_args(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
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
            content = args.get("content")
            if not content or not str(content).strip():
                from_obs = self._conversation_text_from_observations(
                    state.get("observations", [])
                )
                if from_obs:
                    args["content"] = from_obs
                elif conversation_history is not None and self._goal_saves_conversation(
                    state.get("goal", "")
                ):
                    args["content"] = self._format_conversation_for_file(
                        conversation_history
                    )
            if not args.get("file_path"):
                path = self._save_file_path_from_goal(state.get("goal", ""))
                if path:
                    args["file_path"] = path

        return args

    @staticmethod
    def _has_web_search_observation(observations: List[str]) -> bool:
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
        signals = (
            "save this conversation",
            "save our conversation",
            "save the conversation",
            "save to file",
            "save to a file",
            "save to disk",
            "write to file",
            "write it to",
            "export conversation",
            "log of our conversation",
            "save a log",
            "save this chat",
            "write the story",
            "save the story",
            "save as a file",
        )
        return any(sig in lowered for sig in signals)

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
    def _conversation_text_from_observations(observations: List[str]) -> Optional[str]:
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

    async def _get_document_inventory(self) -> Dict[str, int]:
        """File path -> chunk count for every ingested document (empty if none)."""
        if not self.memory_manager:
            return {}
        try:
            segments = await self.memory_manager.get_recent_memory(
                limit=1_000_000, filter_dict={"type": "DOCUMENT_CHUNK"}
            )
        except Exception as e:
            self.logger.warning(f"Could not list ingested documents: {e}")
            return {}
        counts: Dict[str, int] = {}
        for seg in segments:
            fp = seg.metadata.get("file_path")
            if fp:
                counts[fp] = counts.get(fp, 0) + 1
        return counts

    @staticmethod
    def _format_documents_context(inventory: Dict[str, int]) -> str:
        """Prompt block listing ingested documents the orchestrator can search."""
        if not inventory:
            return "No user documents are currently ingested."
        listing = "\n".join(
            f"- {name} ({count} chunks)" for name, count in sorted(inventory.items())
        )
        return (
            "These user documents are ALREADY ingested and searchable via "
            "document_search. Never claim you cannot access them:\n" + listing
        )
