"""
ReAct tool guardrails, observation formatting, and save-to-file helpers.

Extracted from BaseOrchestratorAgent to keep the orchestrator module under
the 500-line project limit while preserving test harness compatibility.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncGenerator
from typing import Any

from core.runtime_paths import exports_subpath, upgrade_runtime_path
from core.schemas import ConversationHistory, StreamData


class OrchestratorToolHelpersMixin:
    """Mixin: preflight guardrails, observation layout, and export helpers."""

    REPEAT_TOOL_FAILURE_LIMIT = 2
    TOOL_TOTAL_FAILURE_LIMIT = 3
    ORCHESTRATOR_BLOCKED_TOOLS = frozenset({"intent_analysis", "json_manipulate"})

    def _guest_allowed_tools(self) -> frozenset[str]:
        from core.guest_access import guest_tools_for_age_band

        state = getattr(self, "_react_state_for_tools", None) or {}
        age_band = state.get("guest_age_band", "teen")
        return guest_tools_for_age_band(age_band, getattr(self, "config", None))

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
        if state.get("user_role") == "guest" and tool_name not in self._guest_allowed_tools():
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
    def _has_read_history_observation(observations: list[str]) -> bool:
        prefix = "Tool read_conversation_history result:"
        return any(o.startswith(prefix) for o in observations)

    @staticmethod
    def _save_file_path_from_goal(goal: str) -> str | None:
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
                return upgrade_runtime_path(match.group(1).replace("\\", "/"))
        return None

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

    def _format_tool_observation(self, tool_name: str, result: Any) -> str:
        """Render a tool result for the ReAct observation."""
        if isinstance(result, dict) and self._is_web_search_result(tool_name, result):
            return self._format_search_observation(tool_name, result)
        if isinstance(result, dict) and self._is_document_search_result(tool_name, result):
            return self._format_document_observation(tool_name, result)
        return f"Tool {tool_name} result: {result}"

    @staticmethod
    def _is_web_search_result(tool_name: str, result: dict[str, Any]) -> bool:
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
    def _is_document_search_result(tool_name: str, result: dict[str, Any]) -> bool:
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
    def _format_document_observation(tool_name: str, result: dict[str, Any]) -> str:
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
    def _format_search_observation(tool_name: str, result: dict[str, Any]) -> str:
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

    # --- final-answer synthesis guard ----------------------------------------

    _REFUSAL_PHRASES = (
        "don't have access",
        "do not have access",
        "cannot access",
        "can't access",
        "not uploaded",
        "no documents",
        "don't have your",
        "do not have your",
        "knowledge cutoff",
        "training data",
        "as an ai",
        "i'm not able to browse",
        "i cannot browse",
    )

    @staticmethod
    def _latest_observation_prefix(observations: list[str], prefix: str) -> str | None:
        for obs in reversed(observations):
            if prefix in obs:
                return obs
        return None

    @staticmethod
    def _significant_terms(text: str, min_len: int = 4) -> set:
        words = re.findall(r"[a-z0-9]+", text.lower())
        stop = {
            "that",
            "this",
            "with",
            "from",
            "have",
            "your",
            "about",
            "what",
            "when",
            "where",
            "which",
            "their",
            "there",
            "would",
            "could",
            "should",
            "been",
            "were",
            "they",
            "them",
            "than",
            "then",
            "into",
            "only",
            "also",
            "just",
            "like",
            "some",
            "such",
            "very",
            "does",
            "answer",
            "using",
            "below",
            "results",
            "search",
            "document",
        }
        return {w for w in words if len(w) >= min_len and w not in stop}

    def _answer_denies_access(self, answer: str) -> bool:
        lowered = answer.lower()
        return any(phrase in lowered for phrase in self._REFUSAL_PHRASES)

    def _answer_references_evidence(
        self, answer: str, evidence_terms: set, min_overlap: int = 2
    ) -> bool:
        if not evidence_terms:
            return True
        answer_terms = self._significant_terms(answer)
        return len(answer_terms & evidence_terms) >= min_overlap

    @classmethod
    def _extract_document_evidence_terms(cls, observation: str) -> set:
        terms: set = set()
        for line in observation.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("document_search"):
                continue
            if stripped.startswith("("):
                continue
            if stripped.startswith("["):
                header, _, body = stripped.partition("]")
                terms |= cls._significant_terms(header)
                if body.strip():
                    terms |= cls._significant_terms(body)
            else:
                terms |= cls._significant_terms(stripped)
        return terms

    @classmethod
    def _extract_web_summary(cls, observation: str) -> str | None:
        for line in observation.splitlines():
            if "summary" in line.lower() and ":" in line:
                return line.split(":", 1)[1].strip()
        return None

    def _goal_expects_documents(self, goal: str) -> bool:
        lowered = (goal or "").lower()
        signals = (
            "document",
            "report",
            "file",
            "upload",
            "audit",
            "my notes",
            "ingested",
            "summarize",
        )
        return any(sig in lowered for sig in signals)

    @staticmethod
    def _document_search_weak(doc_obs: str) -> bool:
        if not doc_obs:
            return False
        if "(no matching passages" in doc_obs or "(search failed:" in doc_obs:
            return True
        import re

        scores = [float(m) for m in re.findall(r"relevance=([0-9.]+)", doc_obs)]
        return bool(scores) and max(scores) < 0.25

    @staticmethod
    def _answer_acknowledges_gap(answer: str) -> bool:
        lowered = (answer or "").lower()
        gap_phrases = (
            "not in your",
            "couldn't find",
            "could not find",
            "no matching",
            "don't have",
            "do not have",
            "not uploaded",
            "insufficient",
            "try uploading",
            "broader query",
            "didn't find",
            "did not find",
        )
        return any(phrase in lowered for phrase in gap_phrases)

    def _insufficient_document_evidence_message(self, goal: str) -> str:
        return (
            f"I searched your ingested documents for “{goal}” but didn't find passages "
            "that clearly answer the question. Try rephrasing, naming a specific file, "
            "or uploading the document if it isn't in the documents folder yet."
        )

    def _validate_final_answer_synthesis(
        self, final_answer: str, state: dict[str, Any]
    ) -> str | None:
        """
        Return a guard message when final_answer ignores usable search observations.
        """
        observations = state.get("observations", [])
        if not observations or not final_answer or not str(final_answer).strip():
            return None

        goal = state.get("goal", "")
        doc_obs = self._latest_observation_prefix(observations, "document_search results")
        if doc_obs and self._document_search_weak(doc_obs) and self._goal_expects_documents(goal):
            if not self._answer_acknowledges_gap(final_answer):
                return (
                    "document_search found no strong excerpts but the answer sounds definitive. "
                    "Say clearly that the uploaded documents do not contain enough evidence."
                )

        if (
            doc_obs
            and "(no matching passages" not in doc_obs
            and not self._document_search_weak(doc_obs)
        ):
            if self._answer_denies_access(final_answer):
                return (
                    "document_search returned excerpts but the answer claims no access. "
                    "Summarize the numbered EXCERPTS only."
                )
            evidence = self._extract_document_evidence_terms(doc_obs)
            if evidence and not self._answer_references_evidence(final_answer, evidence):
                return (
                    "Answer does not appear grounded in document_search EXCERPTS. "
                    "Cite or paraphrase the numbered passages."
                )

        web_obs = self._latest_observation_prefix(observations, "web_search results")
        if web_obs and "(no results found)" not in web_obs:
            if state.get("lookup_search_done") and self._answer_denies_access(final_answer):
                return (
                    "web_search already returned SOURCES but the answer refuses or deflects. "
                    "Use the summary and SOURCES in observations."
                )
            summary = self._extract_web_summary(web_obs)
            if (
                summary
                and len(final_answer.strip()) < 50
                and not self._answer_references_evidence(
                    final_answer, self._significant_terms(summary), min_overlap=1
                )
            ):
                return "Answer is too thin given an available web_search summary."

        return None

    def _auto_synthesize_from_observations(self, state: dict[str, Any]) -> str | None:
        """Build a grounded answer from the latest search observation when the model won't."""
        observations = state.get("observations", [])
        goal = state.get("goal", "")

        web_obs = self._latest_observation_prefix(observations, "web_search results")
        if web_obs and "(no results found)" not in web_obs:
            summary = self._extract_web_summary(web_obs)
            if summary:
                return summary
            lines = [ln.strip() for ln in web_obs.splitlines() if ln.strip().startswith("[")]
            if lines:
                return f"Based on web search for your question: {lines[0]}"

        doc_obs = self._latest_observation_prefix(observations, "document_search results")
        if doc_obs and self._document_search_weak(doc_obs):
            return self._insufficient_document_evidence_message(goal)

        if doc_obs and "(no matching passages" not in doc_obs:
            excerpt_lines = [
                ln.strip()
                for ln in doc_obs.splitlines()
                if ln.strip().startswith("[") or (ln.startswith("    ") and ln.strip())
            ]
            if excerpt_lines:
                body = "\n".join(excerpt_lines[:4])
                return f"From your uploaded documents (re: {goal}):\n{body}"

        return None

    def _resolve_final_answer(self, final_answer: str, state: dict[str, Any]) -> tuple:
        """
        Apply synthesis guard before accepting final_answer.

        Returns:
            (resolved_answer, completed) — completed False means retry the ReAct loop.
        """
        guard_msg = self._validate_final_answer_synthesis(final_answer, state)
        if guard_msg:
            retries = state.get("synthesis_guard_retries", 0)
            if retries == 0:
                state["synthesis_guard_retries"] = 1
                state["observations"].append(f"Synthesis guard: {guard_msg}")
                return final_answer, False
            fallback = self._auto_synthesize_from_observations(state)
            if fallback:
                final_answer = fallback

        state["completed"] = True
        state["final_answer"] = final_answer
        return final_answer, True

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
            content = args.get("content")
            if not content or not str(content).strip():
                from_obs = self._conversation_text_from_observations(state.get("observations", []))
                if from_obs:
                    args["content"] = from_obs
                elif conversation_history is not None and self._goal_saves_conversation(
                    state.get("goal", "")
                ):
                    args["content"] = self._format_conversation_for_file(conversation_history)
            if not args.get("file_path"):
                path = self._save_file_path_from_goal(state.get("goal", ""))
                if path:
                    args["file_path"] = path
            if args.get("file_path"):
                root = self.config.runtime_paths.root if getattr(self, "config", None) else "var"
                args["file_path"] = upgrade_runtime_path(str(args["file_path"]), root)

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

    async def _get_document_inventory(self) -> dict[str, int]:
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
        counts: dict[str, int] = {}
        for seg in segments:
            fp = seg.metadata.get("file_path")
            if fp:
                counts[fp] = counts.get(fp, 0) + 1
        return counts

    @staticmethod
    def _format_documents_context(inventory: dict[str, int]) -> str:
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
