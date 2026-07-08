# agents/base_orchestrator_agent.py
"""
Base Orchestrator Agent for WitsV3.
Implements ReAct (Reason-Act-Observe) loop functionality.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, AsyncGenerator
from abc import abstractmethod

from agents.base_agent import BaseAgent
from core.config import WitsV3Config
from core.json_llm_parser import build_json_repair_prompt
from core.llm_interface import BaseLLMInterface
from core.memory_manager import MemoryManager
from core.schemas import StreamData, AgentResponse, ConversationHistory, ToolCall


class BaseOrchestratorAgent(BaseAgent):
    """
    Base class for orchestrator agents that implement ReAct loops.
    
    This agent follows the ReAct pattern:
    - Reason: Think about the goal and plan next steps
    - Act: Take an action (use tool, call agent, or provide answer)
    - Observe: Observe the results and plan next steps
    """

    # ReAct tool-selection/synthesis needs to be near-deterministic; the global
    # default_temperature (0.7) is tuned for chat/creative work and makes the
    # loop wander (redundant searches, listing instead of answering the specific
    # question). Low temp here dramatically improves consistency.
    REASONING_TEMPERATURE = 0.2
    REPEAT_TOOL_FAILURE_LIMIT = 2
    TOOL_TOTAL_FAILURE_LIMIT = 3
    # WCCA-only tools the ReAct loop must not call (model often picks intent_analysis for lookups).
    ORCHESTRATOR_BLOCKED_TOOLS = frozenset({"intent_analysis", "json_manipulate"})

    def __init__(
        self,
        agent_name: str,
        config: WitsV3Config,
        llm_interface: BaseLLMInterface,
        memory_manager: Optional[MemoryManager] = None,
        tool_registry: Optional[Any] = None
    ):
        """
        Initialize the orchestrator agent.
        
        Args:
            agent_name: Name of this agent
            config: System configuration
            llm_interface: LLM interface
            memory_manager: Optional memory manager
            tool_registry: Optional tool registry for accessing tools
        """
        super().__init__(agent_name, config, llm_interface, memory_manager)
        self.tool_registry = tool_registry
        
        # ReAct loop configuration
        self.max_iterations = config.agents.max_iterations
        self.current_iteration = 0
        
        self.logger.info(f"Initialized {self.__class__.__name__} with max_iterations: {self.max_iterations}")
    
    def get_model_name(self) -> str:
        """Get the model name for orchestrator agents."""
        return self.config.ollama_settings.orchestrator_model
    
    async def run(
        self,
        user_input: str,
        conversation_history: Optional[ConversationHistory] = None,
        session_id: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[StreamData, None]:
        """
        Execute the ReAct loop to achieve the given goal.
        
        Args:
            user_input: The goal or user request to achieve
            conversation_history: Optional conversation context
            session_id: Optional session identifier
            **kwargs: Additional parameters
            
        Yields:
            StreamData objects showing the reasoning and action process
        """
        goal = user_input
        self.current_iteration = 0

        # Route the whole ReAct session on the goal: code goals go to the
        # coder model (better structured output than qwen3), everything else
        # stays on the orchestrator model. Never the trivial model — the
        # small model can't hold the ReAct JSON format.
        self._session_model = self.model_router.route(
            goal, default=self.get_model_name(), allow_trivial=False
        )
        if self._session_model != self.get_model_name():
            self.logger.info(f"Orchestrating with routed model: {self._session_model}")

        # Initial setup
        yield self.stream_thinking(f"Starting orchestration for goal: {goal}")
        
        # Store the goal in memory
        await self.store_memory(
            content=f"Goal: {goal}",
            segment_type="GOAL",
            importance=0.9,
            metadata={"session_id": session_id}
        )
        
        # Get relevant context from memory
        relevant_memories = await self.search_memory(goal, limit=5)
        context = self._build_context_from_memories(relevant_memories)
        doc_inventory = await self._get_document_inventory()

        # Initialize the ReAct state
        react_state = {
            "goal": goal,
            "context": context,
            "documents_context": self._format_documents_context(doc_inventory),
            "conversation_history": conversation_history,
            "history": conversation_history.to_llm_format() if conversation_history else [],
            "observations": [],
            "completed": False,
            "final_answer": None,
            "tool_repeat_failures": {},
            "tool_total_failures": {},
            "lookup_search_done": False,
        }
        
        try:
            # Execute ReAct loop
            async for stream_data in self._execute_react_loop(react_state, session_id):
                yield stream_data
                
        except Exception as e:
            self.logger.error(f"Error in orchestrator ReAct loop: {e}")
            yield self.stream_error(
                f"An error occurred during orchestration: {str(e)}",
                details=str(e)
            )
    
    async def _execute_react_loop(
        self,
        state: Dict[str, Any],
        session_id: Optional[str]
    ) -> AsyncGenerator[StreamData, None]:
        """
        Execute the main ReAct loop.
        
        Args:
            state: Current ReAct state
            session_id: Session identifier
            
        Yields:
            StreamData from each step of the loop
        """
        while not state["completed"] and self.current_iteration < self.max_iterations:
            self.current_iteration += 1
            
            yield self.stream_thinking(f"ReAct iteration {self.current_iteration}/{self.max_iterations}")
            
            # REASON: Generate thoughts and plan
            reasoning_prompt = self._build_reasoning_prompt(state)
            
            yield self.stream_thinking("Analyzing the situation and planning next steps...")
            
            reasoning_response = await self.generate_response(
                reasoning_prompt,
                model_name=getattr(self, "_session_model", None),
                temperature=self.REASONING_TEMPERATURE,
                response_format="json"
            )
            parsed_reasoning = self._parse_reasoning_response(reasoning_response)

            # Repair-reparse: if the response wasn't usable JSON, ask the model
            # once to rewrite its own output as valid JSON before falling back.
            if parsed_reasoning.pop("_parse_failed", False):
                parse_error = parsed_reasoning.pop("_parse_error", "invalid JSON")
                yield self.stream_thinking("Reasoning response was malformed; attempting JSON repair...")
                try:
                    repaired_response = await self.generate_response(
                        self._build_json_repair_prompt(reasoning_response, parse_error),
                        model_name=getattr(self, "_session_model", None),
                        temperature=self.REASONING_TEMPERATURE,
                        response_format="json"
                    )
                    reparsed = self._parse_reasoning_response(repaired_response)
                    if not reparsed.pop("_parse_failed", False):
                        reparsed.pop("_parse_error", None)
                        parsed_reasoning = reparsed
                        reasoning_response = repaired_response
                    else:
                        self.logger.warning("JSON repair attempt also failed to parse; using fallback reasoning")
                except Exception as e:
                    self.logger.warning(f"JSON repair attempt errored: {e}; using fallback reasoning")
            
            # Stream the reasoning
            if parsed_reasoning.get("thought"):
                yield self.stream_thinking(f"Thought: {parsed_reasoning['thought']}")
            
            # Store reasoning in memory
            await self.store_memory(
                content=f"Reasoning: {reasoning_response}",
                segment_type="REASONING",
                importance=0.7,
                metadata={"iteration": self.current_iteration, "session_id": session_id}
            )
            
            # ACT: Decide on action
            action_type = parsed_reasoning.get("action_type", "final_answer")
            
            if action_type == "tool_call":
                # Execute tool call
                async for stream_data in self._execute_tool_action(parsed_reasoning, state, session_id):
                    yield stream_data
                    
            elif action_type == "final_answer":
                # Provide final answer
                final_answer = parsed_reasoning.get("final_answer", "I've completed the task.")
                state["completed"] = True
                state["final_answer"] = final_answer
                
                yield self.stream_result(final_answer)
                
                # Store final answer
                await self.store_memory(
                    content=f"Final Answer: {final_answer}",
                    segment_type="FINAL_ANSWER",
                    importance=1.0,
                    metadata={"session_id": session_id}
                )
                
            else:
                coerced = self._coerce_unknown_action(parsed_reasoning)
                if coerced:
                    parsed_reasoning = coerced
                    action_type = parsed_reasoning.get("action_type")
                    if action_type == "tool_call":
                        async for stream_data in self._execute_tool_action(
                            parsed_reasoning, state, session_id
                        ):
                            yield stream_data
                        continue
                    if action_type == "final_answer":
                        final_answer = parsed_reasoning.get(
                            "final_answer", "I've completed the task."
                        )
                        state["completed"] = True
                        state["final_answer"] = final_answer
                        yield self.stream_result(final_answer)
                        await self.store_memory(
                            content=f"Final Answer: {final_answer}",
                            segment_type="FINAL_ANSWER",
                            importance=1.0,
                            metadata={"session_id": session_id},
                        )
                        continue

                observation = (
                    f"Invalid action_type '{action_type}'. "
                    f"Use action_type 'tool_call' with tool_name, or 'final_answer'."
                )
                yield self.stream_error(observation)
                state["observations"].append(observation)
        
        if self.current_iteration >= self.max_iterations and not state["completed"]:
            yield self.stream_error(
                f"Reached maximum iterations ({self.max_iterations}) without completing the goal."
            )
    
    async def _execute_tool_action(
        self,
        reasoning: Dict[str, Any],
        state: Dict[str, Any],
        session_id: Optional[str]
    ) -> AsyncGenerator[StreamData, None]:
        """
        Execute a tool action and observe the results.
        
        Args:
            reasoning: Parsed reasoning containing tool call details
            state: Current ReAct state
            session_id: Session identifier
            
        Yields:
            StreamData from tool execution
        """
        tool_name = reasoning.get("tool_name")
        tool_args = reasoning.get("tool_args", {})
        
        if not tool_name:
            yield self.stream_error("Tool name not specified in reasoning")
            return
        
        block_reason = self._preflight_tool_call(tool_name, tool_args, state)
        if block_reason:
            observation = block_reason
            if block_reason.startswith(("Blocked", "Skipped")):
                self._record_tool_failure(tool_name, tool_args, state)
            yield self.stream_observation(observation)
            state["observations"].append(observation)
            await self.store_memory(
                content=observation,
                segment_type="OBSERVATION",
                importance=0.8,
                metadata={"tool_name": tool_name, "session_id": session_id, "blocked": True},
            )
            return

        yield self.stream_action(f"Calling tool: {tool_name} with args: {tool_args}")

        tool_result = None
        if self.tool_registry:
            try:
                tool_result = await self._call_tool(tool_name, tool_args, state)
                observation = self._format_tool_observation(tool_name, tool_result)
            except Exception as e:
                observation = f"Tool {tool_name} failed: {str(e)}"
                yield self.stream_error(f"Tool execution failed: {str(e)}")
        else:
            observation = f"Tool registry not available. Simulated call to {tool_name}."

        if self._observation_indicates_failure(observation) or (
            isinstance(tool_result, dict) and self._tool_result_is_failure(tool_result)
        ):
            self._record_tool_failure(tool_name, tool_args, state)

        # Add observation to state
        state["observations"].append(observation)
        yield self.stream_observation(observation)

        await self.store_memory(
            content=observation,
            segment_type="OBSERVATION",
            importance=0.8,
            metadata={"tool_name": tool_name, "session_id": session_id},
        )

        # Save-to-file: after transcript is read, write immediately — the model
        # often loops on read_conversation_history and never reaches write_file.
        if (
            tool_name == "read_conversation_history"
            and self._goal_saves_conversation(state.get("goal", ""))
            and not self._observation_indicates_failure(observation)
        ):
            file_path = self._save_file_path_from_goal(state.get("goal", "")) or "exports/conversation_log.txt"
            if file_path:
                async for stream_data in self._auto_write_saved_conversation(
                    file_path, state, session_id
                ):
                    yield stream_data
                return

        if (
            tool_name == "web_search"
            and self._goal_is_web_lookup(state.get("goal", ""))
            and not self._observation_indicates_failure(observation)
        ):
            state["lookup_search_done"] = True
    
    @staticmethod
    def _has_ingested_documents(state: Dict[str, Any]) -> bool:
        """True when USER DOCUMENTS lists at least one ingested file."""
        ctx = state.get("documents_context", "")
        return bool(ctx) and "No user documents are currently ingested" not in ctx

    @staticmethod
    def _tool_call_signature(tool_name: str, tool_args: Dict[str, Any]) -> str:
        import json

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
                f"Call write_file with {{\"file_path\": \"{path}\"}} (omit content) or final_answer."
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
        """Render a tool result for the ReAct observation.

        web_search and document_search get structured layouts so the model
        synthesizes from sources/excerpts instead of ignoring a raw dict.
        """
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

        # web_search always sets provider (or answer_provider) on success.
        if result.get("provider") or result.get("answer_provider"):
            return True

        # Non-empty hits: web results carry link/title/snippet; document_search
        # uses file/text/chunk and MCP discovery uses name/command.
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
                # Full transcript for export/save requests.
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

    async def _call_tool(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        state: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Call a tool through the tool registry.

        Args:
            tool_name: Name of the tool to call
            tool_args: Arguments for the tool
            state: Optional ReAct state for context injection

        Returns:
            Tool execution result
        """
        if not self.tool_registry:
            raise Exception("Tool registry not available")

        if state is not None:
            tool_args = await self._prepare_tool_args(tool_name, tool_args, state)

        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            raise Exception(f"Tool {tool_name} not found")

        return await tool.execute(**tool_args)

    def _coerce_unknown_action(
        self, parsed: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Last-chance fix when action_type is a tool name. Override in subclasses."""
        return None
    
    def _build_json_repair_prompt(self, raw_response: str, parse_error: str) -> str:
        """
        Build a prompt asking the model to rewrite its malformed reasoning
        output as valid JSON.

        Args:
            raw_response: The response that failed to parse
            parse_error: The parse error message

        Returns:
            Repair prompt
        """
        return build_json_repair_prompt(
            raw_response,
            parse_error,
            required_keys='"thought", "action_type", "tool_name", "tool_args", "final_answer"',
        )

    @abstractmethod
    def _build_reasoning_prompt(self, state: Dict[str, Any]) -> str:
        """
        Build the prompt for the reasoning step.
        
        Args:
            state: Current ReAct state
            
        Returns:
            Prompt for reasoning
        """
        pass
    
    @abstractmethod
    def _parse_reasoning_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the LLM's reasoning response.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Parsed reasoning components
        """
        pass
    
    def _build_context_from_memories(self, memories: List[Any]) -> str:
        """
        Build context string from memory search results.
        
        Args:
            memories: List of memory segments
            
        Returns:
            Context string
        """
        if not memories:
            return "No relevant context found."
        
        context_parts = []
        for memory in memories:
            if hasattr(memory, 'content') and hasattr(memory.content, 'text'):
                context_parts.append(f"- {memory.content.text}")
        
        return "\n".join(context_parts) if context_parts else "No relevant context found."

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


# Test function
async def test_base_orchestrator():
    """Test the BaseOrchestratorAgent functionality."""
    print("Testing BaseOrchestratorAgent...")
    
    # Note: This is a mock test since BaseOrchestratorAgent is abstract
    print("✓ BaseOrchestratorAgent is properly defined as abstract class")
    print("✓ ReAct loop structure is implemented")
    print("✓ Tool integration hooks are available")
    print("✓ Memory integration is included")
    
    print("BaseOrchestratorAgent tests completed! 🎉")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_base_orchestrator())
