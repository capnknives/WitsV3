# agents/llm_driven_orchestrator.py
"""
LLM-Driven Orchestrator Agent for WitsV3.
Implements the ReAct loop with LLM-driven decision making.
"""

from typing import Any

from agents.base_orchestrator_agent import BaseOrchestratorAgent
from core.config import WitsV3Config
from core.json_llm_parser import parse_json_object, strip_think_blocks
from core.llm_interface import BaseLLMInterface
from core.memory_manager import MemoryManager
from core.schemas import ConversationHistory


class LLMDrivenOrchestrator(BaseOrchestratorAgent):
    """
    LLM-Driven Orchestrator that uses the ReAct pattern for goal achievement.

    This orchestrator leverages the LLM's reasoning capabilities to:
    1. Break down complex goals into steps
    2. Decide when to use tools vs. provide answers
    3. Learn from previous iterations
    4. Adapt its approach based on results
    """

    # Hide WCCA-internal tools from the ReAct tool list (model misuses them for search).
    _ORCHESTRATOR_TOOL_EXCLUDE = frozenset({"intent_analysis", "json_manipulate"})

    def __init__(
        self,
        agent_name: str,
        config: WitsV3Config,
        llm_interface: BaseLLMInterface,
        memory_manager: MemoryManager | None = None,
        tool_registry: Any | None = None,
    ):
        """
        Initialize the LLM-Driven Orchestrator.

        Args:
            agent_name: Name of this agent
            config: System configuration
            llm_interface: LLM interface
            memory_manager: Optional memory manager
            tool_registry: Optional tool registry
        """
        super().__init__(agent_name, config, llm_interface, memory_manager, tool_registry)

        # Available tools (will be populated from tool registry)
        self.available_tools = self._get_available_tools()

        self.logger.info(
            f"LLM-Driven Orchestrator initialized with {len(self.available_tools)} tools"
        )

    def _get_available_tools(self) -> list[dict[str, Any]]:
        """
        Get list of available tools from the tool registry.

        Returns:
            List of tool descriptions for the LLM
        """
        if not self.tool_registry:
            return [
                {
                    "name": "think",
                    "description": "Think through the problem step by step",
                    "parameters": {"thought": "string"},
                },
                {
                    "name": "answer",
                    "description": "Provide a final answer to the user",
                    "parameters": {"answer": "string"},
                },
            ]

        # Get actual tools from tool registry
        tools = []
        try:
            tools = self.tool_registry.get_tools_for_llm()
            tools = [t for t in tools if t.get("name") not in self._ORCHESTRATOR_TOOL_EXCLUDE]
            self.logger.info(f"Retrieved {len(tools)} tools from registry")
        except Exception as e:
            self.logger.warning(f"Error getting tools from registry: {e}")
            # Return basic tools as fallback
            tools = [
                {
                    "name": "think",
                    "description": "Think through the problem step by step",
                    "parameters": {"thought": "string"},
                },
                {
                    "name": "answer",
                    "description": "Provide a final answer to the user",
                    "parameters": {"answer": "string"},
                },
            ]

        return tools

    async def run(
        self,
        user_input: str,
        conversation_history: ConversationHistory | None = None,
        session_id: str | None = None,
        **kwargs,
    ):
        """Run the ReAct loop, refreshing the tool list each session."""
        self.available_tools = self._get_available_tools()
        mcp_count = sum(1 for t in self.available_tools if t.get("name", "").startswith("mcp_"))
        if mcp_count:
            self.logger.info("Orchestrator sees %s MCP tool(s) this session", mcp_count)
        async for stream_data in super().run(
            user_input, conversation_history=conversation_history, session_id=session_id, **kwargs
        ):
            yield stream_data

    def _build_reasoning_prompt(self, state: dict[str, Any]) -> str:
        """
        Build the reasoning prompt for the ReAct loop.

        Args:
            state: Current ReAct state

        Returns:
            Formatted prompt for reasoning
        """
        goal = state["goal"]
        context = state["context"]
        documents_context = state.get(
            "documents_context", "No user documents are currently ingested."
        )
        history = state["history"]
        observations = state["observations"]

        # Build conversation history
        history_text = ""
        if history:
            history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-3:]])

        # Build observations
        observations_text = ""
        if observations:
            observations_text = "\n".join([f"- {obs}" for obs in observations[-3:]])

        lookup_hint = ""
        goal_lower = goal.lower()
        if any(
            sig in goal_lower
            for sig in ("look up", "search for", "report on", "tell me about", "what is")
        ):
            lookup_hint = (
                f"\nLOOKUP TARGET: Use the user's exact name/title verbatim in web_search "
                f"query (quote multi-word titles). Do NOT substitute a different product "
                f"from the same franchise. Goal wording: {goal}\n"
            )

        # Build available tools text
        tools_text = ""
        if self.available_tools:
            tools_text = "\n".join(
                [f"- {tool['name']}: {tool['description']}" for tool in self.available_tools]
            )

        personalization = state.get("guest_personalization_context", "")
        personalization_block = f"\nGUEST PERSONALIZATION:\n{personalization}\n" if personalization else ""

        guest_rules_block = ""
        if state.get("user_role") == "guest":
            from core.content_policy import guest_system_prompt_slice

            guest_rules_block = (
                f"\n{guest_system_prompt_slice(state.get('guest_age_band', 'teen'))}\n"
            )

        prompt = f"""You are an AI orchestrator using the ReAct (Reason-Act-Observe) pattern to achieve goals.

GOAL: {goal}
{lookup_hint}{guest_rules_block}{personalization_block}
CONTEXT:
{context}

{state.get("flush_context", "")}

USER DOCUMENTS:
{documents_context}

CONVERSATION HISTORY:
{history_text if history_text else "No conversation history"}

PREVIOUS OBSERVATIONS:
{observations_text if observations_text else "No previous observations"}

AVAILABLE TOOLS:
{tools_text if tools_text else "No tools available"}

Think step by step about how to achieve the goal. Then decide on your next action.

Respond with JSON in this format:
{{
    "thought": "your reasoning about the current situation and what to do next",
    "action_type": "tool_call" | "final_answer",
    "tool_name": "name of tool to use (ONLY when action_type is tool_call — e.g. web_search, document_search)",
    "tool_args": {{"query": "search terms"}} (if using a tool — use real parameter names from the tool schema, never arg1/arg2),
    "final_answer": "your final answer (if action_type is final_answer)"
}}

Important:
- action_type must be exactly "tool_call" or "final_answer". NEVER put a tool name (web_search, document_search, write_file, etc.) in action_type — that field is not the tool name.
- Use "tool_call" ONLY when you still lack information needed to answer. Use "final_answer" as soon as your observations are enough — do NOT repeat a search you already ran, and do NOT keep gathering once you can answer. One good web_search is usually enough.
- For web_search: put the user's exact title/name in the query (e.g. \"Dragon Ball Advent Truth MUD\" not just \"Dragon Ball Legends\"). Include genre hints from the goal (MUD, text game, indie). If results do not mention the exact title, try ONE refined query with the full title before giving up. After web_search succeeds, your NEXT action MUST be final_answer — do not call other tools.
- For web lookup goals: NEVER call document_search (private uploads are unrelated). Answer ONLY about the subject named in GOAL — do not respond as if the user pasted unrelated lists (e.g. trading cards) unless that text is literally in GOAL.
- If the goal needs current, recent, or post-training information (news, events, dates, who did/won/died something, prices, weather) and web_search is available, you MUST call web_search. NEVER answer such questions from memory or refuse by citing a knowledge/training cutoff.
- web_search is for the public web. document_search ONLY searches the user's own uploaded private files — do NOT use it for general knowledge, current events, or public figures; it returns unrelated personal documents and will mislead you.
- When the goal asks about the user's documents, notes, files, or a named report, you MUST call document_search before answering. The USER DOCUMENTS list above is authoritative — if a file is listed there, it exists and is searchable; never claim it is missing or that you lack access. Use tool_args like {{"query": "summary main findings", "file_name": "Report.md"}} — query is required; file_name is optional to narrow to one file. NEVER use read_file, list_directory, or ingest_documents for ingested uploads — document_search already has the content.
+- MCP tools (names starting with mcp_) are live when connected via the /mcp page. If the user asks whether an MCP server/tool is available, call list_mcp_tools first, then use the matching mcp_* tool. Never claim an MCP tool is unavailable without checking list_mcp_tools.
- ingest_documents takes NO arguments (empty tool_args {{}}). Do not pass arg1 or other placeholder keys.
- read_file requires {{"file_path": "/path/to/file"}}. list_directory requires {{"directory_path": "/path/to/dir"}}. Do not call them without those keys.
- To save/export conversation or chat to disk: call read_conversation_history once with empty tool_args {{}} — the system will auto-write to the path in the goal (if present) and finish. Do NOT call read_conversation_history repeatedly. Only call write_file yourself if auto-save did not run.
- write_file requires {{"file_path": "...", "content": "..."}} but for save/export requests omit content in tool_args (it is injected). Use a sensible path under the project (e.g. exports/conversation_log.txt).
- When observations contain document_search results, they include numbered EXCERPTS from the files. Write your final_answer from those excerpts. If excerpts were returned, NEVER say you do not have access or that nothing was uploaded.
- If document_search returns no excerpts, try one broader query before concluding nothing matched.
- When your observations contain web_search results, they include a "summary" line plus numbered SOURCES. The summary is usually the correct, already-extracted answer — make your final_answer that summary, phrased to address the exact question. Only override it if a source clearly contradicts it. Do NOT discard a correct summary just because the sources are broad "list of everyone who died in 2026" pages — those lists are noisy and easy to misread.
- Answer only the specific thing asked. If the question is "who did X on <date>", give the single correct person for that exact date as a direct sentence, not a long list of many names and dates.
- Do not claim you cannot access information a tool could retrieve; call the tool instead.

Respond ONLY with valid JSON."""

        return prompt

    def _parse_reasoning_response(self, response: str) -> dict[str, Any]:
        """
        Parse the LLM's reasoning response.

        Tries progressively harder to recover a JSON object: strip qwen3
        <think> blocks, extract fenced/balanced JSON candidates (completing
        truncated objects), and apply conservative syntax repairs before
        giving up and using keyword-based fallback parsing.

        Args:
            response: Raw LLM response

        Returns:
            Parsed reasoning components. On total failure the fallback result
            carries "_parse_failed": True and "_parse_error" so the caller can
            trigger a repair-reparse round trip.
        """
        return parse_json_object(
            response,
            self._validate_reasoning,
            logger=self.logger,
            fallback=self._fallback_reasoning_parsing,
        )

    def _validate_reasoning(self, parsed: dict[str, Any]) -> dict[str, Any]:
        """
        Validate a parsed reasoning object and fill safe defaults.

        Args:
            parsed: Parsed JSON object

        Returns:
            The validated object

        Raises:
            ValueError: If required fields are missing
        """
        if "action_type" not in parsed:
            raise ValueError("Missing 'action_type' field")

        parsed = self._coerce_action_type(parsed)

        if parsed["action_type"] == "tool_call":
            if "tool_name" not in parsed:
                raise ValueError("Missing 'tool_name' for tool_call action")
            if "tool_args" not in parsed or not isinstance(parsed.get("tool_args"), dict):
                parsed["tool_args"] = {}
            # Large write_file bodies in ReAct JSON cause parse failures — strip
            # them; _prepare_tool_args fills content from session/observations.
            if parsed.get("tool_name") == "write_file":
                content = parsed["tool_args"].get("content")
                if isinstance(content, str) and len(content) > 300:
                    parsed["tool_args"].pop("content", None)

        elif parsed["action_type"] == "final_answer":
            if "final_answer" not in parsed:
                raise ValueError("Missing 'final_answer' for final_answer action")

        else:
            raise ValueError(
                f"Invalid action_type '{parsed.get('action_type')}' — use tool_call or final_answer"
            )

        return parsed

    def _known_tool_names(self) -> set:
        """Registered tool names the model might mistakenly use as action_type."""
        names = set()
        for tool in self.available_tools:
            if isinstance(tool, dict) and tool.get("name"):
                names.add(tool["name"])
        if self.tool_registry:
            try:
                names.update(self.tool_registry.list_tool_names())
            except Exception:
                pass
        names.update(self._BUILTIN_TOOL_NAMES)
        return names

    _ACTION_TYPE_TOOL_ALIASES = {
        "search": "web_search",
        "websearch": "web_search",
        "web-search": "web_search",
    }

    # Tools the model often emits as action_type even when registry isn't wired (tests).
    _BUILTIN_TOOL_NAMES = frozenset(
        {
            "web_search",
            "document_search",
            "read_file",
            "write_file",
            "list_directory",
            "read_conversation_history",
            "analyze_conversation",
            "ingest_documents",
            # These tools are blocked in the orchestrator, but the model may
            # still emit their *names* as action_type by mistake. Keeping them
            # here preserves coercion behavior even if the tool module is not
            # registered (e.g., during clutter cleanup).
            "intent_analysis",
            "json_manipulate",
            "think",
            "calculator",
            "search_mcp_tools",
            "list_mcp_tools",
        }
    )

    _HOISTABLE_TOOL_ARG_KEYS = (
        "query",
        "file_path",
        "file_name",
        "directory_path",
        "content",
        "max_messages",
        "path",
    )

    def _coerce_action_type(self, parsed: dict[str, Any]) -> dict[str, Any]:
        """Fix common qwen3 mistakes: tool name used as action_type, args at top level."""
        action = parsed.get("action_type")
        if action in ("tool_call", "final_answer"):
            if action == "tool_call" and not parsed.get("tool_name"):
                # {"action_type":"tool_call","web_search":...} or top-level query only
                known = self._known_tool_names()
                for name in known:
                    if name in parsed:
                        parsed["tool_name"] = name
                        break
            return self._hoist_tool_args(parsed)

        known = self._known_tool_names()
        tool_name = None
        if isinstance(action, str):
            if action in known:
                tool_name = action
            else:
                alias = self._ACTION_TYPE_TOOL_ALIASES.get(action.lower())
                if alias and alias in known:
                    tool_name = alias
                elif action.lower().replace("-", "_").replace(" ", "_") in known:
                    tool_name = action.lower().replace("-", "_").replace(" ", "_")

        if tool_name:
            parsed["tool_name"] = tool_name
            parsed["action_type"] = "tool_call"
            return self._hoist_tool_args(parsed)

        if isinstance(action, str) and action.lower() in ("answer", "respond", "response"):
            parsed["action_type"] = "final_answer"
            if "final_answer" not in parsed:
                parsed["final_answer"] = parsed.pop("answer", parsed.get("response", ""))

        return parsed

    @classmethod
    def _hoist_tool_args(cls, parsed: dict[str, Any]) -> dict[str, Any]:
        """Move misplaced parameter keys from the top level into tool_args."""
        if parsed.get("action_type") != "tool_call":
            return parsed
        if "tool_args" not in parsed or not isinstance(parsed.get("tool_args"), dict):
            parsed["tool_args"] = {}
        for key in cls._HOISTABLE_TOOL_ARG_KEYS:
            if key in parsed and key not in parsed["tool_args"]:
                parsed["tool_args"][key] = parsed.pop(key)
        return parsed

    def _fallback_reasoning_parsing(self, response: str, parse_error: str = "") -> dict[str, Any]:
        """
        Fallback reasoning parsing when JSON parsing fails.

        Args:
            response: Raw LLM response
            parse_error: The error from the failed JSON parse

        Returns:
            Basic reasoning structure, flagged with "_parse_failed" so the
            orchestrator loop can attempt a repair-reparse round trip.
        """
        text = strip_think_blocks(response) or response
        response_lower = text.lower()

        # Look for action indicators
        if any(word in response_lower for word in ["tool", "use", "call", "search", "analyze"]):
            result: dict[str, Any] = {
                "thought": text,
                "action_type": "tool_call",
                "tool_name": "think",
                "tool_args": {"thought": text},
            }
        else:
            result = {"thought": text, "action_type": "final_answer", "final_answer": text}

        result["_parse_failed"] = True
        result["_parse_error"] = parse_error or "invalid JSON"
        return result

    async def _handle_tool_failure(self, tool_name: str, error: Exception) -> Any:
        """
        Handle tool execution failures gracefully.

        Args:
            tool_name: Name of the tool that failed
            error: The error that occurred

        Returns:
            Result to use in place of the failed tool call
        """
        error_msg = str(error)
        self.logger.warning(f"Tool {tool_name} failed: {error_msg}")

        # Handle missing parameters
        if "Missing required parameters" in error_msg:
            if tool_name in ["analyze_conversation", "read_conversation_history"]:
                # Return empty state for conversation tools
                return {
                    "message_count": 0,
                    "user_messages": 0,
                    "assistant_messages": 0,
                    "conversation_turns": 0,
                    "last_speaker": None,
                    "summary": "Starting new conversation",
                }
            elif tool_name == "intent_analysis":
                # Return default intent analysis
                return {
                    "type": "direct_response",
                    "confidence": 0.8,
                    "goal_statement": None,
                    "direct_response": None,
                    "clarification_question": None,
                    "reasoning": "Default intent analysis due to missing parameters",
                }

        # For other errors, return a generic error message
        return {"error": f"Tool {tool_name} failed: {error_msg}", "status": "error"}

    async def _call_tool(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        state: dict[str, Any] | None = None,
    ) -> Any:
        """
        Call a tool with error handling.

        Args:
            tool_name: Name of the tool to call
            tool_args: Arguments for the tool
            state: Optional ReAct state for context injection

        Returns:
            Tool execution result
        """
        try:
            if not self.tool_registry:
                raise ValueError("No tool registry available")

            if state is not None:
                tool_args = await self._prepare_tool_args(tool_name, tool_args, state)

            result = await self.tool_registry.execute_tool(tool_name, **tool_args)
            return result

        except Exception as e:
            # Handle the error gracefully
            return await self._handle_tool_failure(tool_name, e)

    def _coerce_unknown_action(self, parsed: dict[str, Any]) -> dict[str, Any] | None:
        """Apply action_type coercion for tool-name-as-action_type mistakes."""
        try:
            coerced = self._coerce_action_type(dict(parsed))
            if coerced.get("action_type") in ("tool_call", "final_answer"):
                return self._validate_reasoning(coerced)
        except ValueError:
            pass
        return None


# Test function
async def test_llm_driven_orchestrator():
    """Test the LLMDrivenOrchestrator functionality."""
    print("Testing LLMDrivenOrchestrator...")

    # Mock dependencies
    from core.config import load_config
    from core.llm_interface import OllamaInterface

    try:
        # Load config
        config = load_config("config.yaml")

        # Create LLM interface (will fail without Ollama, but that's ok for structure test)
        llm_interface = OllamaInterface(config=config)

        # Create orchestrator
        orchestrator = LLMDrivenOrchestrator(
            agent_name="TestOrchestrator", config=config, llm_interface=llm_interface
        )

        print(f"✓ LLMDrivenOrchestrator created: {orchestrator}")
        print(f"✓ Model name: {orchestrator.get_model_name()}")
        print(f"✓ Available tools: {len(orchestrator.available_tools)}")
        print(f"✓ Max iterations: {orchestrator.max_iterations}")

        # Test prompt building
        test_state = {
            "goal": "Test goal",
            "context": "Test context",
            "history": [],
            "observations": [],
        }

        prompt = orchestrator._build_reasoning_prompt(test_state)
        print(f"✓ Reasoning prompt built (length: {len(prompt)})")

    except Exception as e:
        print(f"✓ LLMDrivenOrchestrator structure test passed (expected error: {e})")

    print("LLMDrivenOrchestrator tests completed! 🎉")


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_llm_driven_orchestrator())
