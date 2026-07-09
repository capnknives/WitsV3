"""Native Ollama tool-calling orchestrator (pilot — opt-in via config)."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

from agents.base_orchestrator_agent import BaseOrchestratorAgent
from core.config import WitsV3Config
from core.llm_interface import BaseLLMInterface, LLMMessage
from core.memory_manager import MemoryManager
from core.schemas import ConversationHistory, StreamData


class NativeToolOrchestrator(BaseOrchestratorAgent):
    """Thin tool-calling loop using Ollama /api/chat tools API."""

    _ORCHESTRATOR_TOOL_EXCLUDE = frozenset({"intent_analysis", "json_manipulate"})

    def __init__(
        self,
        agent_name: str,
        config: WitsV3Config,
        llm_interface: BaseLLMInterface,
        memory_manager: MemoryManager | None = None,
        tool_registry: Any | None = None,
    ):
        super().__init__(agent_name, config, llm_interface, memory_manager, tool_registry)
        self.available_tools = self._get_available_tools()

    def _get_available_tools(self) -> list[dict[str, Any]]:
        if not self.tool_registry:
            return []
        tools = []
        for name, tool in self.tool_registry.tools.items():
            if name in self._ORCHESTRATOR_TOOL_EXCLUDE:
                continue
            schema = tool.get_schema() if hasattr(tool, "get_schema") else {}
            desc = tool.get_llm_description()
            description = desc.get("description", "") if isinstance(desc, dict) else str(desc)
            if not isinstance(description, str):
                description = str(description)
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": description,
                        "parameters": schema.get("parameters", {"type": "object", "properties": {}}),
                    },
                }
            )
        return tools

    def _ollama_tools_payload(self) -> list[dict[str, Any]]:
        return self.available_tools

    async def run(
        self,
        user_input: str,
        conversation_history: ConversationHistory | None = None,
        session_id: str | None = None,
        **kwargs,
    ):
        self.available_tools = self._get_available_tools()
        goal = user_input
        self.current_iteration = 0
        self._session_model = self.model_router.route(
            goal, default=self.get_model_name(), allow_trivial=False
        )
        self._skip_global_memory_store = kwargs.get("user_role") == "guest"

        yield self.stream_thinking(f"Starting native tool orchestration for: {goal}")

        doc_inventory = await self._get_document_inventory()
        react_state = {
            "goal": goal,
            "context": "",
            "documents_context": self._format_documents_context(doc_inventory),
            "conversation_history": conversation_history,
            "history": conversation_history.to_llm_format() if conversation_history else [],
            "observations": [],
            "completed": False,
            "final_answer": None,
            "tool_repeat_failures": {},
            "tool_total_failures": {},
            "lookup_search_done": False,
            "synthesis_guard_retries": 0,
            "user_role": kwargs.get("user_role", "owner"),
            "guest_profile": kwargs.get("guest_profile") or {},
            "guest_age_band": "teen",
            "guest_personalization_context": kwargs.get("guest_personalization_context", ""),
            "preferred_tool": kwargs.get("preferred_tool", ""),
        }
        self._react_state_for_tools = react_state

        async for stream_data in self._bootstrap_codebase_intro(react_state, session_id):
            yield stream_data

        messages: list[LLMMessage] = [
            LLMMessage(
                role="system",
                content=(
                    "You are WitsV3 orchestrator. Use tools when needed, then answer the user. "
                    f"Goal: {goal}\n{react_state['documents_context']}"
                ),
            ),
        ]
        hist_turns = self.config.orchestrator.history_turns
        for msg in react_state["history"][-hist_turns:]:
            messages.append(LLMMessage(role=msg["role"], content=msg["content"]))

        while not react_state["completed"] and self.current_iteration < self.max_iterations:
            self.current_iteration += 1
            yield self.stream_thinking(
                f"Native tool iteration {self.current_iteration}/{self.max_iterations}"
            )

            if self.current_iteration == 1 and react_state.get("preferred_tool") == "calculator":
                short = await self._try_math_shortcircuit(react_state, session_id)
                if short is not None:
                    yield self.stream_result(short)
                    react_state["completed"] = True
                    return

            chat_fn = getattr(self.llm_interface, "generate_chat", None)
            if not chat_fn:
                yield self.stream_error(
                    "Native tool calling not supported by LLM interface; use json_react mode."
                )
                return

            response = await chat_fn(
                messages=messages,
                tools=self._ollama_tools_payload(),
                model=getattr(self, "_session_model", None),
                temperature=self.REASONING_TEMPERATURE,
            )

            if response.tool_calls:
                for tc in response.tool_calls:
                    tool_name = tc.get("name", "")
                    tool_args = tc.get("arguments") or {}
                    if isinstance(tool_args, str):
                        try:
                            tool_args = json.loads(tool_args)
                        except json.JSONDecodeError:
                            tool_args = {}
                    yield self.stream_action(f"Calling tool: {tool_name}")
                    block_reason = self._preflight_tool_call(tool_name, tool_args, react_state)
                    if block_reason:
                        observation = block_reason
                    else:
                        try:
                            result = await self._call_tool(tool_name, tool_args, react_state)
                            observation = self._format_tool_observation(tool_name, result)
                        except Exception as e:
                            observation = f"Tool {tool_name} failed: {e}"
                    react_state["observations"].append(observation)
                    yield self.stream_observation(observation)
                    messages.append(
                        LLMMessage(role="tool", content=observation, name=tool_name)
                    )
                continue

            final_answer = (response.content or "").strip() or "Done."
            final_answer, done = self._resolve_final_answer(final_answer, react_state)
            if not done:
                messages.append(
                    LLMMessage(role="user", content=react_state["observations"][-1])
                )
                continue
            yield self.stream_result(final_answer)
            react_state["completed"] = True
            return

        fallback = self._auto_synthesize_from_observations(react_state)
        if fallback:
            yield self.stream_result(fallback)
        else:
            yield self.stream_error("Could not complete within iteration limit.")

    async def _try_math_shortcircuit(
        self, state: dict[str, Any], session_id: str | None
    ) -> str | None:
        """Skip first LLM call for pure calculator goals."""
        goal = state.get("goal", "")
        if not self._goal_is_pure_math(goal):
            return None
        args = self._normalize_calculator_args({"expression": self._extract_calculator_expression(goal)})
        try:
            result = await self._call_tool("calculator", args, state)
            obs = self._format_tool_observation("calculator", result)
            state["observations"].append(obs)
            if isinstance(result, dict):
                return str(result.get("result", result.get("content", obs)))
            return str(result)
        except Exception:
            return None

    def _build_reasoning_prompt(self, state: dict[str, Any]) -> str:
        return ""

    def _parse_reasoning_response(self, response: str) -> dict[str, Any]:
        return {"action_type": "final_answer", "final_answer": response}
