"""Native Ollama tool-calling orchestrator (pilot — opt-in via config).

Every turn sends a fixed prefix (core/prompt_prefix.py) and passes each model round
through the tool-call guard (core/tool_call_guard.py).
"""

from __future__ import annotations

from typing import Any

from agents.base_orchestrator_agent import BaseOrchestratorAgent
from agents.native_loop_support import (
    NATIVE_SYSTEM_PROMPT,
    STEP_LIMIT_ERROR,
    assistant_tool_message,
    build_tool_specs,
    build_turn_context,
    decode_arguments,
    history_window,
    tool_call_event,
)
from core.config import WitsV3Config
from core.llm_interface import BaseLLMInterface, LLMMessage
from core.memory_manager import MemoryManager
from core.prompt_prefix import build_messages
from core.schemas import ConversationHistory
from core.tool_call_guard import OUT_OF_STEPS, TurnGuard
from core.turn_metrics import current_turn


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
        self.available_tools, self._tool_schemas = self._get_tool_specs()

    def _get_tool_specs(self) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
        return build_tool_specs(self.tool_registry, self._ORCHESTRATOR_TOOL_EXCLUDE)

    def _ollama_tools_payload(self) -> list[dict[str, Any]]:
        return self.available_tools

    async def _memory_context(self, goal: str) -> str:
        """Retrieved memory for this turn, or "" when none is found."""
        memories = await self.search_memory(goal, limit=5)
        return self._build_context_from_memories(memories) if memories else ""

    async def run(
        self,
        user_input: str,
        conversation_history: ConversationHistory | None = None,
        session_id: str | None = None,
        **kwargs,
    ):
        self.available_tools, self._tool_schemas = self._get_tool_specs()
        goal = user_input
        self.current_iteration = 0
        self._session_model = self.model_router.route(
            goal, default=self.get_model_name(), allow_trivial=False
        )
        self._skip_global_memory_store = kwargs.get("user_role") == "guest"
        turn = current_turn()
        if turn is not None:
            turn.path = "native"

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

        if react_state.get("preferred_tool") == "calculator":
            short = await self._try_math_shortcircuit(react_state, session_id)
            if short is not None:
                yield tool_call_event(self.agent_name, "calculator", True)
                yield self.stream_result(short)
                react_state["completed"] = True
                return

        chat_fn = getattr(self.llm_interface, "generate_chat", None)
        if not chat_fn:
            yield self.stream_error(
                "Native tool calling not supported by LLM interface; use json_react mode."
            )
            return

        react_state["context"] = await self._memory_context(goal)
        hist_turns = self.config.orchestrator.history_turns
        messages = build_messages(
            NATIVE_SYSTEM_PROMPT,
            "",
            history_window(react_state["history"], hist_turns),
            user_input,
            build_turn_context(
                goal, user_input, react_state["documents_context"], react_state["context"]
            ),
        )
        conv = self.config.conversation
        guard = TurnGuard(conv.max_steps, conv.max_identical_calls, conv.max_schema_retries)

        # Reason: rounds 1..max_steps may run tools and round max_steps+1 may still answer;
        # the guard turns any later tool call into `stop`, so this bound is only a backstop.
        while guard.rounds <= guard.max_steps:
            guard.note_round()
            self.current_iteration = guard.rounds
            response = await chat_fn(
                messages=messages,
                tools=self._ollama_tools_payload(),
                model=getattr(self, "_session_model", None),
                temperature=self.REASONING_TEMPERATURE,
            )
            calls = response.tool_calls or []
            stop_message = None
            if calls:
                # Reason: the model must see what it asked for before each reply, refused
                # calls included, so it can tell which reply answers which call.
                messages.append(assistant_tool_message(response.content or "", calls))
                for tc in calls:
                    verdict = guard.check(tc.get("name"), tc.get("arguments"), self._tool_schemas)
                    if verdict.action == "stop":
                        stop_message = verdict.message
                        break
                    if verdict.action == "reject":
                        messages.append(LLMMessage(role="tool", content=verdict.message))
                        continue
                    async for event in self._run_tool(tc, react_state, messages):
                        yield event
            else:
                text = response.content or ""
                verdict = guard.text_call_verdict(text, has_structured_call=False)
                if verdict is not None and verdict.action == "stop":
                    stop_message = verdict.message
                elif verdict is not None:
                    # Reason: a call written as text is never shown to the owner.
                    messages.append(LLMMessage(role="tool", content=verdict.message))
                else:
                    final_answer = text.strip() or "Done."
                    final_answer, done = self._resolve_final_answer(final_answer, react_state)
                    if done:
                        yield self.stream_result(final_answer)
                        react_state["completed"] = True
                        return
                    messages.append(
                        LLMMessage(role="user", content=react_state["observations"][-1])
                    )
            if stop_message is not None and stop_message != OUT_OF_STEPS:
                yield self.stream_error(stop_message)
                return
            if stop_message == OUT_OF_STEPS:
                break

        async for event in self._answer_out_of_steps(react_state, messages, chat_fn):
            yield event

    async def _run_tool(
        self, tc: dict[str, Any], state: dict[str, Any], messages: list[LLMMessage]
    ):
        """Run one guard-approved call and answer it with a role "tool" message."""
        tool_name = tc.get("name", "")
        tool_args = decode_arguments(tc.get("arguments"))
        block_reason = self._preflight_tool_call(tool_name, tool_args, state)
        if block_reason:
            observation = block_reason
        else:
            try:
                result = await self._call_tool(tool_name, tool_args, state)
                observation = self._format_tool_observation(tool_name, result)
            except Exception as e:
                observation = f"Tool {tool_name} failed: {e}"
            ok = not self._observation_indicates_failure(observation)
            yield tool_call_event(self.agent_name, tool_name, ok)
        state["observations"].append(observation)
        yield self.stream_observation(observation)
        messages.append(LLMMessage(role="tool", content=observation, name=tool_name))

    async def _answer_out_of_steps(
        self, state: dict[str, Any], messages: list[LLMMessage], chat_fn: Any
    ):
        """End a turn that ran out of steps with an answer from the observations."""
        fallback = self._auto_synthesize_from_observations(state)
        if not fallback and state["observations"]:
            # Reason: the observations are not search results the fallback can quote, so
            # ask once more with no tools offered; the model can then only answer.
            messages.append(LLMMessage(role="user", content=OUT_OF_STEPS))
            response = await chat_fn(
                messages=messages,
                tools=None,
                model=getattr(self, "_session_model", None),
                temperature=self.REASONING_TEMPERATURE,
            )
            text = (response.content or "").strip()
            if text and not response.tool_calls and "<tool_call>" not in text:
                fallback = text
        if fallback:
            yield self.stream_result(fallback)
        else:
            yield self.stream_error(STEP_LIMIT_ERROR)

    async def _try_math_shortcircuit(
        self, state: dict[str, Any], session_id: str | None
    ) -> str | None:
        """Skip first LLM call for pure calculator goals."""
        goal = state.get("goal", "")
        if not self._goal_is_pure_math(goal):
            return None
        args = self._normalize_calculator_args(
            {"expression": self._extract_calculator_expression(goal)}
        )
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
