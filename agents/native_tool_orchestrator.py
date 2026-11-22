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
    as_sent_history,
    assistant_tool_message,
    build_tool_specs,
    build_turn_context,
    decode_arguments,
    estimate_tokens,
    history_window,
    model_call_error,
    resident_model,
    store_sent_context,
    tool_call_event,
    tool_reply,
)
from core.config import WitsV3Config
from core.llm_interface import BaseLLMInterface, LLMMessage
from core.memory_manager import MemoryManager
from core.prompt_prefix import build_messages, context_block
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
        # Reason: one resident model, never the router, so no turn can ask for a model the
        # owner has not installed and the model stays loaded between rounds.
        self._session_model = resident_model(self.config)
        self._skip_global_memory_store = kwargs.get("user_role") == "guest"
        turn = current_turn()
        if turn is not None:
            turn.path = "native"

        doc_inventory = await self._get_document_inventory()
        history, current_message = as_sent_history(conversation_history, user_input)
        react_state = {
            "goal": goal,
            "context": "",
            "documents_context": self._format_documents_context(doc_inventory),
            "conversation_history": conversation_history,
            # Reason: built from the stored messages, not to_llm_format, so earlier user
            # messages carry the context block they were sent with; history_window trims.
            "history": history,
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

        chat_fn = getattr(self.llm_interface, "generate_chat", None)
        if not chat_fn:
            yield self.stream_error(
                "Native tool calling not supported by LLM interface; use json_react mode."
            )
            return

        react_state["context"] = await self._memory_context(goal)
        hist_turns = self.config.orchestrator.history_turns
        turn_context = build_turn_context(
            goal, user_input, react_state["documents_context"], react_state["context"]
        )
        # Reason: store the block now so the next turn resends this message exactly as sent.
        store_sent_context(current_message, context_block(turn_context))
        # Reason: the tool list is fixed for the turn, so its size is estimated once.
        base_tokens = estimate_tokens(self._ollama_tools_payload()) + estimate_tokens(
            NATIVE_SYSTEM_PROMPT
        )
        window = history_window(
            react_state["history"],
            hist_turns,
            token_budget=0.6 * self.config.ollama_settings.num_ctx,
            base_tokens=base_tokens,
        )
        messages = build_messages(NATIVE_SYSTEM_PROMPT, "", window, user_input, turn_context)
        conv = self.config.conversation
        guard = TurnGuard(conv.max_steps, conv.max_identical_calls, conv.max_schema_retries)

        # Reason: rounds 1..max_steps may run tools and round max_steps+1 may still answer;
        # the guard turns any later tool call into `stop`, so this bound is only a backstop.
        while guard.rounds <= guard.max_steps:
            guard.note_round()
            self.current_iteration = guard.rounds
            try:
                response = await self._chat(chat_fn, messages, self._ollama_tools_payload())
            except Exception as exc:
                yield self.stream_error(model_call_error(exc, self._session_model))
                return
            calls = response.tool_calls or []
            stop_message = None
            if calls:
                # Reason: the model must see what it asked for before each reply, refused
                # calls included, so it can tell which reply answers which call.
                messages.append(assistant_tool_message(response.content or "", calls))
                for index, tc in enumerate(calls):
                    verdict = guard.check(tc.get("name"), tc.get("arguments"), self._tool_schemas)
                    if verdict.action == "stop":
                        stop_message = verdict.message
                        # Reason: every call in the assistant message needs a reply, or
                        # the next request carries an unanswered tool call.
                        for rest in calls[index:]:
                            messages.append(tool_reply(rest, verdict.message))
                        break
                    if verdict.action == "reject":
                        messages.append(tool_reply(tc, verdict.message))
                        continue
                    async for event in self._run_tool(tc, react_state, messages):
                        yield event
            else:
                text = response.content or ""
                verdict = guard.text_call_verdict(text, has_structured_call=False)
                if verdict is not None:
                    # Reason: a call written as text is never shown to the owner; it has no
                    # structured call to answer, so the verdict goes back as a user message.
                    messages.append(LLMMessage(role="assistant", content=text))
                    if verdict.action == "stop":
                        stop_message = verdict.message
                    else:
                        messages.append(LLMMessage(role="user", content=verdict.message))
                else:
                    final_answer = text.strip() or "Done."
                    final_answer, done = self._resolve_final_answer(final_answer, react_state)
                    if done:
                        yield self.stream_result(final_answer)
                        react_state["completed"] = True
                        return
                    messages.append(LLMMessage(role="assistant", content=text))
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

    async def _chat(self, chat_fn: Any, messages: list[LLMMessage], tools: Any) -> Any:
        """One model call with this turn's model and temperature."""
        return await chat_fn(
            messages=messages,
            tools=tools,
            model=self._session_model,
            temperature=self.REASONING_TEMPERATURE,
        )

    async def _run_tool(
        self, tc: dict[str, Any], state: dict[str, Any], messages: list[LLMMessage]
    ):
        """Run one guard-approved call and answer it with a role "tool" message."""
        tool_name = tc.get("name", "")
        # Reason: the guard accepted these arguments, and it reads null as no arguments.
        tool_args = decode_arguments(tc.get("arguments")) or {}
        block_reason = self._preflight_tool_call(tool_name, tool_args, state)
        if block_reason:
            observation = block_reason
        else:
            try:
                # Reason: state=None keeps _prepare_tool_args from merging loop state (such
                # as user_role) into the arguments the guard validated.
                result = await self._call_tool(tool_name, tool_args, None)
                observation = self._format_tool_observation(tool_name, result)
                ok = not self._tool_result_is_failure(result)
            except Exception as e:
                observation = f"Tool {tool_name} failed: {e}"
                ok = False
            yield tool_call_event(self.agent_name, tool_name, ok)
        state["observations"].append(observation)
        yield self.stream_observation(observation)
        messages.append(tool_reply(tc, observation))

    async def _answer_out_of_steps(
        self, state: dict[str, Any], messages: list[LLMMessage], chat_fn: Any
    ):
        """End a turn that ran out of steps with an answer from what it has."""
        fallback = self._auto_synthesize_from_observations(state)
        if not fallback:
            # Reason: nothing quotable, so ask once more with no tools offered; the model
            # can then only answer.
            messages.append(LLMMessage(role="user", content=OUT_OF_STEPS))
            try:
                response = await self._chat(chat_fn, messages, None)
            except Exception as exc:
                yield self.stream_error(model_call_error(exc, self._session_model))
                return
            text = (response.content or "").strip()
            if text and not response.tool_calls and "<tool_call>" not in text:
                fallback = text
        if fallback:
            yield self.stream_result(fallback)
        else:
            yield self.stream_error(STEP_LIMIT_ERROR)

    def _build_reasoning_prompt(self, state: dict[str, Any]) -> str:
        return ""

    def _parse_reasoning_response(self, response: str) -> dict[str, Any]:
        return {"action_type": "final_answer", "final_answer": response}
