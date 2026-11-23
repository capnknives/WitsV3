"""Native Ollama tool-calling orchestrator (pilot — opt-in via config).

Every turn sends a fixed prefix (core/prompt_prefix.py) and passes each model round
through the tool-call guard (core/tool_call_guard.py).
"""

from __future__ import annotations

import time
from typing import Any

from agents.base_orchestrator_agent import BaseOrchestratorAgent
from agents.native_loop_support import (
    MEMORY_FETCH_LIMIT,
    NEVER_RECALLED_TYPES,
    STEP_LIMIT_ERROR,
    as_sent_history,
    assistant_tool_message,
    build_turn_context,
    decode_arguments,
    estimate_tokens,
    history_window,
    model_call_error,
    native_system_prompt,
    recalled_facts,
    resident_model,
    store_sent_context,
    tool_call_event,
    tool_reply,
)
from agents.tool_tiers import (
    FIND_TOOLS,
    LOOP_TOOLS,
    ApprovalGate,
    TierSpecs,
    build_tier_specs,
    find_tools_reply,
    route_call,
)
from agents.turn_context import bind_turn_if_unbound
from core.config import WitsV3Config
from core.llm_interface import BaseLLMInterface, LLMMessage
from core.memory_manager import MemoryManager
from core.prompt_prefix import build_messages, context_block
from core.schemas import ConversationHistory
from core.tool_call_guard import OUT_OF_STEPS, TurnGuard
from core.tool_context import ToolContext, use_tool_context
from core.turn_metrics import current_turn, record_tool
from core.untrusted_content import (
    UntrustedWriteGate,
    is_untrusted_source,
    mark_untrusted,
    turn_gate,
    write_gate_enabled,
)


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
        self._load_tool_specs()

    def _load_tool_specs(self) -> None:
        """Resolve the tool tiers; only the ``always`` tier is sent to the model."""
        self._specs: TierSpecs = build_tier_specs(
            self.tool_registry, self._ORCHESTRATOR_TOOL_EXCLUDE, self.config.conversation.tools
        )
        self.available_tools, self._tool_schemas = self._specs.payload, self._specs.schemas

    def _ollama_tools_payload(self) -> list[dict[str, Any]]:
        return self.available_tools

    async def _memory_context(self, goal: str) -> str:
        """Durable owner-stated facts for this turn, or "" when none qualify."""
        conv = self.config.conversation
        # Reason: the store is mostly old-path transcript rows that would crowd facts out
        # of a plain top-N, so each allowed type is queried on its own (equality filter);
        # recalled_facts re-checks the type for backends that ignore the filter.
        found: list[Any] = []
        for seg_type in sorted({t.upper() for t in conv.memory_types} - NEVER_RECALLED_TYPES):
            found += await self.search_memory(
                goal, limit=MEMORY_FETCH_LIMIT, filter_dict={"type": seg_type}
            )
        found.sort(key=lambda seg: -(getattr(seg, "relevance_score", None) or 0.0))
        return recalled_facts(found, conv.memory_types, conv.memory_min_score)

    @bind_turn_if_unbound
    async def run(
        self,
        user_input: str,
        conversation_history: ConversationHistory | None = None,
        session_id: str | None = None,
        **kwargs,
    ):
        self._load_tool_specs()
        goal = user_input
        self.current_iteration = 0
        # Reason: one resident model, never the router, so no turn can ask for a model the
        # owner has not installed and the model stays loaded between rounds.
        self._session_model = resident_model(self.config)
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
        # Reason: kept in this turn's state, not on self, so concurrent turns on one
        # orchestrator never see each other's context.
        react_state["tool_context"] = ToolContext(
            session_id=session_id,
            # Reason: the role exactly as the caller sent it; "owner" when absent only
            # because the front door already sends every non-owner turn to the old path.
            user_role=kwargs.get("user_role", "owner"),
            conversation_history=conversation_history,
            turn_id=turn.turn_id if turn is not None else None,
        )
        # Reason: per turn, like tool_context, so a read in one turn never gates the next.
        # Reason: the gate asks python_sandboxed() at each decision, from load_config() as the
        # python tool does, so it never trusts a turn-start snapshot of the sandbox mode.
        react_state["untrusted_gate"] = UntrustedWriteGate(write_gate_enabled(self.config))

        chat_fn = getattr(self.llm_interface, "generate_chat", None)
        if not chat_fn:
            yield self.stream_error(
                "Native tool calling not supported by LLM interface; use json_react mode."
            )
            return

        react_state["context"] = await self._memory_context(goal)
        hist_turns = self.config.orchestrator.history_turns
        # Reason: with nothing ingested the documents line is left out entirely.
        documents = react_state["documents_context"] if doc_inventory else ""
        turn_context = build_turn_context(goal, user_input, documents, react_state["context"])
        # Reason: store the block now so the next turn resends this message exactly as sent.
        store_sent_context(current_message, context_block(turn_context))
        # Reason: the tool list is fixed for the turn, so its size is estimated once.
        system_prompt = native_system_prompt(self._session_model)
        base_tokens = estimate_tokens(self._ollama_tools_payload()) + estimate_tokens(system_prompt)
        window = history_window(
            react_state["history"],
            hist_turns,
            token_budget=0.6 * self.config.ollama_settings.num_ctx,
            base_tokens=base_tokens,
        )
        messages = build_messages(system_prompt, "", window, user_input, turn_context)
        conv = self.config.conversation
        guard = TurnGuard(conv.max_steps, conv.max_identical_calls, conv.max_schema_retries)
        approvals = ApprovalGate()

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
            stop_message, stop_answers = None, False
            if calls:
                # Reason: the model must see what it asked for before each reply, refused
                # calls included, so it can tell which reply answers which call.
                messages.append(assistant_tool_message(response.content or "", calls))
                for index, tc in enumerate(calls):
                    # Reason: use_tool is unwrapped first so every rule, count and event
                    # applies to the real tool name.
                    route = route_call(tc.get("name"), tc.get("arguments"), self._specs.tiers)
                    if route.reject:
                        messages.append(tool_reply(tc, route.reject))
                        continue
                    if self._specs.tiers.approval_name(route.name):
                        verdict = approvals.check(route.name)
                        stop_answers = verdict.action == "stop"
                    else:
                        verdict = guard.check(route.name, route.arguments, self._tool_schemas)
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
                    if route.name == FIND_TOOLS:
                        yield self._find_tools(tc, route.arguments, react_state, messages)
                        continue
                    if route.name in LOOP_TOOLS:
                        # Reason: use_tool must have been unwrapped by now; never hand a
                        # loop-only name to the registry.
                        messages.append(tool_reply(tc, f"'{route.name}' cannot run here."))
                        continue
                    run = self._run_tool(tc, react_state, messages, route.name, route.arguments)
                    async for event in run:
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
            if stop_answers:
                # Reason: a refused approval is an answer to the owner, not a failure.
                yield self.stream_result(stop_message)
                return
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

    def _find_tools(
        self, tc: dict[str, Any], arguments: Any, state: dict[str, Any], messages: list
    ) -> Any:
        """Answer a guard-approved find_tools call from the lookup tier."""
        started = time.perf_counter()
        query = (decode_arguments(arguments) or {}).get("query", "")
        observation = find_tools_reply(query, self._specs.lookup)
        record_tool(FIND_TOOLS, True, (time.perf_counter() - started) * 1000)
        state["observations"].append(observation)
        messages.append(tool_reply(tc, observation))
        return tool_call_event(self.agent_name, FIND_TOOLS, True)

    async def _run_tool(
        self,
        tc: dict[str, Any],
        state: dict[str, Any],
        messages: list[LLMMessage],
        tool_name: str,
        arguments: Any,
    ):
        """Run one guard-approved call and answer ``tc`` with a role "tool" message.

        ``tool_name`` and ``arguments`` are the real tool's, after use_tool is unwrapped.
        """
        # Reason: the guard accepted these arguments, and it reads null as no arguments.
        tool_args = decode_arguments(arguments) or {}
        gate = turn_gate(state)
        block_reason = self._preflight_tool_call(tool_name, tool_args, state)
        block_reason = block_reason or gate.block_reason(tool_name)
        untrusted = is_untrusted_source(tool_name)
        if block_reason:
            observation = block_reason
        else:
            # Reason: taint before the call, so a source that raises still gates the turn.
            gate.note_dispatch(tool_name)
            try:
                # Reason: state=None keeps _prepare_tool_args from merging loop state (such
                # as user_role) into the arguments the guard validated.
                # Reason: the context is set around each call, not across the whole
                # generator, because a ContextVar held over a yield would leak into the
                # consumer and could be reset from another Context.
                with use_tool_context(state["tool_context"]):
                    result = await self._call_tool(tool_name, tool_args, None)
                observation = self._format_tool_observation(tool_name, result)
                ok = not self._tool_result_is_failure(result)
            except Exception as e:
                observation = f"Tool {tool_name} failed: {e}"
                ok = False
            yield tool_call_event(self.agent_name, tool_name, ok)
        state["observations"].append(observation)
        yield self.stream_observation(observation)
        # Reason: only the copy sent to the model is wrapped; observations stay bare for
        # the owner-facing synthesis, and the only observation re-sent to a model (the
        # retry after a synthesis/save guard) is the guard's own message.
        reply = mark_untrusted(tool_name, observation) if untrusted and not block_reason else None
        messages.append(tool_reply(tc, reply or observation))

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
