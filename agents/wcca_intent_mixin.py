# agents/wcca_intent_mixin.py
"""Intent analysis helpers for WitsControlCenterAgent."""

from typing import Any

from core.json_llm_parser import build_json_repair_prompt, parse_json_object
from core.schemas import ConversationHistory

_SLIM_INTENT_INSTRUCTIONS = """You are a routing classifier for WitsV3, a local AI assistant.
Analyze the user message and pick the best response strategy. Do NOT answer the user — only classify.

Guidelines:
- Use "goal_defined" for clear, actionable requests that need orchestration or tools
- Use "clarification_question" for ambiguous requests needing more information
- Use "direct_response" ONLY for simple greetings, thanks, or small talk with no task
- Short follow-ups ("yes", "that one", "summarize it", "look it up") after a prior task are usually "goal_defined"
- If answering needs current/real-time info OR the user says to search/look it up, use "goal_defined"
- Any request about the user's documents or files is "goal_defined" (needs document_search)
- For "remember/recall/don't forget" facts, use "goal_defined" (memory tools handle it)

Respond ONLY with valid JSON."""


class WCCAIntentMixin:
    """LLM-driven and heuristic intent analysis for the control center."""

    async def _analyze_user_intent(
        self,
        user_input: str,
        conversation_history: ConversationHistory | None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Analyze user input to determine intent and appropriate response strategy.

        Args:
            user_input: The user's input
            conversation_history: Conversation context
            session_id: Session identifier for continuation routing

        Returns:
            Intent analysis with response strategy
        """
        decision = await self._classify_routing(user_input, conversation_history, session_id)
        intent = decision.to_intent()
        if intent is not None:
            return intent

        if (
            self.config.routing.enable_meta_reasoning_intent
            and self.has_enhanced_capabilities
            and self.meta_reasoning
            and len(user_input.split()) > 8
        ):
            try:
                context = {"source": "user_input"}
                if conversation_history:
                    context["history_length"] = str(len(conversation_history.messages))

                problem_space = await self.meta_reasoning.analyze_problem_space(user_input, context)

                return {
                    "type": "task",
                    "complexity": problem_space.complexity.value,
                    "requires_tools": len(problem_space.required_capabilities) > 0,
                    "required_capabilities": problem_space.required_capabilities,
                    "estimated_steps": problem_space.estimated_steps,
                    "confidence": problem_space.confidence,
                    "suggested_response": (
                        "orchestrator"
                        if problem_space.complexity.value in ["complex", "research"]
                        else "direct"
                    ),
                    "notes": f"Task analyzed with meta-reasoning: {problem_space.complexity.value} complexity.",
                }
            except Exception as e:
                self.logger.error(f"Error using enhanced capabilities for intent analysis: {e}")

        doc_inventory = await self._get_document_inventory()
        history_turns = self.config.routing.intent_history_turns
        history_context = ""
        if conversation_history and conversation_history.messages:
            recent_messages = conversation_history.get_recent_messages(history_turns)
            history_context = "\n".join([f"{msg.role}: {msg.content}" for msg in recent_messages])

        documents_context = ""
        if self._message_references_documents(user_input, doc_inventory):
            documents_context = self._documents_context(doc_inventory)

        prompt = self._build_intent_analysis_prompt(
            user_input, history_context, documents_context
        )

        try:
            response = await self.generate_response(
                prompt=prompt,
                max_tokens=1024,
                temperature=self.INTENT_ANALYSIS_TEMPERATURE,
                response_format="json",
            )

            intent_analysis = self._parse_intent_response(response)

            if intent_analysis.pop("_parse_failed", False):
                parse_error = intent_analysis.pop("_parse_error", "invalid JSON")
                self.logger.warning(
                    "Intent response was malformed (%s); attempting JSON repair",
                    parse_error,
                )
                try:
                    repaired_response = await self.generate_response(
                        self._build_intent_json_repair_prompt(response, parse_error),
                        max_tokens=1024,
                        temperature=self.INTENT_ANALYSIS_TEMPERATURE,
                        response_format="json",
                    )
                    reparsed = self._parse_intent_response(repaired_response)
                    if not reparsed.pop("_parse_failed", False):
                        reparsed.pop("_parse_error", None)
                        intent_analysis = reparsed
                    else:
                        self.logger.warning(
                            "Intent JSON repair attempt also failed; using heuristic fallback"
                        )
                except Exception as repair_error:
                    self.logger.warning(
                        "Intent JSON repair attempt errored: %s; using heuristic fallback",
                        repair_error,
                    )

            return intent_analysis
        except Exception as e:
            self.logger.error(f"Error in intent analysis: {e}")
            return self._fallback_intent_parsing(user_input)

    def _build_intent_analysis_prompt(
        self, user_input: str, history_context: str, documents_context: str = ""
    ) -> str:
        """
        Build the prompt for intent analysis.

        Args:
            user_input: User's input
            history_context: Recent conversation context
            documents_context: Which user documents are ingested (empty when irrelevant)

        Returns:
            Formatted prompt for intent analysis
        """
        if self.config.routing.slim_intent_prompt:
            header = _SLIM_INTENT_INSTRUCTIONS
        else:
            from core.personality_manager import get_personality_manager

            header = (
                get_personality_manager().get_system_prompt()
                + "\n\nYou are analyzing user input to determine the best response strategy."
            )

        doc_block = ""
        if documents_context:
            doc_block = f"\nUSER DOCUMENTS:\n{documents_context}\n"

        prompt = f"""{header}

CONVERSATION HISTORY:
{history_context if history_context else "No previous conversation"}
{doc_block}
USER INPUT: {user_input}

Analyze this input and respond with JSON containing:
{{
    "type": "goal_defined" | "clarification_question" | "direct_response",
    "confidence": 0.0-1.0,
    "reasoning": "your reasoning",
    "goal_statement": "clear goal if type is goal_defined",
    "clarification_question": "question if type is clarification_question",
    "direct_response": "response if type is direct_response"
}}

Respond ONLY with valid JSON."""

        return prompt

    def _validate_intent(self, parsed: dict[str, Any]) -> dict[str, Any]:
        """Validate a parsed intent object and apply routing metadata."""
        if "type" not in parsed:
            raise ValueError("Missing 'type' field")
        return self._normalize_parsed_intent(parsed)

    def _parse_intent_response(self, response: str) -> dict[str, Any]:
        """
        Parse the LLM's intent analysis response.

        Uses the same progressive JSON recovery as the orchestrator ReAct loop.

        Args:
            response: Raw LLM response

        Returns:
            Parsed intent analysis. On total failure the fallback result carries
            "_parse_failed": True so the caller can attempt a repair-reparse.
        """
        return parse_json_object(
            response,
            self._validate_intent,
            logger=self.logger,
            fallback=self._fallback_intent_parsing,
        )

    def _build_intent_json_repair_prompt(self, raw_response: str, parse_error: str) -> str:
        """Build a prompt asking the model to rewrite malformed intent JSON."""
        return build_json_repair_prompt(
            raw_response,
            parse_error,
            required_keys=(
                '"type", "confidence", "reasoning", "goal_statement", '
                '"clarification_question", "direct_response"'
            ),
        )

    def _fallback_intent_parsing(self, response: str, parse_error: str = "") -> dict[str, Any]:
        """
        Fallback intent parsing when JSON parsing fails.

        Args:
            response: Raw LLM response
            parse_error: Optional parse error from json_llm_parser

        Returns:
            Basic intent analysis
        """
        response_lower = response.lower()

        if any(
            word in response_lower
            for word in ["unclear", "clarify", "question", "what do you mean"]
        ):
            return self._flag_intent_parse_failure(
                self._normalize_parsed_intent(
                    {
                        "type": "clarification_question",
                        "confidence": 0.6,
                        "reasoning": "Response suggests need for clarification",
                        "clarification_question": "Could you please provide more details about what you'd like me to help you with?",
                    }
                ),
                parse_error,
            )
        if any(
            word in response_lower for word in ["hello", "hi", "how are you", "what can you do"]
        ):
            return self._flag_intent_parse_failure(
                self._normalize_parsed_intent(
                    {
                        "type": "direct_response",
                        "confidence": 0.8,
                        "reasoning": "Greeting or general question",
                        "direct_response": "Hello! I'm WITS, your AI assistant. I can help you with various tasks and questions.",
                    }
                ),
                parse_error,
            )
        return self._flag_intent_parse_failure(
            self._normalize_parsed_intent(
                {
                    "type": "goal_defined",
                    "confidence": 0.5,
                    "reasoning": "Defaulting to task delegation",
                    "goal_statement": response,
                }
            ),
            parse_error,
        )

    @staticmethod
    def _flag_intent_parse_failure(result: dict[str, Any], parse_error: str) -> dict[str, Any]:
        if parse_error:
            result["_parse_failed"] = True
            result["_parse_error"] = parse_error
        return result
