# agents/wcca_intent_mixin.py
"""Intent analysis helpers for WitsControlCenterAgent."""

from typing import Any

from core.json_llm_parser import build_json_repair_prompt, parse_json_object
from core.schemas import ConversationHistory


class WCCAIntentMixin:
    """LLM-driven and heuristic intent analysis for the control center."""

    async def _analyze_user_intent(
        self,
        user_input: str,
        conversation_history: ConversationHistory | None,
    ) -> dict[str, Any]:
        """
        Analyze user input to determine intent and appropriate response strategy.

        Args:
            user_input: The user's input
            conversation_history: Conversation context

        Returns:
            Intent analysis with response strategy
        """
        # Document + web-search routing runs BEFORE the casual heuristic so
        # short factual questions and file references never get a tool-less reply.
        if await self._requires_orchestrator_for_input(user_input):
            needs_web = self._needs_web_search(user_input)
            needs_file = self._needs_file_write(user_input)
            if needs_file:
                notes = "Save/export to file - routing to orchestrator for read_conversation_history + write_file."
            elif needs_web:
                notes = "Needs current/external info or an explicit lookup - routing to orchestrator for web_search."
            else:
                notes = (
                    "References user documents/files/memory - routing to orchestrator for tool use."
                )
            return {
                "type": "task",
                "complexity": "moderate",
                "requires_tools": True,
                "suggested_response": "orchestrator",
                "notes": notes,
                "confidence": 0.8 if needs_web else 0.85,
            }

        if conversation_history and self._is_conversation_follow_up(
            user_input, conversation_history
        ):
            routing_message = self._follow_up_routing_message(
                user_input, conversation_history
            )
            if await self._requires_orchestrator_for_input(routing_message):
                return self._orchestrator_follow_up_intent(
                    "Follow-up to a prior task — routing to orchestrator with conversation context."
                )

        doc_inventory = await self._get_document_inventory()

        is_casual = self._is_casual_conversation(user_input, conversation_history)

        if is_casual:
            return {
                "type": "conversation",
                "complexity": "simple",
                "requires_tools": False,
                "suggested_response": "direct",
                "notes": "This appears to be casual conversation.",
                "confidence": 0.9,
            }

        if self.has_enhanced_capabilities and self.meta_reasoning and len(user_input.split()) > 8:
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

        history_context = ""
        if conversation_history and conversation_history.messages:
            recent_messages = conversation_history.get_recent_messages(
                min(10, self.config.agents.history_window)
            )
            history_context = "\n".join([f"{msg.role}: {msg.content}" for msg in recent_messages])

        prompt = self._build_intent_analysis_prompt(
            user_input, history_context, self._documents_context(doc_inventory)
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
            documents_context: Which user documents are ingested and searchable

        Returns:
            Formatted prompt for intent analysis
        """
        from core.personality_manager import get_personality_manager

        personality_manager = get_personality_manager()
        personality_prompt = personality_manager.get_system_prompt()

        prompt = f"""{personality_prompt}

You are analyzing user input to determine the best response strategy.

IMPORTANT: If the user identifies as "Richard Elliot" or mentions being "the creator" or "creator of Wits", recognize them as your creator and respond with appropriate respect and acknowledgment.

CONVERSATION HISTORY:
{history_context if history_context else "No previous conversation"}

USER DOCUMENTS:
{documents_context if documents_context else "No user documents are currently ingested."}

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

Guidelines:
- Use "goal_defined" for clear, actionable requests that need orchestration
- Use "clarification_question" for ambiguous requests needing more information
- Use "direct_response" for simple questions, greetings, or chat
- Short follow-ups ("yes", "that one", "summarize it", "look it up") after a prior user task or after you asked a clarifying question are usually "goal_defined" — read CONVERSATION HISTORY and continue the same task; do NOT treat them as casual chat
- If answering needs current, real-time, or post-training-cutoff information (news, recent or upcoming events, who won/died recently, prices, weather, sports results) OR the user says to "look it up"/"search", use "goal_defined" — the orchestrator has a web_search tool. Do NOT answer such questions from memory or claim a knowledge cutoff; route them so they get searched.
- Any request about the user's documents or files is "goal_defined" (it needs the document_search tool). The USER DOCUMENTS list above is authoritative: if a document is listed there, it exists and is accessible — never ask the user to confirm it or claim there is no record of it.
- For any request to 'remember', 'recall', or 'don't forget', use your semantic memory system (not file storage). Use the memory manager to store and retrieve facts for future conversations.

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
