# agents/wcca_intent_mixin.py
"""Intent analysis helpers for WitsControlCenterAgent."""

import json
from typing import Any, Dict, Optional

from core.schemas import ConversationHistory


class WCCAIntentMixin:
    """LLM-driven and heuristic intent analysis for the control center."""

    async def _analyze_user_intent(
        self,
        user_input: str,
        conversation_history: Optional[ConversationHistory],
    ) -> Dict[str, Any]:
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
                notes = "References user documents/files/memory - routing to orchestrator for tool use."
            return {
                "type": "task",
                "complexity": "moderate",
                "requires_tools": True,
                "suggested_response": "orchestrator",
                "notes": notes,
                "confidence": 0.8 if needs_web else 0.85,
            }

        doc_inventory = await self._get_document_inventory()

        # Check if this is casual conversation with a simple heuristic
        is_casual = self._is_casual_conversation(user_input)

        if is_casual:
            return {
                "type": "conversation",
                "complexity": "simple",
                "requires_tools": False,
                "suggested_response": "direct",
                "notes": "This appears to be casual conversation.",
                "confidence": 0.9,
            }

        # For task-oriented messages, use enhanced capabilities if available
        if self.has_enhanced_capabilities and self.meta_reasoning and len(user_input.split()) > 8:
            try:
                # Use meta-reasoning to analyze complexity
                context = {"source": "user_input"}
                if conversation_history:
                    context["history_length"] = str(len(conversation_history.messages))

                problem_space = await self.meta_reasoning.analyze_problem_space(user_input, context)

                # Convert problem space to intent analysis
                return {
                    "type": "task",
                    "complexity": problem_space.complexity.value,
                    "requires_tools": len(problem_space.required_capabilities) > 0,
                    "required_capabilities": problem_space.required_capabilities,
                    "estimated_steps": problem_space.estimated_steps,
                    "confidence": problem_space.confidence,
                    "suggested_response": "orchestrator" if problem_space.complexity.value in ["complex", "research"] else "direct",
                    "notes": f"Task analyzed with meta-reasoning: {problem_space.complexity.value} complexity.",
                }
            except Exception as e:
                self.logger.error(f"Error using enhanced capabilities for intent analysis: {e}")
                # Fall back to traditional analysis

        # Build context from conversation history
        history_context = ""
        if conversation_history and conversation_history.messages:
            recent_messages = conversation_history.get_recent_messages(
                min(10, self.config.agents.history_window)
            )
            history_context = "\n".join([
                f"{msg.role}: {msg.content}" for msg in recent_messages
            ])

        # Build the intent analysis prompt
        prompt = self._build_intent_analysis_prompt(
            user_input, history_context, self._documents_context(doc_inventory)
        )

        try:
            # Get LLM response
            response = await self.llm_interface.generate_text(
                prompt=prompt,
                max_tokens=1024,
                temperature=0.3,
            )

            # Parse the response
            intent_analysis = self._parse_intent_response(response)

            return intent_analysis
        except Exception as e:
            self.logger.error(f"Error in intent analysis: {e}")
            # Return a fallback analysis
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
        # Get personality-based system prompt
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
- If answering needs current, real-time, or post-training-cutoff information (news, recent or upcoming events, who won/died recently, prices, weather, sports results) OR the user says to "look it up"/"search", use "goal_defined" — the orchestrator has a web_search tool. Do NOT answer such questions from memory or claim a knowledge cutoff; route them so they get searched.
- Any request about the user's documents or files is "goal_defined" (it needs the document_search tool). The USER DOCUMENTS list above is authoritative: if a document is listed there, it exists and is accessible — never ask the user to confirm it or claim there is no record of it.
- For any request to 'remember', 'recall', or 'don't forget', use your semantic memory system (not file storage). Use the memory manager to store and retrieve facts for future conversations.

Respond ONLY with valid JSON."""

        return prompt

    def _parse_intent_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the LLM's intent analysis response.

        Args:
            response: Raw LLM response

        Returns:
            Parsed intent analysis
        """
        try:
            # Try to extract JSON from the response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                parsed = json.loads(json_str)

                # Validate required fields
                if "type" not in parsed:
                    raise ValueError("Missing 'type' field")

                # goal_defined means "needs orchestration", but the response
                # handler picks the orchestrator from complexity/requires_tools
                # — without these, every LLM-classified goal defaulted to
                # "simple" and got a tool-less direct response.
                return self._normalize_parsed_intent(parsed)
            else:
                raise ValueError("No JSON found in response")

        except (json.JSONDecodeError, ValueError) as e:
            self.logger.warning(f"Failed to parse intent response: {e}")
            # Fallback parsing
            return self._fallback_intent_parsing(response)

    def _fallback_intent_parsing(self, response: str) -> Dict[str, Any]:
        """
        Fallback intent parsing when JSON parsing fails.

        Args:
            response: Raw LLM response

        Returns:
            Basic intent analysis
        """
        response_lower = response.lower()

        # Simple heuristics for intent detection
        if any(word in response_lower for word in ["unclear", "clarify", "question", "what do you mean"]):
            return self._normalize_parsed_intent({
                "type": "clarification_question",
                "confidence": 0.6,
                "reasoning": "Response suggests need for clarification",
                "clarification_question": "Could you please provide more details about what you'd like me to help you with?",
            })
        if any(word in response_lower for word in ["hello", "hi", "how are you", "what can you do"]):
            return self._normalize_parsed_intent({
                "type": "direct_response",
                "confidence": 0.8,
                "reasoning": "Greeting or general question",
                "direct_response": "Hello! I'm WITS, your AI assistant. I can help you with various tasks and questions.",
            })
        return self._normalize_parsed_intent({
            "type": "goal_defined",
            "confidence": 0.5,
            "reasoning": "Defaulting to task delegation",
            "goal_statement": response,
        })
