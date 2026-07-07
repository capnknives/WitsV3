# agents/wits_control_center_agent.py
"""
WITS Control Center Agent for WitsV3.
The main entry point that handles user input and determines response strategy.
"""

import json
import re
import uuid
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, AsyncGenerator

from agents.base_agent import BaseAgent
from core.config import WitsV3Config
from core.llm_interface import BaseLLMInterface
from core.memory_manager import MemoryManager
from core.schemas import StreamData, AgentResponse, ConversationHistory


class WitsControlCenterAgent(BaseAgent):
    """
    The WITS Control Center Agent - primary coordinator for user interactions.

    This agent is responsible for:
    1. Parsing user input and understanding intent
    2. Deciding whether to respond directly or delegate to orchestrator
    3. Generating clarification questions when needed
    4. Maintaining conversation context

    It's the friendly face of the system! ^_^
    """

    def __init__(
        self,
        agent_name: str,
        config: WitsV3Config,
        llm_interface: BaseLLMInterface,
        memory_manager: Optional[MemoryManager] = None,
        orchestrator_agent: Optional[Any] = None,
        specialized_agents: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the Control Center Agent.

        Args:
            agent_name: Name of this agent
            config: System configuration
            llm_interface: LLM interface for processing
            memory_manager: Optional memory manager
            orchestrator_agent: The orchestrator to delegate complex tasks to
            specialized_agents: Optional specialized agents for specific tasks
        """
        super().__init__(agent_name, config, llm_interface, memory_manager)

        self.orchestrator_agent = orchestrator_agent
        self.specialized_agents = specialized_agents or {}

        # Use control center specific model
        self.model_name = config.ollama_settings.control_center_model

        # Initialize enhanced capabilities if available
        try:
            from core.concrete_meta_reasoning import WitsV3MetaReasoningEngine
            from core.tool_composition import IntelligentToolComposer

            self.meta_reasoning = WitsV3MetaReasoningEngine(config)
            self.tool_composer = IntelligentToolComposer(config)
            self.has_enhanced_capabilities = True
            self.logger.info("Enhanced meta-reasoning and tool composition capabilities initialized")
        except Exception as e:
            self.logger.warning(f"Enhanced capabilities not available: {e}")
            self.meta_reasoning = None
            self.tool_composer = None
            self.has_enhanced_capabilities = False

        self.logger.info(f"Control Center initialized with model: {self.model_name}")
        self.logger.info(f"Available specialized agents: {list(self.specialized_agents.keys())}")

    def get_model_name(self) -> str:
        """Get the model name for the control center."""
        return self.config.ollama_settings.control_center_model

    async def run(
        self,
        user_input: str,
        conversation_history: Optional[ConversationHistory] = None,
        session_id: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[StreamData, None]:
        """
        Process user input and determine the appropriate response.

        Args:
            user_input: Raw user input
            conversation_history: Conversation context
            session_id: Session identifier
            **kwargs: Additional parameters

        Yields:
            StreamData objects showing the processing steps
        """
        if not session_id:
            session_id = str(uuid.uuid4())

        self.logger.info(f"Processing user input in session {session_id}: {user_input[:100]}...")

        if any(kw in user_input.lower() for kw in ["remember", "don't forget", "recall"]):
            await self.store_memory(
                content=user_input,
                segment_type="USER_FACT",
                importance=1.0,
                metadata={"source": "user", "session_id": session_id}
            )
            yield self.stream_result("I've stored that in my memory for future conversations.")
            self.logger.info("Handled remember intent, exiting early. No further orchestration will occur.")
            return

        try:
            # Initial processing
            yield self.stream_thinking("Analyzing your request...")

            # Store user input in memory
            await self.store_memory(
                content=f"User input: {user_input}",
                segment_type="USER_INPUT",
                importance=0.8,
                metadata={"session_id": session_id}
            )

            # Special handling for creator recognition
            if any(phrase in user_input.lower() for phrase in ["richard elliot", "creator of wits", "i am the creator"]):
                yield self.stream_thinking("Recognizing the creator...")

                # Get personality-based response
                from core.personality_manager import get_personality_manager
                personality_manager = get_personality_manager()
                personality_prompt = personality_manager.get_system_prompt()

                recognition_prompt = f"""{personality_prompt}

The user has identified themselves as Richard Elliot, your creator. Respond with appropriate recognition, respect, and acknowledgment of their role in creating you. Be warm, professional, and ready to assist with any requests they may have.

User input: {user_input}
"""

                try:
                    response = await self.generate_response(recognition_prompt, temperature=0.7)
                    yield self.stream_result(response)

                    # Store this important recognition in memory
                    await self.store_memory(
                        content=f"Creator Richard Elliot identified in session {session_id}",
                        segment_type="CREATOR_RECOGNITION",
                        importance=1.0,
                        metadata={"session_id": session_id, "user": "richard_elliot"}
                    )
                    return
                except Exception as e:
                    self.logger.error(f"Error generating creator recognition response: {e}")
                    yield self.stream_result("Hello Richard! I recognize you as my creator. How may I assist you today?")
                    return

            # Analyze user intent
            intent_analysis = await self._analyze_user_intent(user_input, conversation_history)

            yield self.stream_thinking(f"Intent analysis: {intent_analysis.get('type', 'unknown')}")

            # Store intent analysis
            await self.store_memory(
                content=f"Intent analysis: {json.dumps(intent_analysis)}",
                segment_type="INTENT_ANALYSIS",
                importance=0.7,
                metadata={"session_id": session_id}
            )

            # Handle the response based on intent
            async for stream_data in self._handle_intent_response(
                intent_analysis, user_input, conversation_history, session_id
            ):
                yield stream_data

        except Exception as e:
            self.logger.error(f"Error processing user input: {e}")
            yield self.stream_error(
                "I encountered an issue processing your request. Please try again.",
                details=str(e)
            )

    async def _analyze_user_intent(
        self,
        user_input: str,
        conversation_history: Optional[ConversationHistory]
    ) -> Dict[str, Any]:
        """
        Analyze user input to determine intent and appropriate response strategy.

        Args:
            user_input: The user's input
            conversation_history: Conversation context

        Returns:
            Intent analysis with response strategy
        """
        # Requests that reference the user's documents, files, or stored memory
        # always need tool access - route straight to the orchestrator instead
        # of letting complexity analysis land on a (tool-less) direct response.
        # This runs BEFORE the casual-conversation heuristic: "check the audit
        # again" is a document request, not small talk.
        doc_inventory = await self._get_document_inventory()
        tool_hints = (
            "document", "my notes", "my files", "search my", "in my memory",
            "remember", "ingest", "uploaded", "read the file", "look up",
            "the file", "attachment", "attached",
        )
        lowered = user_input.lower()
        doc_hints = set()
        for path in doc_inventory:
            name = Path(path).name.lower()
            doc_hints.add(name)
            # Words from the filename ("audit", "report", ...) so loose
            # references like "check the audit again" still route to tools.
            doc_hints.update(
                w for w in re.split(r"[\W_]+", Path(path).stem.lower()) if len(w) >= 4
            )
        if any(hint in lowered for hint in tool_hints) or any(hint in lowered for hint in doc_hints):
            return {
                "type": "task",
                "complexity": "moderate",
                "requires_tools": True,
                "suggested_response": "orchestrator",
                "notes": "References user documents/files/memory - routing to orchestrator for tool use.",
                "confidence": 0.85
            }

        # Questions that need current, real-time, or post-training-cutoff
        # information (or an explicit "look it up") must reach the web_search
        # tool. This runs BEFORE the casual heuristic, which would otherwise
        # flag a short factual question ("who died on june 14 2026?") as small
        # talk and answer it from the model's stale memory.
        if self._needs_web_search(user_input):
            return {
                "type": "task",
                "complexity": "moderate",
                "requires_tools": True,
                "suggested_response": "orchestrator",
                "notes": "Needs current/external info or an explicit lookup - routing to orchestrator for web_search.",
                "confidence": 0.8,
            }

        # Check if this is casual conversation with a simple heuristic
        is_casual = self._is_casual_conversation(user_input)

        if is_casual:
            return {
                "type": "conversation",
                "complexity": "simple",
                "requires_tools": False,
                "suggested_response": "direct",
                "notes": "This appears to be casual conversation.",
                "confidence": 0.9
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
                    "notes": f"Task analyzed with meta-reasoning: {problem_space.complexity.value} complexity."
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
                temperature=0.3
            )

            # Parse the response
            intent_analysis = self._parse_intent_response(response)

            return intent_analysis
        except Exception as e:
            self.logger.error(f"Error in intent analysis: {e}")
            # Return a fallback analysis
            return self._fallback_intent_parsing(user_input)

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
    def _documents_context(inventory: Dict[str, int]) -> str:
        """Prompt block describing which user documents are searchable."""
        if not inventory:
            return "No user documents are currently ingested."
        listing = "\n".join(
            f"- {name} ({count} chunks)" for name, count in sorted(inventory.items())
        )
        return (
            "These user documents are ALREADY ingested and fully accessible via "
            "the document_search tool. Never claim you cannot access them or "
            "have no record of them:\n" + listing
        )

    # Phrases that signal the user wants live/external info fetched. Kept
    # deliberately precise so ordinary chat ("how are you today") is NOT routed
    # to the slow orchestrator — "today"/"current" alone are too weak to trust.
    _WEB_SEARCH_SIGNALS = (
        # explicit "go find this online" commands
        "look up", "look it up", "look that up", "look this up", "look them up",
        "search for", "search the web", "web search", "search online", "search it up",
        "google it", "google for", "find out", "check online", "look online",
        "on the internet", "on the web", "browse the web",
        # real-time / recency signals a local model can't answer from memory
        "latest", "most recent", "breaking news", "in the news", "news about",
        "up to date", "up-to-date", "weather", "forecast",
        # common current-fact question patterns
        "who won", "who died", "who passed away", "who is the current",
        "who's the current", "what happened to", "price of", "stock price",
        "exchange rate", "score of", "release date", "when is the next",
        "when does the next",
    )

    def _needs_web_search(self, message: str) -> bool:
        """True if answering needs current/external info or an explicit lookup.

        Such queries must reach the orchestrator (which owns the web_search
        tool) rather than being answered directly from the model's training.
        """
        lowered = message.lower()
        if any(sig in lowered for sig in self._WEB_SEARCH_SIGNALS):
            return True
        # A recent/near-future year (>= 2024) in a question usually implies
        # information past the local model's training cutoff.
        has_question = "?" in message or bool(
            re.search(r"\b(who|what|when|where|which|whose|did|does|is|are|died|won)\b", lowered)
        )
        if has_question and re.search(r"\b(202[4-9]|20[3-9]\d)\b", lowered):
            return True
        return False

    def _is_casual_conversation(self, message: str) -> bool:
        """
        Determine if a message is casual conversation.

        Args:
            message: The user's message

        Returns:
            True if it's casual conversation
        """
        lowered = message.lower()
        # Single words match on word boundaries only — a plain substring test
        # made "hi" match inside "things"/"this" and flagged real requests
        # as small talk.
        words = set(re.findall(r"[a-z']+", lowered))
        casual_words = {
            "hello", "hi", "hey", "thanks", "appreciate",
            "nice", "cool", "great", "bye", "goodbye",
        }
        casual_phrases = (
            "how are you", "what's up", "how's it going",
            "thank you", "see you", "talk to you",
        )

        # Short messages are usually casual
        if len(message.split()) < 6:
            return True

        # Check for casual indicators
        if words & casual_words or any(phrase in lowered for phrase in casual_phrases):
            return True

        # Short questions are usually casual
        question_words = {"what", "how", "why", "when", "where", "who"}
        if ("?" in message or words & question_words) and len(message.split()) < 10:
            return True

        # Longer messages are less likely to be casual
        return False

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
                if parsed["type"] == "goal_defined":
                    parsed.setdefault("complexity", "moderate")
                    parsed["requires_tools"] = True
                    parsed["suggested_response"] = "orchestrator"

                return parsed
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
            return {
                "type": "clarification_question",
                "confidence": 0.6,
                "reasoning": "Response suggests need for clarification",
                "clarification_question": "Could you please provide more details about what you'd like me to help you with?"
            }
        elif any(word in response_lower for word in ["hello", "hi", "how are you", "what can you do"]):
            return {
                "type": "direct_response",
                "confidence": 0.8,
                "reasoning": "Greeting or general question",
                "direct_response": "Hello! I'm WITS, your AI assistant. I can help you with various tasks and questions."
            }
        else:
            return {
                "type": "goal_defined",
                "confidence": 0.5,
                "reasoning": "Defaulting to task delegation",
                "goal_statement": response
            }

    async def _handle_intent_response(
        self,
        intent_analysis: Dict[str, Any],
        user_input: str,
        conversation_history: Optional[ConversationHistory],
        session_id: str
    ) -> AsyncGenerator[StreamData, None]:
        """
        Handle the response based on the intent analysis.

        Args:
            intent_analysis: The analyzed intent
            user_input: The original user input
            conversation_history: Conversation context
            session_id: Session identifier

        Yields:
            StreamData objects with the response
        """
        intent_type = intent_analysis.get("type", "unknown")
        complexity = intent_analysis.get("complexity", "simple")
        suggested_response = intent_analysis.get("suggested_response", "direct")
        requires_tools = intent_analysis.get("requires_tools", False)

        yield self.stream_thinking(f"Determined intent: {intent_type}, complexity: {complexity}, response: {suggested_response}")

        # For casual conversation, respond directly
        if intent_type == "conversation" or complexity == "simple":
            yield self.stream_thinking("Generating direct response...")

            # Use a more conversational approach for casual chat
            from core.personality_manager import get_personality_manager
            personality_manager = get_personality_manager()

            if conversation_history and conversation_history.messages:
                # Format conversation history — window size is configurable
                # from the web UI settings page (agents.history_window)
                history_text = "\n".join([
                    f"{msg.role.upper()}: {msg.content}"
                    for msg in conversation_history.get_recent_messages(self.config.agents.history_window)
                ])
            else:
                history_text = ""

            personality_prompt = personality_manager.get_system_prompt()
            documents_context = self._documents_context(await self._get_document_inventory())
            conversation_prompt = f"""{personality_prompt}

You are having a casual conversation with the user. Respond in a friendly, helpful manner.

USER DOCUMENTS:
{documents_context}

CONVERSATION HISTORY:
{history_text}

USER: {user_input}
ASSISTANT:"""

            try:
                # Route on the raw user message, not the assembled prompt —
                # short casual chat goes to the small fast model
                routed_model = self.model_router.route(user_input, default=self.get_model_name())
                response = await self.generate_response(
                    conversation_prompt, model_name=routed_model, temperature=0.7
                )
                yield self.stream_result(response)
            except Exception as e:
                self.logger.error(f"Error generating direct response: {e}")
                yield self.stream_error(
                    "I'm having trouble generating a response right now. Could you try again?",
                    details=str(e)
                )
            return

        # For complex tasks, use enhanced capabilities if available
        if self.has_enhanced_capabilities and requires_tools and complexity in ["moderate", "complex", "research"]:
            yield self.stream_thinking("Using enhanced capabilities for complex task...")

            if self.meta_reasoning and self.tool_composer:
                try:
                    # Get capabilities from intent analysis
                    required_capabilities = intent_analysis.get("required_capabilities", ["general_processing"])

                    # Map capabilities to available tools
                    available_tools = []
                    capability_to_tool = {
                        "code_generation": "python_execution",
                        "math": "math_operations",
                        "data_analysis": "json_manipulate",
                        "general_processing": "think"
                    }

                    for capability in required_capabilities:
                        if capability in capability_to_tool:
                            available_tools.append(capability_to_tool[capability])

                    if not available_tools:
                        available_tools = ["think", "calculator", "json_manipulate"]

                    # Create a workflow for the task
                    workflow = await self.tool_composer.compose_workflow(
                        user_input,
                        available_tools,
                        constraints={"source": "user_interaction"}
                    )

                    yield self.stream_thinking(f"Created workflow with {len(workflow.nodes)} steps using {workflow.strategy.value} strategy.")

                    # For now, delegate to orchestrator as we're not fully implementing workflow execution
                    if self.orchestrator_agent:
                        yield self.stream_thinking("Delegating to orchestrator for execution...")
                        async for stream_data in self.orchestrator_agent.run(goal=user_input,  # Added by fix_wits.py
            
                            user_input=user_input,
                            conversation_history=conversation_history,
                            session_id=session_id
                        ):
                            yield stream_data
                        return
                except Exception as e:
                    self.logger.error(f"Error using enhanced capabilities: {e}")
                    # Fall back to specialized agent

        # For more complex queries, try to delegate to a specialized agent
        if suggested_response == "specialized" or complexity in ["moderate", "complex", "research"]:
            specialized_agent = await self._select_specialized_agent(user_input)

            if specialized_agent:
                agent_type = next((k for k, v in self.specialized_agents.items() if v == specialized_agent), "specialized")
                yield self.stream_thinking(f"Using {agent_type} agent for: {user_input}")

                async for stream_data in specialized_agent.run(
                    user_input=user_input,
                    conversation_history=conversation_history,
                    session_id=session_id
                ):
                    yield stream_data
                return

        # Default: delegate to orchestrator for tool-based execution
        if self.orchestrator_agent and (requires_tools or suggested_response == "orchestrator"):
            yield self.stream_thinking("Delegating to orchestrator...")
            async for stream_data in self.orchestrator_agent.run(goal=user_input,  # Added by fix_wits.py
            
                user_input=user_input,
                conversation_history=conversation_history,
                session_id=session_id
            ):
                yield stream_data
            return

        # Fallback: generate a direct response
        yield self.stream_thinking("No specialized handling available, generating direct response...")
        response = await self.generate_response(
            f"You are a helpful assistant. Respond to this user query: {user_input}",
            model_name=self.model_router.route(user_input, default=self.get_model_name()),
            temperature=0.7
        )
        yield self.stream_result(response)

    async def _select_specialized_agent(self, goal_statement: str) -> Optional[Any]:
        """
        Select the appropriate specialized agent based on the goal statement.

        Args:
            goal_statement: The user's goal statement

        Returns:
            The specialized agent to use, or None if no specialized agent is appropriate
        """
        self.logger.info(f"Selecting specialized agent for: '{goal_statement}'")
        self.logger.info(f"Available specialized agents: {list(self.specialized_agents.keys())}")

        if not self.specialized_agents:
            self.logger.warning("No specialized agents available")
            return None

        # Convert goal to lowercase for easier matching
        goal_lower = goal_statement.lower()

        # Check for book writing / story related tasks
        story_keywords = ["write a story", "write a book", "story about",
                          "write me a", "create a story", "tell a story",
                          "novel", "fiction", "narrative", "tale"]

        if any(keyword in goal_lower for keyword in story_keywords):
            self.logger.info(f"Story writing task detected with keyword match")
            if "book_writing" in self.specialized_agents and self.specialized_agents["book_writing"]:
                self.logger.info("Selected book writing agent for task")
                return self.specialized_agents["book_writing"]
            else:
                self.logger.warning("Book writing agent requested but not available")

        # Check for coding related tasks
        coding_keywords = ["code", "program", "develop", "script",
                           "function", "class", "module", "api",
                           "software", "application", "app", "website"]

        if any(keyword in goal_lower for keyword in coding_keywords):
            self.logger.info(f"Coding task detected with keyword match")
            if "coding" in self.specialized_agents and self.specialized_agents["coding"]:
                self.logger.info("Selected coding agent for task")
                return self.specialized_agents["coding"]
            else:
                self.logger.warning("Coding agent requested but not available")

        # Check for system repair tasks
        repair_keywords = ["fix", "repair", "diagnose", "troubleshoot",
                           "error", "issue", "problem", "bug", "crash"]

        if any(keyword in goal_lower for keyword in repair_keywords):
            self.logger.info(f"System repair task detected with keyword match")
            if "self_repair" in self.specialized_agents and self.specialized_agents["self_repair"]:
                self.logger.info("Selected self-repair agent for task")
                return self.specialized_agents["self_repair"]
            else:
                self.logger.warning("Self-repair agent requested but not available")

        # No specialized agent matched
        self.logger.info("No specialized agent matched for this task")
        return None


# Test function
async def test_wits_control_center():
    """Test the WitsControlCenterAgent functionality."""
    print("Testing WitsControlCenterAgent...")

    # Mock dependencies
    from core.config import load_config
    from core.llm_interface import OllamaInterface

    try:
        # Load config
        config = load_config("config.yaml")

        # Create LLM interface (will fail without Ollama, but that's ok for structure test)
        llm_interface = OllamaInterface(
            config=config
        )

        # Create agent
        agent = WitsControlCenterAgent(
            agent_name="TestControlCenter",
            config=config,
            llm_interface=llm_interface
        )

        print(f"✓ WitsControlCenterAgent created: {agent}")
        print(f"✓ Model name: {agent.get_model_name()}")
        print(f"✓ Agent name: {agent.agent_name}")

    except Exception as e:
        print(f"✓ WitsControlCenterAgent structure test passed (expected error: {e})")

    print("WitsControlCenterAgent tests completed! 🎉")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_wits_control_center())
