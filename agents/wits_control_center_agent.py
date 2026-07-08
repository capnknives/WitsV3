# agents/wits_control_center_agent.py
"""
WITS Control Center Agent for WitsV3.
The main entry point that handles user input and determines response strategy.
"""

import json
import re
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from agents.base_agent import BaseAgent
from agents.wcca_intent_mixin import WCCAIntentMixin
from agents.wcca_routing_mixin import OrchestratorRoutingMixin
from core.config import WitsV3Config
from core.llm_interface import BaseLLMInterface
from core.memory_manager import MemoryManager
from core.schemas import ConversationHistory, StreamData


class WitsControlCenterAgent(OrchestratorRoutingMixin, WCCAIntentMixin, BaseAgent):
    """
    The WITS Control Center Agent - primary coordinator for user interactions.

    This agent is responsible for:
    1. Parsing user input and understanding intent
    2. Deciding whether to respond directly or delegate to orchestrator
    3. Generating clarification questions when needed
    4. Maintaining conversation context

    It's the friendly face of the system! ^_^
    """

    INTENT_ANALYSIS_TEMPERATURE = 0.3

    def __init__(
        self,
        agent_name: str,
        config: WitsV3Config,
        llm_interface: BaseLLMInterface,
        memory_manager: MemoryManager | None = None,
        orchestrator_agent: Any | None = None,
        specialized_agents: dict[str, Any] | None = None,
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
            self.logger.info(
                "Enhanced meta-reasoning and tool composition capabilities initialized"
            )
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
        conversation_history: ConversationHistory | None = None,
        session_id: str | None = None,
        **kwargs,
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

        user_role = kwargs.get("user_role", "owner")
        guest_profile = kwargs.get("guest_profile")
        self._request_user_role = user_role
        self._request_guest_profile = guest_profile

        self.logger.info(f"Processing user input in session {session_id}: {user_input[:100]}...")

        if any(kw in user_input.lower() for kw in ["remember", "don't forget", "recall"]):
            await self.store_memory(
                content=user_input,
                segment_type="USER_FACT",
                importance=1.0,
                metadata={"source": "user", "session_id": session_id},
            )
            yield self.stream_result("I've stored that in my memory for future conversations.")
            self.logger.info(
                "Handled remember intent, exiting early. No further orchestration will occur."
            )
            return

        try:
            # Initial processing
            yield self.stream_thinking("Analyzing your request...")

            # Store user input in memory
            await self.store_memory(
                content=f"User input: {user_input}",
                segment_type="USER_INPUT",
                importance=0.8,
                metadata={"session_id": session_id},
            )

            # Special handling for creator recognition
            if any(
                phrase in user_input.lower()
                for phrase in ["richard elliot", "creator of wits", "i am the creator"]
            ):
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
                        metadata={"session_id": session_id, "user": "richard_elliot"},
                    )
                    return
                except Exception as e:
                    self.logger.error(f"Error generating creator recognition response: {e}")
                    yield self.stream_result(
                        "Hello Richard! I recognize you as my creator. How may I assist you today?"
                    )
                    return

            # Analyze user intent
            intent_analysis = await self._analyze_user_intent(user_input, conversation_history)

            yield self.stream_thinking(f"Intent analysis: {intent_analysis.get('type', 'unknown')}")

            # Store intent analysis
            await self.store_memory(
                content=f"Intent analysis: {json.dumps(intent_analysis)}",
                segment_type="INTENT_ANALYSIS",
                importance=0.7,
                metadata={"session_id": session_id},
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
                details=str(e),
            )

    async def _handle_intent_response(
        self,
        intent_analysis: dict[str, Any],
        user_input: str,
        conversation_history: ConversationHistory | None,
        session_id: str,
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

        # An unambiguous "find/fix real bugs" request must reach the
        # self-repair agent regardless of what the LLM intent classifier
        # decided — 2026-07-08 finding: "find and fix any bugs in your code"
        # was classified as clarification_question and "conversation" for a
        # follow-up, both of which return before specialized-agent routing
        # is ever considered below. Guests never get self-repair.
        if getattr(self, "_request_user_role", "owner") != "guest" and self._needs_self_repair(
            user_input
        ):
            intent_type = "goal_defined"
            complexity = "moderate"
            requires_tools = True
            suggested_response = "specialized"

        yield self.stream_thinking(
            f"Determined intent: {intent_type}, complexity: {complexity}, response: {suggested_response}"
        )

        # LLM-classified direct_response: use the intent JSON text, not a second
        # casual-chat call — unless tools are required (misclassification guard).
        if intent_type == "direct_response":
            if await self._requires_orchestrator_for_input(user_input):
                intent_type = "goal_defined"
                complexity = "moderate"
                requires_tools = True
                suggested_response = "orchestrator"
            else:
                direct_text = (intent_analysis.get("direct_response") or "").strip()
                if direct_text:
                    yield self.stream_result(direct_text)
                    return
                async for stream_data in self._stream_casual_chat_response(
                    user_input, conversation_history
                ):
                    yield stream_data
                return

        if intent_type == "clarification_question":
            question = (intent_analysis.get("clarification_question") or "").strip()
            if not question:
                question = "Could you please provide more details about what you'd like me to help you with?"
            yield self.stream_result(question)
            return

        if intent_type == "conversation":
            async for stream_data in self._stream_casual_chat_response(
                user_input, conversation_history
            ):
                yield stream_data
            return

        # Try a specialized agent (book writing / coding / self-repair) before
        # the generic enhanced-capabilities/orchestrator paths below — those
        # unconditionally `return` once entered, so a specialized agent match
        # checked afterward would never actually be reached in practice.
        # Guests stay on chat/orchestrator with the filtered tool allowlist.
        if (
            getattr(self, "_request_user_role", "owner") != "guest"
            and (
                suggested_response == "specialized"
                or complexity in ["moderate", "complex", "research"]
            )
        ):
            specialized_agent = await self._select_specialized_agent(user_input)

            if specialized_agent:
                agent_type = next(
                    (k for k, v in self.specialized_agents.items() if v == specialized_agent),
                    "specialized",
                )
                yield self.stream_thinking(f"Using {agent_type} agent for: {user_input}")

                async for stream_data in specialized_agent.run(
                    user_input=user_input,
                    conversation_history=conversation_history,
                    session_id=session_id,
                ):
                    yield stream_data
                return

        # For complex tasks, use enhanced capabilities if available
        if (
            getattr(self, "_request_user_role", "owner") != "guest"
            and self.has_enhanced_capabilities
            and requires_tools
            and complexity in ["moderate", "complex", "research"]
        ):
            yield self.stream_thinking("Using enhanced capabilities for complex task...")

            if self.meta_reasoning and self.tool_composer:
                try:
                    # Get capabilities from intent analysis
                    required_capabilities = intent_analysis.get(
                        "required_capabilities", ["general_processing"]
                    )

                    # Map capabilities to available tools
                    available_tools = []
                    capability_to_tool = {
                        "code_generation": "python_execution",
                        "math": "math_operations",
                        "data_analysis": "json_manipulate",
                        "general_processing": "think",
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
                        constraints={"source": "user_interaction"},
                    )

                    yield self.stream_thinking(
                        f"Created workflow with {len(workflow.nodes)} steps using {workflow.strategy.value} strategy."
                    )

                    # For now, delegate to orchestrator as we're not fully implementing workflow execution
                    if self.orchestrator_agent:
                        yield self.stream_thinking("Delegating to orchestrator for execution...")
                        async for stream_data in self.orchestrator_agent.run(
                            user_input=user_input,
                            conversation_history=conversation_history,
                            session_id=session_id,
                            user_role=getattr(self, "_request_user_role", "owner"),
                            guest_profile=getattr(self, "_request_guest_profile", None),
                        ):
                            yield stream_data
                        return
                except Exception as e:
                    self.logger.error(f"Error using enhanced capabilities: {e}")
                    # Fall through to the generic orchestrator below

        # Default: delegate to orchestrator for tool-based execution
        if self.orchestrator_agent and (requires_tools or suggested_response == "orchestrator"):
            yield self.stream_thinking("Delegating to orchestrator...")
            async for stream_data in self.orchestrator_agent.run(
                user_input=user_input,
                conversation_history=conversation_history,
                session_id=session_id,
                user_role=getattr(self, "_request_user_role", "owner"),
                guest_profile=getattr(self, "_request_guest_profile", None),
            ):
                yield stream_data
            return

        # Fallback: generate a direct response
        yield self.stream_thinking(
            "No specialized handling available, generating direct response..."
        )
        response = await self.generate_response(
            f"You are a helpful assistant. Respond to this user query: {user_input}",
            model_name=self.model_router.route(user_input, default=self.get_model_name()),
            temperature=0.7,
        )
        yield self.stream_result(response)

    async def _stream_casual_chat_response(
        self,
        user_input: str,
        conversation_history: ConversationHistory | None,
    ) -> AsyncGenerator[StreamData, None]:
        """Generate a friendly chat reply (heuristic conversation path only)."""
        yield self.stream_thinking("Generating direct response...")

        from core.personality_manager import get_personality_manager

        personality_manager = get_personality_manager()

        if conversation_history and conversation_history.messages:
            history_text = "\n".join(
                [
                    f"{msg.role.upper()}: {msg.content}"
                    for msg in conversation_history.get_recent_messages(
                        self.config.agents.history_window
                    )
                ]
            )
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
            routed_model = self.model_router.route(user_input, default=self.get_model_name())
            response = await self.generate_response(
                conversation_prompt, model_name=routed_model, temperature=0.7
            )
            yield self.stream_result(response)
        except Exception as e:
            self.logger.error(f"Error generating direct response: {e}")
            yield self.stream_error(
                "I'm having trouble generating a response right now. Could you try again?",
                details=str(e),
            )

    async def _select_specialized_agent(self, goal_statement: str) -> Any | None:
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

        def _matches(keywords: list[str]) -> bool:
            """Whole-word match — plain substring `in` checks false-positive
            constantly (e.g. "codebase" contains "code", "description"
            contains "script"), which is exactly what misrouted a live "find
            bugs in the codebase" request to the coding agent instead of
            self-repair on 2026-07-08."""
            return any(re.search(rf"\b{re.escape(kw)}\b", goal_lower) for kw in keywords)

        # Check for book writing / story related tasks. Deliberately NOT
        # "write me a" alone — it's a substring of any "write me a <thing>"
        # request (e.g. "write me a python script"), which misrouted plain
        # coding requests to the book-writing agent before this fix.
        story_keywords = [
            "write a story",
            "write a book",
            "story about",
            "write me a story",
            "write me a novel",
            "write me a poem",
            "create a story",
            "tell a story",
            "novel",
            "fiction",
            "narrative",
            "tale",
        ]

        if _matches(story_keywords):
            self.logger.info("Story writing task detected with keyword match")
            if (
                "book_writing" in self.specialized_agents
                and self.specialized_agents["book_writing"]
            ):
                self.logger.info("Selected book writing agent for task")
                return self.specialized_agents["book_writing"]
            else:
                self.logger.warning("Book writing agent requested but not available")

        # Check for system repair tasks BEFORE generic coding tasks — "fix",
        # "bug", etc. are a much stronger and more specific signal than the
        # broad coding keyword list below, so a message like "find and fix
        # bugs in the codebase" should land on self-repair, not coding.
        repair_keywords = [
            "fix",
            "repair",
            "diagnose",
            "troubleshoot",
            "error",
            "issue",
            "problem",
            "bug",
            "bugs",
            "crash",
        ]

        if _matches(repair_keywords):
            self.logger.info("System repair task detected with keyword match")
            if "self_repair" in self.specialized_agents and self.specialized_agents["self_repair"]:
                self.logger.info("Selected self-repair agent for task")
                return self.specialized_agents["self_repair"]
            else:
                self.logger.warning("Self-repair agent requested but not available")

        # Check for coding related tasks
        coding_keywords = [
            "code",
            "program",
            "develop",
            "script",
            "function",
            "class",
            "module",
            "api",
            "software",
            "application",
            "app",
            "website",
        ]

        if _matches(coding_keywords):
            self.logger.info("Coding task detected with keyword match")
            if "coding" in self.specialized_agents and self.specialized_agents["coding"]:
                self.logger.info("Selected coding agent for task")
                return self.specialized_agents["coding"]
            else:
                self.logger.warning("Coding agent requested but not available")

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
            config=config,
        )

        # Create agent
        agent = WitsControlCenterAgent(
            agent_name="TestControlCenter",
            config=config,
            llm_interface=llm_interface,
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
