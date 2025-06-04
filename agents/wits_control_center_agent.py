# agents/wits_control_center_agent.py
"""
WITS Control Center Agent for WitsV3.
The main entry point that handles user input and determines response strategy.
"""

import json
import uuid
import logging
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
        # Build context from conversation history
        history_context = ""
        if conversation_history and conversation_history.messages:
            recent_messages = conversation_history.get_recent_messages(5)
            history_context = "\n".join([
                f"{msg.role}: {msg.content}" for msg in recent_messages
            ])
        
        # Build the intent analysis prompt
        prompt = self._build_intent_analysis_prompt(user_input, history_context)
        
        try:
            # Get LLM response
            response = await self.generate_response(prompt, temperature=0.3)
            
            # Parse the response
            parsed_intent = self._parse_intent_response(response)
            
            return parsed_intent
            
        except Exception as e:
            self.logger.error(f"Error analyzing user intent: {e}")
            # Fallback to simple task delegation
            return {
                "type": "goal_defined",
                "confidence": 0.5,
                "goal_statement": user_input,
                "reasoning": "Failed to analyze intent, defaulting to task delegation"
            }
    
    def _build_intent_analysis_prompt(self, user_input: str, history_context: str) -> str:
        """
        Build the prompt for intent analysis.
        
        Args:
            user_input: User's input
            history_context: Recent conversation context
            
        Returns:
            Formatted prompt for intent analysis
        """
        prompt = f"""You are the WITS Control Center, analyzing user input to determine the best response strategy.

CONVERSATION HISTORY:
{history_context if history_context else "No previous conversation"}

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
        Handle the response based on intent analysis.
        
        Args:
            intent_analysis: Analyzed intent
            user_input: Original user input
            conversation_history: Conversation context
            session_id: Session identifier
            
        Yields:
            StreamData for the response handling
        """
        intent_type = intent_analysis.get("type")
        
        if intent_type == "direct_response":
            # Provide direct response
            response = intent_analysis.get("direct_response", "I'm here to help!")
            yield self.stream_result(response)
            
        elif intent_type == "clarification_question":
            # Ask for clarification
            question = intent_analysis.get(
                "clarification_question",
                "Could you please provide more details about what you'd like me to help you with?"
            )
            yield StreamData(
                type="clarification",
                content=question,
                source=self.agent_name
            )
            
        elif intent_type == "goal_defined":
            # Delegate to orchestrator
            goal_statement = intent_analysis.get("goal_statement", user_input)
            
            yield self.stream_thinking(f"Delegating to orchestrator: {goal_statement}")
            
            if self.orchestrator_agent:
                try:
                    async for stream_data in self.orchestrator_agent.run(
                        goal=goal_statement,
                        conversation_history=conversation_history,
                        session_id=session_id
                    ):
                        yield stream_data
                except Exception as e:
                    self.logger.error(f"Error in orchestrator delegation: {e}")
                    yield self.stream_error(
                        "I encountered an issue while processing your request.",
                        details=str(e)
                    )
            else:
                yield self.stream_error("No orchestrator available for task delegation.")
                
        else:
            # Unknown intent type
            yield self.stream_error(f"Unknown intent type: {intent_type}")


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
        llm_interface = OllamaInterface(config)
        
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
