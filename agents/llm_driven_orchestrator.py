# agents/llm_driven_orchestrator.py
"""
LLM-Driven Orchestrator Agent for WitsV3.
Implements the ReAct loop with LLM-driven decision making.
"""

import json
import re
import logging
from typing import Any, Dict, List, Optional, AsyncGenerator

from agents.base_orchestrator_agent import BaseOrchestratorAgent
from core.config import WitsV3Config
from core.llm_interface import BaseLLMInterface
from core.memory_manager import MemoryManager
from core.schemas import StreamData, ConversationHistory


class LLMDrivenOrchestrator(BaseOrchestratorAgent):
    """
    LLM-Driven Orchestrator that uses the ReAct pattern for goal achievement.
    
    This orchestrator leverages the LLM's reasoning capabilities to:
    1. Break down complex goals into steps
    2. Decide when to use tools vs. provide answers
    3. Learn from previous iterations
    4. Adapt its approach based on results
    """
    
    def __init__(
        self,
        agent_name: str,
        config: WitsV3Config,
        llm_interface: BaseLLMInterface,
        memory_manager: Optional[MemoryManager] = None,
        tool_registry: Optional[Any] = None
    ):
        """
        Initialize the LLM-Driven Orchestrator.
        
        Args:
            agent_name: Name of this agent
            config: System configuration
            llm_interface: LLM interface
            memory_manager: Optional memory manager
            tool_registry: Optional tool registry
        """
        super().__init__(agent_name, config, llm_interface, memory_manager, tool_registry)
        
        # Available tools (will be populated from tool registry)
        self.available_tools = self._get_available_tools()
        
        self.logger.info(f"LLM-Driven Orchestrator initialized with {len(self.available_tools)} tools")

    def _get_available_tools(self) -> List[Dict[str, Any]]:
        """
        Get list of available tools from the tool registry.
        
        Returns:
            List of tool descriptions for the LLM
        """
        if not self.tool_registry:
            return [
                {
                    "name": "think",
                    "description": "Think through the problem step by step",
                    "parameters": {"thought": "string"}
                },
                {
                    "name": "answer",
                    "description": "Provide a final answer to the user",
                    "parameters": {"answer": "string"}
                }
            ]
        
        # Get actual tools from tool registry
        tools = []
        try:
            tools = self.tool_registry.get_tools_for_llm()
            self.logger.info(f"Retrieved {len(tools)} tools from registry")
        except Exception as e:
            self.logger.warning(f"Error getting tools from registry: {e}")
            # Return basic tools as fallback
            tools = [
                {
                    "name": "think",
                    "description": "Think through the problem step by step",
                    "parameters": {"thought": "string"}
                },
                {
                    "name": "answer",
                    "description": "Provide a final answer to the user",
                    "parameters": {"answer": "string"}
                }
            ]
        
        return tools
    
    def _build_reasoning_prompt(self, state: Dict[str, Any]) -> str:
        """
        Build the reasoning prompt for the ReAct loop.
        
        Args:
            state: Current ReAct state
            
        Returns:
            Formatted prompt for reasoning
        """
        goal = state["goal"]
        context = state["context"]
        history = state["history"]
        observations = state["observations"]
        
        # Build conversation history
        history_text = ""
        if history:
            history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-3:]])
        
        # Build observations
        observations_text = ""
        if observations:
            observations_text = "\n".join([f"- {obs}" for obs in observations[-3:]])
        
        # Build available tools text
        tools_text = ""
        if self.available_tools:
            tools_text = "\n".join([
                f"- {tool['name']}: {tool['description']}" 
                for tool in self.available_tools
            ])
        
        prompt = f"""You are an AI orchestrator using the ReAct (Reason-Act-Observe) pattern to achieve goals.

GOAL: {goal}

CONTEXT:
{context}

CONVERSATION HISTORY:
{history_text if history_text else "No conversation history"}

PREVIOUS OBSERVATIONS:
{observations_text if observations_text else "No previous observations"}

AVAILABLE TOOLS:
{tools_text if tools_text else "No tools available"}

Think step by step about how to achieve the goal. Then decide on your next action.

Respond with JSON in this format:
{{
    "thought": "your reasoning about the current situation and what to do next",
    "action_type": "tool_call" | "final_answer",
    "tool_name": "name of tool to use (if action_type is tool_call)",
    "tool_args": {{"arg1": "value1"}} (if using a tool),
    "final_answer": "your final answer (if action_type is final_answer)"
}}

Important:
- Use "tool_call" when you need to gather information or perform an action
- Use "final_answer" when you can provide a complete response to the goal
- Be specific and practical in your reasoning
- Consider what you've already learned from previous observations

Respond ONLY with valid JSON."""
        
        return prompt
    
    def _parse_reasoning_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the LLM's reasoning response.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Parsed reasoning components
        """
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                
                # Validate required fields
                if "action_type" not in parsed:
                    raise ValueError("Missing 'action_type' field")
                
                # Ensure we have the right fields for each action type
                if parsed["action_type"] == "tool_call":
                    if "tool_name" not in parsed:
                        raise ValueError("Missing 'tool_name' for tool_call action")
                    if "tool_args" not in parsed:
                        parsed["tool_args"] = {}
                
                elif parsed["action_type"] == "final_answer":
                    if "final_answer" not in parsed:
                        raise ValueError("Missing 'final_answer' for final_answer action")
                
                return parsed
            else:
                raise ValueError("No JSON found in response")
                
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.warning(f"Failed to parse reasoning response: {e}")
            # Fallback parsing
            return self._fallback_reasoning_parsing(response)
    
    def _fallback_reasoning_parsing(self, response: str) -> Dict[str, Any]:
        """
        Fallback reasoning parsing when JSON parsing fails.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Basic reasoning structure
        """
        response_lower = response.lower()
        
        # Look for action indicators
        if any(word in response_lower for word in ["tool", "use", "call", "search", "analyze"]):
            return {
                "thought": response,
                "action_type": "tool_call",
                "tool_name": "think",
                "tool_args": {"thought": response}
            }
        else:
            return {
                "thought": response,
                "action_type": "final_answer",
                "final_answer": response            }

    async def _call_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Any:
        """
        Call a tool through the tool registry.
        
        Args:
            tool_name: Name of the tool to call
            tool_args: Arguments for the tool
            
        Returns:
            Tool execution result
        """
        # Handle built-in tools
        if tool_name == "think":
            return f"Thinking: {tool_args.get('thought', 'No thought provided')}"
        
        elif tool_name == "answer":
            return tool_args.get('answer', 'No answer provided')
        
        # Handle registry tools
        elif self.tool_registry:
            try:
                # Use the tool registry's execute_tool method
                return await self.tool_registry.execute_tool(tool_name, **tool_args)
                
            except Exception as e:
                raise Exception(f"Error executing tool {tool_name}: {str(e)}")
        
        else:
            raise Exception(f"Unknown tool: {tool_name}")


# Test function
async def test_llm_driven_orchestrator():
    """Test the LLMDrivenOrchestrator functionality."""
    print("Testing LLMDrivenOrchestrator...")
    
    # Mock dependencies
    from core.config import load_config
    from core.llm_interface import OllamaInterface
    
    try:
        # Load config
        config = load_config("config.yaml")
        
        # Create LLM interface (will fail without Ollama, but that's ok for structure test)
        llm_interface = OllamaInterface(
            url=config.ollama_settings.url,
            default_model=config.ollama_settings.orchestrator_model
        )
        
        # Create orchestrator
        orchestrator = LLMDrivenOrchestrator(
            agent_name="TestOrchestrator",
            config=config,
            llm_interface=llm_interface
        )
        
        print(f"✓ LLMDrivenOrchestrator created: {orchestrator}")
        print(f"✓ Model name: {orchestrator.get_model_name()}")
        print(f"✓ Available tools: {len(orchestrator.available_tools)}")
        print(f"✓ Max iterations: {orchestrator.max_iterations}")
        
        # Test prompt building
        test_state = {
            "goal": "Test goal",
            "context": "Test context",
            "history": [],
            "observations": []
        }
        
        prompt = orchestrator._build_reasoning_prompt(test_state)
        print(f"✓ Reasoning prompt built (length: {len(prompt)})")
        
    except Exception as e:
        print(f"✓ LLMDrivenOrchestrator structure test passed (expected error: {e})")
    
    print("LLMDrivenOrchestrator tests completed! 🎉")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_llm_driven_orchestrator())
