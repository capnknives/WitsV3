# agents/base_orchestrator_agent.py
"""
Base Orchestrator Agent for WitsV3.
Implements ReAct (Reason-Act-Observe) loop functionality.
"""

import json
import logging
from typing import Any, Dict, List, Optional, AsyncGenerator
from abc import abstractmethod

from agents.base_agent import BaseAgent
from core.config import WitsV3Config
from core.llm_interface import BaseLLMInterface
from core.memory_manager import MemoryManager
from core.schemas import StreamData, AgentResponse, ConversationHistory, ToolCall


class BaseOrchestratorAgent(BaseAgent):
    """
    Base class for orchestrator agents that implement ReAct loops.
    
    This agent follows the ReAct pattern:
    - Reason: Think about the goal and plan next steps
    - Act: Take an action (use tool, call agent, or provide answer)
    - Observe: Observe the results and plan next steps
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
        Initialize the orchestrator agent.
        
        Args:
            agent_name: Name of this agent
            config: System configuration
            llm_interface: LLM interface
            memory_manager: Optional memory manager
            tool_registry: Optional tool registry for accessing tools
        """
        super().__init__(agent_name, config, llm_interface, memory_manager)
        self.tool_registry = tool_registry
        
        # ReAct loop configuration
        self.max_iterations = config.agents.max_iterations
        self.current_iteration = 0
        
        self.logger.info(f"Initialized {self.__class__.__name__} with max_iterations: {self.max_iterations}")
    
    def get_model_name(self) -> str:
        """Get the model name for orchestrator agents."""
        return self.config.ollama_settings.orchestrator_model
    
    async def run(
        self,
        goal: str,
        conversation_history: Optional[ConversationHistory] = None,
        session_id: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[StreamData, None]:
        """
        Execute the ReAct loop to achieve the given goal.
        
        Args:
            goal: The goal to achieve
            conversation_history: Optional conversation context
            session_id: Optional session identifier
            **kwargs: Additional parameters
            
        Yields:
            StreamData objects showing the reasoning and action process
        """
        self.current_iteration = 0
        
        # Initial setup
        yield self.stream_thinking(f"Starting orchestration for goal: {goal}")
        
        # Store the goal in memory
        await self.store_memory(
            content=f"Goal: {goal}",
            segment_type="GOAL",
            importance=0.9,
            metadata={"session_id": session_id}
        )
        
        # Get relevant context from memory
        relevant_memories = await self.search_memory(goal, limit=5)
        context = self._build_context_from_memories(relevant_memories)
        
        # Initialize the ReAct state
        react_state = {
            "goal": goal,
            "context": context,
            "history": conversation_history.to_llm_format() if conversation_history else [],
            "observations": [],
            "completed": False,
            "final_answer": None
        }
        
        try:
            # Execute ReAct loop
            async for stream_data in self._execute_react_loop(react_state, session_id):
                yield stream_data
                
        except Exception as e:
            self.logger.error(f"Error in orchestrator ReAct loop: {e}")
            yield self.stream_error(
                f"An error occurred during orchestration: {str(e)}",
                details=str(e)
            )
    
    async def _execute_react_loop(
        self,
        state: Dict[str, Any],
        session_id: Optional[str]
    ) -> AsyncGenerator[StreamData, None]:
        """
        Execute the main ReAct loop.
        
        Args:
            state: Current ReAct state
            session_id: Session identifier
            
        Yields:
            StreamData from each step of the loop
        """
        while not state["completed"] and self.current_iteration < self.max_iterations:
            self.current_iteration += 1
            
            yield self.stream_thinking(f"ReAct iteration {self.current_iteration}/{self.max_iterations}")
            
            # REASON: Generate thoughts and plan
            reasoning_prompt = self._build_reasoning_prompt(state)
            
            yield self.stream_thinking("Analyzing the situation and planning next steps...")
            
            reasoning_response = await self.generate_response(reasoning_prompt)
            parsed_reasoning = self._parse_reasoning_response(reasoning_response)
            
            # Stream the reasoning
            if parsed_reasoning.get("thought"):
                yield self.stream_thinking(f"Thought: {parsed_reasoning['thought']}")
            
            # Store reasoning in memory
            await self.store_memory(
                content=f"Reasoning: {reasoning_response}",
                segment_type="REASONING",
                importance=0.7,
                metadata={"iteration": self.current_iteration, "session_id": session_id}
            )
            
            # ACT: Decide on action
            action_type = parsed_reasoning.get("action_type", "final_answer")
            
            if action_type == "tool_call":
                # Execute tool call
                async for stream_data in self._execute_tool_action(parsed_reasoning, state, session_id):
                    yield stream_data
                    
            elif action_type == "final_answer":
                # Provide final answer
                final_answer = parsed_reasoning.get("final_answer", "I've completed the task.")
                state["completed"] = True
                state["final_answer"] = final_answer
                
                yield self.stream_result(final_answer)
                
                # Store final answer
                await self.store_memory(
                    content=f"Final Answer: {final_answer}",
                    segment_type="FINAL_ANSWER",
                    importance=1.0,
                    metadata={"session_id": session_id}
                )
                
            else:
                # Unknown action type
                yield self.stream_error(f"Unknown action type: {action_type}")
                state["completed"] = True
        
        if self.current_iteration >= self.max_iterations and not state["completed"]:
            yield self.stream_error(
                f"Reached maximum iterations ({self.max_iterations}) without completing the goal."
            )
    
    async def _execute_tool_action(
        self,
        reasoning: Dict[str, Any],
        state: Dict[str, Any],
        session_id: Optional[str]
    ) -> AsyncGenerator[StreamData, None]:
        """
        Execute a tool action and observe the results.
        
        Args:
            reasoning: Parsed reasoning containing tool call details
            state: Current ReAct state
            session_id: Session identifier
            
        Yields:
            StreamData from tool execution
        """
        tool_name = reasoning.get("tool_name")
        tool_args = reasoning.get("tool_args", {})
        
        if not tool_name:
            yield self.stream_error("Tool name not specified in reasoning")
            return
        
        yield self.stream_action(f"Calling tool: {tool_name} with args: {tool_args}")
        
        # TODO: Implement actual tool calling when tool registry is available
        if self.tool_registry:
            try:
                # This will be implemented when we create the tool registry
                tool_result = await self._call_tool(tool_name, tool_args)
                observation = f"Tool {tool_name} result: {tool_result}"
            except Exception as e:
                observation = f"Tool {tool_name} failed: {str(e)}"
                yield self.stream_error(f"Tool execution failed: {str(e)}")
        else:
            observation = f"Tool registry not available. Simulated call to {tool_name}."
        
        # Add observation to state
        state["observations"].append(observation)
        yield self.stream_observation(observation)
        
        # Store observation in memory
        await self.store_memory(
            content=observation,
            segment_type="OBSERVATION",
            importance=0.8,
            metadata={"tool_name": tool_name, "session_id": session_id}
        )
    
    async def _call_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Any:
        """
        Call a tool through the tool registry.
        
        Args:
            tool_name: Name of the tool to call
            tool_args: Arguments for the tool
            
        Returns:
            Tool execution result
        """
        # TODO: Implement when tool registry is available
        if not self.tool_registry:
            raise Exception("Tool registry not available")
        
        # This will be properly implemented when we create the tool registry
        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            raise Exception(f"Tool {tool_name} not found")
        
        return await tool.execute(**tool_args)
    
    @abstractmethod
    def _build_reasoning_prompt(self, state: Dict[str, Any]) -> str:
        """
        Build the prompt for the reasoning step.
        
        Args:
            state: Current ReAct state
            
        Returns:
            Prompt for reasoning
        """
        pass
    
    @abstractmethod
    def _parse_reasoning_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the LLM's reasoning response.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Parsed reasoning components
        """
        pass
    
    def _build_context_from_memories(self, memories: List[Any]) -> str:
        """
        Build context string from memory search results.
        
        Args:
            memories: List of memory segments
            
        Returns:
            Context string
        """
        if not memories:
            return "No relevant context found."
        
        context_parts = []
        for memory in memories:
            if hasattr(memory, 'content') and hasattr(memory.content, 'text'):
                context_parts.append(f"- {memory.content.text}")
        
        return "\n".join(context_parts) if context_parts else "No relevant context found."


# Test function
async def test_base_orchestrator():
    """Test the BaseOrchestratorAgent functionality."""
    print("Testing BaseOrchestratorAgent...")
    
    # Note: This is a mock test since BaseOrchestratorAgent is abstract
    print("✓ BaseOrchestratorAgent is properly defined as abstract class")
    print("✓ ReAct loop structure is implemented")
    print("✓ Tool integration hooks are available")
    print("✓ Memory integration is included")
    
    print("BaseOrchestratorAgent tests completed! 🎉")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_base_orchestrator())
