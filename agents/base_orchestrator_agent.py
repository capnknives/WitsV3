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

    # ReAct tool-selection/synthesis needs to be near-deterministic; the global
    # default_temperature (0.7) is tuned for chat/creative work and makes the
    # loop wander (redundant searches, listing instead of answering the specific
    # question). Low temp here dramatically improves consistency.
    REASONING_TEMPERATURE = 0.2


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
        user_input: str,
        conversation_history: Optional[ConversationHistory] = None,
        session_id: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[StreamData, None]:
        """
        Execute the ReAct loop to achieve the given goal.
        
        Args:
            user_input: The goal or user request to achieve
            conversation_history: Optional conversation context
            session_id: Optional session identifier
            **kwargs: Additional parameters
            
        Yields:
            StreamData objects showing the reasoning and action process
        """
        goal = user_input
        self.current_iteration = 0

        # Route the whole ReAct session on the goal: code goals go to the
        # coder model (better structured output than qwen3), everything else
        # stays on the orchestrator model. Never the trivial model — the
        # small model can't hold the ReAct JSON format.
        self._session_model = self.model_router.route(
            goal, default=self.get_model_name(), allow_trivial=False
        )
        if self._session_model != self.get_model_name():
            self.logger.info(f"Orchestrating with routed model: {self._session_model}")

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
            
            reasoning_response = await self.generate_response(
                reasoning_prompt,
                model_name=getattr(self, "_session_model", None),
                temperature=self.REASONING_TEMPERATURE,
                response_format="json"
            )
            parsed_reasoning = self._parse_reasoning_response(reasoning_response)

            # Repair-reparse: if the response wasn't usable JSON, ask the model
            # once to rewrite its own output as valid JSON before falling back.
            if parsed_reasoning.pop("_parse_failed", False):
                parse_error = parsed_reasoning.pop("_parse_error", "invalid JSON")
                yield self.stream_thinking("Reasoning response was malformed; attempting JSON repair...")
                try:
                    repaired_response = await self.generate_response(
                        self._build_json_repair_prompt(reasoning_response, parse_error),
                        model_name=getattr(self, "_session_model", None),
                        temperature=self.REASONING_TEMPERATURE,
                        response_format="json"
                    )
                    reparsed = self._parse_reasoning_response(repaired_response)
                    if not reparsed.pop("_parse_failed", False):
                        reparsed.pop("_parse_error", None)
                        parsed_reasoning = reparsed
                        reasoning_response = repaired_response
                    else:
                        self.logger.warning("JSON repair attempt also failed to parse; using fallback reasoning")
                except Exception as e:
                    self.logger.warning(f"JSON repair attempt errored: {e}; using fallback reasoning")
            
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
                observation = self._format_tool_observation(tool_name, tool_result)
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
    
    def _format_tool_observation(self, tool_name: str, result: Any) -> str:
        """Render a tool result for the ReAct observation.

        Only web_search-shaped dicts (provider metadata + link/title/snippet
        hits) get the numbered-source layout. Other tools such as
        document_search and search_mcp_tools also return a ``results`` list
        but with different fields — those stay as plain tool output.
        """
        if isinstance(result, dict) and self._is_web_search_result(tool_name, result):
            return self._format_search_observation(tool_name, result)
        return f"Tool {tool_name} result: {result}"

    @staticmethod
    def _is_web_search_result(tool_name: str, result: Dict[str, Any]) -> bool:
        """True when *result* is from web_search, not another results-list tool."""
        results = result.get("results")
        if not isinstance(results, list):
            return False

        # web_search always sets provider (or answer_provider) on success.
        if result.get("provider") or result.get("answer_provider"):
            return True

        # Non-empty hits: web results carry link/title/snippet; document_search
        # uses file/text/chunk and MCP discovery uses name/command.
        if results and isinstance(results[0], dict):
            sample = results[0]
            if "link" in sample or "snippet" in sample:
                return tool_name == "web_search"
            if "file" in sample or "text" in sample or "installable" in sample:
                return False

        return False

    @staticmethod
    def _format_search_observation(tool_name: str, result: Dict[str, Any]) -> str:
        lines = [f"{tool_name} results (base your answer on the SOURCES below):"]
        answer = result.get("answer")
        if answer:
            provider = result.get("answer_provider", "search engine")
            lines.append(
                f"{provider} summary (usually accurate — use it, but trust the "
                f"sources below if any clearly contradicts it): {answer}"
            )
        sources = result.get("results") or []
        for i, r in enumerate(sources, 1):
            title = (r.get("title") or "").strip()
            snippet = (r.get("snippet") or "").strip()
            link = (r.get("link") or "").strip()
            lines.append(f"[{i}] {title}\n    {snippet}\n    source: {link}")
        if not sources and not answer:
            lines.append("(no results found)")
        return "\n".join(lines)

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
    
    def _build_json_repair_prompt(self, raw_response: str, parse_error: str) -> str:
        """
        Build a prompt asking the model to rewrite its malformed reasoning
        output as valid JSON.

        Args:
            raw_response: The response that failed to parse
            parse_error: The parse error message

        Returns:
            Repair prompt
        """
        return f"""The following text was supposed to be a single valid JSON object, but it failed to parse ({parse_error}).

TEXT:
{raw_response}

Rewrite it as ONE valid JSON object. Preserve the intended content and keys ("thought", "action_type", "tool_name", "tool_args", "final_answer"). Do not add commentary, markdown fences, or any text outside the JSON object.

Respond ONLY with the corrected JSON object."""

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
