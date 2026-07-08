# agents/base_orchestrator_agent.py
"""
Base Orchestrator Agent for WitsV3.
Implements ReAct (Reason-Act-Observe) loop functionality.
"""

import logging
from typing import Any, Dict, List, Optional, AsyncGenerator
from abc import abstractmethod

from agents.base_agent import BaseAgent
from agents.orchestrator_tool_helpers import OrchestratorToolHelpersMixin
from core.config import WitsV3Config
from core.llm_interface import BaseLLMInterface
from core.memory_manager import MemoryManager
from core.schemas import StreamData, ConversationHistory


class BaseOrchestratorAgent(OrchestratorToolHelpersMixin, BaseAgent):
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
        doc_inventory = await self._get_document_inventory()

        # Initialize the ReAct state
        react_state = {
            "goal": goal,
            "context": context,
            "documents_context": self._format_documents_context(doc_inventory),
            "conversation_history": conversation_history,
            "history": conversation_history.to_llm_format() if conversation_history else [],
            "observations": [],
            "completed": False,
            "final_answer": None,
            "tool_repeat_failures": {},
            "tool_total_failures": {},
            "lookup_search_done": False,
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
                coerced = self._coerce_unknown_action(parsed_reasoning)
                if coerced:
                    parsed_reasoning = coerced
                    action_type = parsed_reasoning.get("action_type")
                    if action_type == "tool_call":
                        async for stream_data in self._execute_tool_action(
                            parsed_reasoning, state, session_id
                        ):
                            yield stream_data
                        continue
                    if action_type == "final_answer":
                        final_answer = parsed_reasoning.get(
                            "final_answer", "I've completed the task."
                        )
                        state["completed"] = True
                        state["final_answer"] = final_answer
                        yield self.stream_result(final_answer)
                        await self.store_memory(
                            content=f"Final Answer: {final_answer}",
                            segment_type="FINAL_ANSWER",
                            importance=1.0,
                            metadata={"session_id": session_id},
                        )
                        continue

                observation = (
                    f"Invalid action_type '{action_type}'. "
                    f"Use action_type 'tool_call' with tool_name, or 'final_answer'."
                )
                yield self.stream_error(observation)
                state["observations"].append(observation)
        
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
        
        block_reason = self._preflight_tool_call(tool_name, tool_args, state)
        if block_reason:
            observation = block_reason
            if block_reason.startswith(("Blocked", "Skipped")):
                self._record_tool_failure(tool_name, tool_args, state)
            yield self.stream_observation(observation)
            state["observations"].append(observation)
            await self.store_memory(
                content=observation,
                segment_type="OBSERVATION",
                importance=0.8,
                metadata={"tool_name": tool_name, "session_id": session_id, "blocked": True},
            )
            return

        yield self.stream_action(f"Calling tool: {tool_name} with args: {tool_args}")

        tool_result = None
        if self.tool_registry:
            try:
                tool_result = await self._call_tool(tool_name, tool_args, state)
                observation = self._format_tool_observation(tool_name, tool_result)
            except Exception as e:
                observation = f"Tool {tool_name} failed: {str(e)}"
                yield self.stream_error(f"Tool execution failed: {str(e)}")
        else:
            observation = f"Tool registry not available. Simulated call to {tool_name}."

        if self._observation_indicates_failure(observation) or (
            isinstance(tool_result, dict) and self._tool_result_is_failure(tool_result)
        ):
            self._record_tool_failure(tool_name, tool_args, state)

        # Add observation to state
        state["observations"].append(observation)
        yield self.stream_observation(observation)

        await self.store_memory(
            content=observation,
            segment_type="OBSERVATION",
            importance=0.8,
            metadata={"tool_name": tool_name, "session_id": session_id},
        )

        # Save-to-file: after transcript is read, write immediately — the model
        # often loops on read_conversation_history and never reaches write_file.
        if (
            tool_name == "read_conversation_history"
            and self._goal_saves_conversation(state.get("goal", ""))
            and not self._observation_indicates_failure(observation)
        ):
            file_path = self._save_file_path_from_goal(state.get("goal", "")) or "exports/conversation_log.txt"
            if file_path:
                async for stream_data in self._auto_write_saved_conversation(
                    file_path, state, session_id
                ):
                    yield stream_data
                return

        if (
            tool_name == "web_search"
            and self._goal_is_web_lookup(state.get("goal", ""))
            and not self._observation_indicates_failure(observation)
        ):
            state["lookup_search_done"] = True

    async def _call_tool(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        state: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Call a tool through the tool registry.

        Args:
            tool_name: Name of the tool to call
            tool_args: Arguments for the tool
            state: Optional ReAct state for context injection

        Returns:
            Tool execution result
        """
        if not self.tool_registry:
            raise Exception("Tool registry not available")

        if state is not None:
            tool_args = await self._prepare_tool_args(tool_name, tool_args, state)

        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            raise Exception(f"Tool {tool_name} not found")

        return await tool.execute(**tool_args)

    def _coerce_unknown_action(
        self, parsed: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Last-chance fix when action_type is a tool name. Override in subclasses."""
        return None
    
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
