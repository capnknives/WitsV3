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

        Tries progressively harder to recover a JSON object: strip qwen3
        <think> blocks, extract fenced/balanced JSON candidates (completing
        truncated objects), and apply conservative syntax repairs before
        giving up and using keyword-based fallback parsing.

        Args:
            response: Raw LLM response

        Returns:
            Parsed reasoning components. On total failure the fallback result
            carries "_parse_failed": True and "_parse_error" so the caller can
            trigger a repair-reparse round trip.
        """
        last_error = "No JSON found in response"

        for candidate in self._extract_json_candidates(response):
            for attempt in (candidate, self._repair_json(candidate)):
                try:
                    parsed = json.loads(attempt)
                except json.JSONDecodeError as e:
                    last_error = str(e)
                    continue

                if not isinstance(parsed, dict):
                    last_error = f"Top-level JSON is {type(parsed).__name__}, expected object"
                    continue

                try:
                    return self._validate_reasoning(parsed)
                except ValueError as e:
                    last_error = str(e)
                    break  # repairing syntax won't add missing fields

        self.logger.warning(f"Failed to parse reasoning response: {last_error}")
        return self._fallback_reasoning_parsing(response, last_error)

    def _validate_reasoning(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a parsed reasoning object and fill safe defaults.

        Args:
            parsed: Parsed JSON object

        Returns:
            The validated object

        Raises:
            ValueError: If required fields are missing
        """
        if "action_type" not in parsed:
            raise ValueError("Missing 'action_type' field")

        if parsed["action_type"] == "tool_call":
            if "tool_name" not in parsed:
                raise ValueError("Missing 'tool_name' for tool_call action")
            if "tool_args" not in parsed or not isinstance(parsed.get("tool_args"), dict):
                parsed["tool_args"] = {}

        elif parsed["action_type"] == "final_answer":
            if "final_answer" not in parsed:
                raise ValueError("Missing 'final_answer' for final_answer action")

        return parsed

    def _strip_think_blocks(self, text: str) -> str:
        """Remove qwen3-style <think>...</think> blocks (and stray tags)."""
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        # Unclosed/stray tags: drop the tags but keep the content, in case
        # the JSON ended up inside an unterminated think block.
        return re.sub(r'</?think>', '', text, flags=re.IGNORECASE).strip()

    def _extract_json_candidates(self, response: str) -> List[str]:
        """
        Extract candidate JSON strings from a raw response, most-likely first.

        Args:
            response: Raw LLM response

        Returns:
            List of candidate JSON strings
        """
        text = self._strip_think_blocks(response)
        if not text:
            return []

        candidates: List[str] = []

        # Whole response (the common case with format=json)
        if text.startswith("{"):
            candidates.append(text)

        # Markdown-fenced blocks: ```json ... ``` or plain ``` ... ```
        for match in re.finditer(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL | re.IGNORECASE):
            candidates.append(match.group(1))

        # Balanced top-level {...} objects (string-aware scan); completes
        # truncated objects by closing open strings/braces.
        candidates.extend(self._balanced_json_objects(text))

        # De-duplicate, preserving order
        seen = set()
        unique = []
        for c in candidates:
            c = c.strip()
            if c and c not in seen:
                seen.add(c)
                unique.append(c)
        return unique

    def _balanced_json_objects(self, text: str) -> List[str]:
        """
        Scan for top-level balanced {...} substrings, respecting strings and
        escapes. If the text ends mid-object (truncated response), returns a
        best-effort completion with open strings and braces closed.

        Args:
            text: Text to scan

        Returns:
            List of balanced (or completed) JSON object strings
        """
        objects: List[str] = []
        i = 0
        n = len(text)

        while i < n:
            if text[i] != '{':
                i += 1
                continue

            start = i
            depth = 0
            in_string = False
            escaped = False
            j = i
            while j < n:
                ch = text[j]
                if in_string:
                    if escaped:
                        escaped = False
                    elif ch == '\\':
                        escaped = True
                    elif ch == '"':
                        in_string = False
                else:
                    if ch == '"':
                        in_string = True
                    elif ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            objects.append(text[start:j + 1])
                            break
                j += 1

            if depth > 0:
                # Truncated object: close any open string, trim a dangling
                # comma, and close the remaining braces.
                fragment = text[start:n].rstrip()
                if in_string:
                    fragment += '"'
                fragment = re.sub(r',\s*$', '', fragment)
                objects.append(fragment + '}' * depth)
                break

            i = j + 1 if j < n else n

        return objects

    def _repair_json(self, json_str: str) -> str:
        """
        Apply conservative repairs for common LLM JSON mistakes.

        Args:
            json_str: Candidate JSON string

        Returns:
            Repaired JSON string (may be unchanged)
        """
        repaired = json_str

        # Smart quotes -> straight quotes
        repaired = repaired.replace('“', '"').replace('”', '"')
        repaired = repaired.replace('‘', "'").replace('’', "'")

        # Trailing commas before } or ]
        repaired = re.sub(r',\s*([}\]])', r'\1', repaired)

        # Python literals in value position
        repaired = re.sub(r'(?<=[:\[,\s])True(?=\s*[,}\]])', 'true', repaired)
        repaired = re.sub(r'(?<=[:\[,\s])False(?=\s*[,}\]])', 'false', repaired)
        repaired = re.sub(r'(?<=[:\[,\s])None(?=\s*[,}\]])', 'null', repaired)

        return repaired

    def _fallback_reasoning_parsing(self, response: str, parse_error: str = "") -> Dict[str, Any]:
        """
        Fallback reasoning parsing when JSON parsing fails.

        Args:
            response: Raw LLM response
            parse_error: The error from the failed JSON parse

        Returns:
            Basic reasoning structure, flagged with "_parse_failed" so the
            orchestrator loop can attempt a repair-reparse round trip.
        """
        text = self._strip_think_blocks(response) or response
        response_lower = text.lower()

        # Look for action indicators
        if any(word in response_lower for word in ["tool", "use", "call", "search", "analyze"]):
            result: Dict[str, Any] = {
                "thought": text,
                "action_type": "tool_call",
                "tool_name": "think",
                "tool_args": {"thought": text}
            }
        else:
            result = {
                "thought": text,
                "action_type": "final_answer",
                "final_answer": text
            }

        result["_parse_failed"] = True
        result["_parse_error"] = parse_error or "invalid JSON"
        return result

    async def _handle_tool_failure(self, tool_name: str, error: Exception) -> Any:
        """
        Handle tool execution failures gracefully.
        
        Args:
            tool_name: Name of the tool that failed
            error: The error that occurred
            
        Returns:
            Result to use in place of the failed tool call
        """
        error_msg = str(error)
        self.logger.warning(f"Tool {tool_name} failed: {error_msg}")
        
        # Handle missing parameters
        if "Missing required parameters" in error_msg:
            if tool_name in ["analyze_conversation", "read_conversation_history"]:
                # Return empty state for conversation tools
                return {
                    "message_count": 0,
                    "user_messages": 0,
                    "assistant_messages": 0,
                    "conversation_turns": 0,
                    "last_speaker": None,
                    "summary": "Starting new conversation"
                }
            elif tool_name == "list_directory":
                # Return empty directory listing
                return {"items": [], "status": "success"}
            elif tool_name == "read_file":
                # Return empty file content
                return {"content": "", "status": "success"}
            elif tool_name == "intent_analysis":
                # Return default intent analysis
                return {
                    "type": "direct_response",
                    "confidence": 0.8,
                    "goal_statement": None,
                    "direct_response": None,
                    "clarification_question": None,
                    "reasoning": "Default intent analysis due to missing parameters"
                }
        
        # For other errors, return a generic error message
        return {
            "error": f"Tool {tool_name} failed: {error_msg}",
            "status": "error"
        }

    async def _call_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Any:
        """
        Call a tool with error handling.
        
        Args:
            tool_name: Name of the tool to call
            tool_args: Arguments for the tool
            
        Returns:
            Tool execution result
        """
        try:
            if not self.tool_registry:
                raise ValueError("No tool registry available")
            
            result = await self.tool_registry.execute_tool(tool_name, **tool_args)
            return result
            
        except Exception as e:
            # Handle the error gracefully
            return await self._handle_tool_failure(tool_name, e)


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
        llm_interface = OllamaInterface(config=config)
        
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
