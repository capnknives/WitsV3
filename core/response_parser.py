"""
Response parsing utilities for WitsV3
Provides robust parsing of LLM responses for tool calls, structured data, and error handling
"""

import re
import json
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum

from .schemas import ToolCall, AgentResponse, StreamData

logger = logging.getLogger(__name__)


class ResponseType(Enum):
    """Types of responses that can be parsed"""
    TOOL_CALL = "tool_call"
    REASONING = "reasoning"
    FINAL_ANSWER = "final_answer"
    ERROR = "error"
    OBSERVATION = "observation"


@dataclass
class ParsedResponse:
    """Represents a parsed LLM response"""
    response_type: ResponseType
    content: str
    tool_calls: List[ToolCall]
    reasoning: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ResponseParser:
    """Parser for LLM responses with support for various formats and patterns"""
    
    def __init__(self):
        # Patterns for different response formats
        self.tool_call_patterns = [
            # JSON format: {"tool": "name", "arguments": {...}}
            r'```json\s*(\{[^`]+\})\s*```',
            r'\{[^}]*"tool"[^}]*\}',
            
            # Function call format: function_name(arg1=val1, arg2=val2)
            r'(\w+)\s*\(\s*([^)]*)\s*\)',
            
            # XML format: <tool name="name">...</tool>
            r'<tool\s+name="([^"]+)"[^>]*>(.*?)</tool>',
            
            # Markdown format: [tool:name](arguments)
            r'\[tool:([^\]]+)\]\(([^)]*)\)',
        ]
        
        self.reasoning_patterns = [
            # ReAct reasoning patterns
            r'(?:Thought|Reasoning|Think):\s*(.+?)(?=\n(?:Action|Tool|Observation|$))',
            r'(?:Let me think|I need to|I should)\s*(.+?)(?=\n|$)',
        ]
        
        self.observation_patterns = [
            r'(?:Observation|Result|Output):\s*(.+?)(?=\n(?:Thought|Action|$)|$)',
        ]
        
        self.final_answer_patterns = [
            r'(?:Final Answer|Answer|Conclusion):\s*(.+?)(?=$)',
            r'(?:Therefore|In conclusion|To summarize):\s*(.+?)(?=$)',
        ]
    
    def parse_response(self, response: str, context: Optional[Dict[str, Any]] = None) -> ParsedResponse:
        """
        Parse an LLM response and extract structured information
        
        Args:
            response: Raw LLM response text
            context: Optional context for parsing hints
            
        Returns:
            ParsedResponse with extracted information
        """
        try:
            response = response.strip()
            if not response:
                return ParsedResponse(
                    response_type=ResponseType.ERROR,
                    content="Empty response",
                    tool_calls=[],
                    confidence=0.0
                )
            
            # Try to extract tool calls
            tool_calls = self._extract_tool_calls(response)
            
            # Extract reasoning
            reasoning = self._extract_reasoning(response)
            
            # Extract observations
            observations = self._extract_observations(response)
            
            # Extract final answer
            final_answer = self._extract_final_answer(response)
            
            # Determine response type
            if tool_calls:
                response_type = ResponseType.TOOL_CALL
                content = reasoning or response
            elif final_answer:
                response_type = ResponseType.FINAL_ANSWER
                content = final_answer
            elif observations:
                response_type = ResponseType.OBSERVATION
                content = observations
            elif reasoning:
                response_type = ResponseType.REASONING
                content = reasoning
            else:
                response_type = ResponseType.REASONING
                content = response
            
            # Calculate confidence based on parsing success
            confidence = self._calculate_confidence(response, tool_calls, reasoning, final_answer)
            
            return ParsedResponse(
                response_type=response_type,
                content=content,
                tool_calls=tool_calls,
                reasoning=reasoning,
                confidence=confidence,
                metadata={
                    "original_response": response,
                    "has_observations": bool(observations),
                    "parsing_patterns_matched": self._get_matched_patterns(response)
                }
            )
            
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return ParsedResponse(
                response_type=ResponseType.ERROR,
                content=f"Parse error: {str(e)}",
                tool_calls=[],
                confidence=0.0,
                metadata={"error": str(e), "original_response": response}
            )
    
    def _extract_tool_calls(self, text: str) -> List[ToolCall]:
        """Extract tool calls from text using various patterns"""
        tool_calls = []
        call_id_counter = 1
        
        # Try JSON format first
        tool_calls.extend(self._parse_json_tool_calls(text, call_id_counter))
        call_id_counter += len(tool_calls)
        
        # Try function call format
        if not tool_calls:
            tool_calls.extend(self._parse_function_tool_calls(text, call_id_counter))
            call_id_counter += len(tool_calls)
        
        # Try XML format
        if not tool_calls:
            tool_calls.extend(self._parse_xml_tool_calls(text, call_id_counter))
            call_id_counter += len(tool_calls)
        
        # Try markdown format
        if not tool_calls:
            tool_calls.extend(self._parse_markdown_tool_calls(text, call_id_counter))
        
        return tool_calls
    
    def _parse_json_tool_calls(self, text: str, start_id: int) -> List[ToolCall]:
        """Parse JSON format tool calls"""
        tool_calls = []
        
        # Look for JSON blocks
        json_pattern = r'```json\s*(\{.*?\})\s*```'
        matches = re.findall(json_pattern, text, re.DOTALL | re.IGNORECASE)
        
        for i, match in enumerate(matches):
            try:
                data = json.loads(match)
                if isinstance(data, dict) and 'tool' in data:
                    tool_call = ToolCall(
                        call_id=f"call_{start_id + i}",
                        name=data['tool'],
                        arguments=data.get('arguments', data.get('args', {}))
                    )
                    tool_calls.append(tool_call)
            except json.JSONDecodeError:
                continue
        
        # Also look for inline JSON
        if not tool_calls:
            inline_pattern = r'\{[^}]*"tool"[^}]*\}'
            matches = re.findall(inline_pattern, text)
            
            for i, match in enumerate(matches):
                try:
                    data = json.loads(match)
                    if isinstance(data, dict) and 'tool' in data:
                        tool_call = ToolCall(
                            call_id=f"call_{start_id + i}",
                            name=data['tool'],
                            arguments=data.get('arguments', data.get('args', {}))
                        )
                        tool_calls.append(tool_call)
                except json.JSONDecodeError:
                    continue
        
        return tool_calls
    
    def _parse_function_tool_calls(self, text: str, start_id: int) -> List[ToolCall]:
        """Parse function call format tool calls"""
        tool_calls = []
        
        # Pattern: function_name(arg1=val1, arg2=val2)
        pattern = r'(\w+)\s*\(\s*([^)]*)\s*\)'
        matches = re.findall(pattern, text)
        
        for i, (func_name, args_str) in enumerate(matches):
            try:
                # Parse arguments
                arguments = {}
                if args_str.strip():
                    # Simple key=value parsing
                    for arg in args_str.split(','):
                        if '=' in arg:
                            key, value = arg.split('=', 1)
                            key = key.strip().strip('"\'')
                            value = value.strip().strip('"\'')
                            
                            # Try to parse as JSON if it looks like it
                            if value.startswith(('{', '[')):
                                try:
                                    value = json.loads(value)
                                except json.JSONDecodeError:
                                    pass
                            elif value.lower() in ('true', 'false'):
                                value = value.lower() == 'true'
                            elif value.isdigit():
                                value = int(value)
                            
                            arguments[key] = value
                
                tool_call = ToolCall(
                    call_id=f"call_{start_id + i}",
                    name=func_name,
                    arguments=arguments
                )
                tool_calls.append(tool_call)
                
            except Exception as e:
                logger.debug(f"Failed to parse function call {func_name}: {e}")
                continue
        
        return tool_calls
    
    def _parse_xml_tool_calls(self, text: str, start_id: int) -> List[ToolCall]:
        """Parse XML format tool calls"""
        tool_calls = []
        
        # Pattern: <tool name="name">arguments</tool>
        pattern = r'<tool\s+name="([^"]+)"[^>]*>(.*?)</tool>'
        matches = re.findall(pattern, text, re.DOTALL)
        
        for i, (tool_name, args_content) in enumerate(matches):
            try:
                # Try to parse arguments as JSON
                arguments = {}
                args_content = args_content.strip()
                
                if args_content:
                    try:
                        arguments = json.loads(args_content)
                    except json.JSONDecodeError:
                        # Fallback to simple key=value parsing
                        for line in args_content.split('\n'):
                            if '=' in line:
                                key, value = line.split('=', 1)
                                arguments[key.strip()] = value.strip()
                
                tool_call = ToolCall(
                    call_id=f"call_{start_id + i}",
                    name=tool_name,
                    arguments=arguments
                )
                tool_calls.append(tool_call)
                
            except Exception as e:
                logger.debug(f"Failed to parse XML tool call {tool_name}: {e}")
                continue
        
        return tool_calls
    
    def _parse_markdown_tool_calls(self, text: str, start_id: int) -> List[ToolCall]:
        """Parse markdown format tool calls"""
        tool_calls = []
        
        # Pattern: [tool:name](arguments)
        pattern = r'\[tool:([^\]]+)\]\(([^)]*)\)'
        matches = re.findall(pattern, text)
        
        for i, (tool_name, args_str) in enumerate(matches):
            try:
                arguments = {}
                if args_str.strip():
                    try:
                        arguments = json.loads(args_str)
                    except json.JSONDecodeError:
                        # Simple parsing
                        arguments = {"input": args_str}
                
                tool_call = ToolCall(
                    call_id=f"call_{start_id + i}",
                    name=tool_name,
                    arguments=arguments
                )
                tool_calls.append(tool_call)
                
            except Exception as e:
                logger.debug(f"Failed to parse markdown tool call {tool_name}: {e}")
                continue
        
        return tool_calls
    
    def _extract_reasoning(self, text: str) -> Optional[str]:
        """Extract reasoning/thinking text"""
        for pattern in self.reasoning_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            if matches:
                return '\n'.join(matches).strip()
        return None
    
    def _extract_observations(self, text: str) -> Optional[str]:
        """Extract observation text"""
        for pattern in self.observation_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            if matches:
                return '\n'.join(matches).strip()
        return None
    
    def _extract_final_answer(self, text: str) -> Optional[str]:
        """Extract final answer text"""
        for pattern in self.final_answer_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            if matches:
                return '\n'.join(matches).strip()
        return None
    
    def _calculate_confidence(self, text: str, tool_calls: List[ToolCall], 
                            reasoning: Optional[str], final_answer: Optional[str]) -> float:
        """Calculate confidence score for parsing"""
        confidence = 0.5  # Base confidence
        
        # Boost confidence for successful extractions
        if tool_calls:
            confidence += 0.3
        if reasoning:
            confidence += 0.1
        if final_answer:
            confidence += 0.1
        
        # Boost for structured patterns
        if re.search(r'```json', text, re.IGNORECASE):
            confidence += 0.1
        if re.search(r'(?:Thought|Action|Observation):', text, re.IGNORECASE):
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _get_matched_patterns(self, text: str) -> List[str]:
        """Get list of patterns that matched in the text"""
        patterns = []
        
        if re.search(r'```json', text, re.IGNORECASE):
            patterns.append("json_block")
        if re.search(r'\w+\s*\([^)]*\)', text):
            patterns.append("function_call")
        if re.search(r'<tool[^>]*>', text, re.IGNORECASE):
            patterns.append("xml_tool")
        if re.search(r'\[tool:[^\]]+\]', text, re.IGNORECASE):
            patterns.append("markdown_tool")
        if re.search(r'(?:Thought|Reasoning):', text, re.IGNORECASE):
            patterns.append("reasoning")
        if re.search(r'(?:Observation|Result):', text, re.IGNORECASE):
            patterns.append("observation")
        if re.search(r'(?:Final Answer|Answer):', text, re.IGNORECASE):
            patterns.append("final_answer")
        
        return patterns


class ReactParser(ResponseParser):
    """Specialized parser for ReAct (Reason-Act-Observe) format responses"""
    
    def parse_react_response(self, response: str) -> Tuple[Optional[str], List[ToolCall], Optional[str]]:
        """
        Parse a ReAct format response into thought, actions, and observations
        
        Returns:
            Tuple of (thought, tool_calls, observation)
        """
        # Split response into sections
        sections = self._split_react_sections(response)
        
        thought = sections.get('thought')
        action_text = sections.get('action', '')
        observation = sections.get('observation')
        
        # Parse tool calls from action section
        tool_calls = self._extract_tool_calls(action_text) if action_text else []
        
        return thought, tool_calls, observation
    
    def _split_react_sections(self, text: str) -> Dict[str, str]:
        """Split ReAct response into thought, action, observation sections"""
        sections = {}
        
        # Pattern to match ReAct sections
        section_pattern = r'(Thought|Action|Observation):\s*(.*?)(?=\n(?:Thought|Action|Observation):|$)'
        matches = re.findall(section_pattern, text, re.IGNORECASE | re.DOTALL)
        
        for section_type, content in matches:
            sections[section_type.lower()] = content.strip()
        
        return sections


def create_structured_prompt(tools: List[Dict[str, Any]], format_type: str = "json") -> str:
    """
    Create a structured prompt for tool usage
    
    Args:
        tools: List of available tools with their schemas
        format_type: Format for tool calls ("json", "function", "xml", "markdown")
        
    Returns:
        Formatted prompt string
    """
    if format_type == "json":
        return _create_json_prompt(tools)
    elif format_type == "function":
        return _create_function_prompt(tools)
    elif format_type == "xml":
        return _create_xml_prompt(tools)
    elif format_type == "markdown":
        return _create_markdown_prompt(tools)
    else:
        return _create_json_prompt(tools)  # Default


def _create_json_prompt(tools: List[Dict[str, Any]]) -> str:
    """Create JSON format prompt"""
    tool_list = "\n".join([f"- {tool['name']}: {tool.get('description', '')}" for tool in tools])
    
    return f"""You have access to the following tools:
{tool_list}

To use a tool, respond with a JSON object in this format:
```json
{{
    "tool": "tool_name",
    "arguments": {{
        "param1": "value1",
        "param2": "value2"
    }}
}}
```

Use the Thought-Action-Observation pattern:
- Thought: Explain your reasoning
- Action: Use a tool or provide final answer
- Observation: Note the result and plan next steps

For your final answer, start with "Final Answer:"
"""


def _create_function_prompt(tools: List[Dict[str, Any]]) -> str:
    """Create function call format prompt"""
    tool_list = "\n".join([f"- {tool['name']}: {tool.get('description', '')}" for tool in tools])
    
    return f"""You have access to the following tools:
{tool_list}

To use a tool, call it like a function:
tool_name(param1="value1", param2="value2")

Use the Thought-Action-Observation pattern:
- Thought: Explain your reasoning
- Action: Call a tool or provide final answer
- Observation: Note the result and plan next steps

For your final answer, start with "Final Answer:"
"""


def _create_xml_prompt(tools: List[Dict[str, Any]]) -> str:
    """Create XML format prompt"""
    tool_list = "\n".join([f"- {tool['name']}: {tool.get('description', '')}" for tool in tools])
    
    return f"""You have access to the following tools:
{tool_list}

To use a tool, use XML format:
<tool name="tool_name">
{{"param1": "value1", "param2": "value2"}}
</tool>

Use the Thought-Action-Observation pattern:
- Thought: Explain your reasoning
- Action: Use a tool or provide final answer
- Observation: Note the result and plan next steps

For your final answer, start with "Final Answer:"
"""


def _create_markdown_prompt(tools: List[Dict[str, Any]]) -> str:
    """Create markdown format prompt"""
    tool_list = "\n".join([f"- {tool['name']}: {tool.get('description', '')}" for tool in tools])
    
    return f"""You have access to the following tools:
{tool_list}

To use a tool, use markdown format:
[tool:tool_name](arguments_as_json)

Use the Thought-Action-Observation pattern:
- Thought: Explain your reasoning
- Action: Use a tool or provide final answer
- Observation: Note the result and plan next steps

For your final answer, start with "Final Answer:"
"""


# Test function
def test_response_parser():
    """Test the response parser functionality"""
    parser = ResponseParser()
    
    # Test various response formats
    test_responses = [
        # JSON format
        '''
        Thought: I need to read a file to understand the content.
        
        Action: I'll use the file reading tool.
        ```json
        {
            "tool": "file_read",
            "arguments": {"file_path": "/path/to/file.txt"}
        }
        ```
        ''',
        
        # Function format
        '''
        Thought: Let me check the current time.
        
        Action: get_current_time()
        ''',
        
        # XML format
        '''
        Thought: I should list the directory contents.
        
        Action: 
        <tool name="list_directory">
        {"path": "/home/user"}
        </tool>
        ''',
        
        # Final answer
        '''
        Thought: Based on my analysis of the file.
        
        Final Answer: The file contains configuration data for the application.
        '''
    ]
    
    for i, response in enumerate(test_responses):
        print(f"\n--- Test {i+1} ---")
        parsed = parser.parse_response(response)
        print(f"Type: {parsed.response_type}")
        print(f"Content: {parsed.content[:100]}...")
        print(f"Tool calls: {len(parsed.tool_calls)}")
        print(f"Confidence: {parsed.confidence}")
        
        if parsed.tool_calls:
            for call in parsed.tool_calls:
                print(f"  Tool: {call.name}, Args: {call.arguments}")


if __name__ == "__main__":
    test_response_parser()
