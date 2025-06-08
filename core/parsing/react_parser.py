"""
ReAct (Reason-Act-Observe) pattern parser for LLM responses
"""

import re
import json
import logging
from typing import Dict, Any, List, Optional, Tuple

from ..schemas import ToolCall
from .base_parser import BaseParser, ParsedResponse, ResponseType

logger = logging.getLogger(__name__)


class ReactParser(BaseParser):
    """Specialized parser for ReAct (Reason-Act-Observe) format responses"""
    
    def __init__(self):
        super().__init__()
        self.section_patterns = {
            'thought': r'(?:Thought|Reasoning|Think):\s*(.+?)(?=\n(?:Action|Tool|Observation|$))',
            'action': r'(?:Action|Tool):\s*(.+?)(?=\n(?:Thought|Observation|$))',
            'observation': r'(?:Observation|Result|Output):\s*(.+?)(?=\n(?:Thought|Action|$)|$)',
            'final_answer': r'(?:Final Answer|Answer|Conclusion):\s*(.+?)(?=$)',
        }
        
        self.tool_patterns = [
            # Function call format: tool_name(arg1=val1, arg2=val2)
            r'(\w+)\s*\(\s*([^)]*)\s*\)',
            # JSON inline format
            r'\{[^}]*"tool"[^}]*\}',
            # Markdown format: [tool:name](arguments)
            r'\[tool:([^\]]+)\]\(([^)]*)\)',
        ]
    
    def can_parse(self, response: str) -> bool:
        """Check if response contains ReAct patterns"""
        # Check for ReAct section markers
        for pattern in self.section_patterns.values():
            if re.search(pattern, response, re.IGNORECASE | re.MULTILINE):
                return True
        return False
    
    def parse_response(self, response: str, context: Optional[Dict[str, Any]] = None) -> ParsedResponse:
        """Parse ReAct format response"""
        try:
            response = response.strip()
            if not response:
                return ParsedResponse(
                    response_type=ResponseType.ERROR,
                    content="Empty response",
                    tool_calls=[],
                    confidence=0.0
                )
            
            # Parse ReAct sections
            thought, tool_calls, observation = self.parse_react_response(response)
            final_answer = self._extract_section(response, 'final_answer')
            
            # Determine response type and content
            if tool_calls:
                response_type = ResponseType.TOOL_CALL
                content = thought or "Executing tool call"
            elif final_answer:
                response_type = ResponseType.FINAL_ANSWER
                content = final_answer
            elif observation:
                response_type = ResponseType.OBSERVATION
                content = observation
            elif thought:
                response_type = ResponseType.REASONING
                content = thought
            else:
                response_type = ResponseType.REASONING
                content = response
            
            # Calculate confidence
            parsed = ParsedResponse(
                response_type=response_type,
                content=content,
                tool_calls=tool_calls,
                reasoning=thought
            )
            confidence = self.calculate_confidence(response, parsed)
            
            # Add ReAct-specific confidence boost
            if thought and (tool_calls or observation):
                confidence = min(confidence + 0.2, 1.0)
            
            return ParsedResponse(
                response_type=response_type,
                content=content,
                tool_calls=tool_calls,
                reasoning=thought,
                confidence=confidence,
                metadata={
                    "original_response": response,
                    "observation": observation,
                    "parser": "ReactParser",
                    "react_sections_found": self._count_sections(response)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing ReAct response: {e}")
            return ParsedResponse(
                response_type=ResponseType.ERROR,
                content=f"ReAct parse error: {str(e)}",
                tool_calls=[],
                confidence=0.0,
                metadata={"error": str(e), "original_response": response}
            )
    
    def parse_react_response(self, response: str) -> Tuple[Optional[str], List[ToolCall], Optional[str]]:
        """
        Parse a ReAct format response into thought, actions, and observations
        
        Returns:
            Tuple of (thought, tool_calls, observation)
        """
        # Extract sections
        thought = self._extract_section(response, 'thought')
        action_text = self._extract_section(response, 'action')
        observation = self._extract_section(response, 'observation')
        
        # Parse tool calls from action section
        tool_calls = []
        if action_text:
            tool_calls = self._extract_tool_calls_from_action(action_text)
        
        return thought, tool_calls, observation
    
    def _extract_section(self, text: str, section_type: str) -> Optional[str]:
        """Extract a specific section from ReAct response"""
        if section_type not in self.section_patterns:
            return None
        
        pattern = self.section_patterns[section_type]
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
        
        if matches:
            # Join multiple matches if found
            return '\n'.join(match.strip() for match in matches)
        
        return None
    
    def _extract_tool_calls_from_action(self, action_text: str) -> List[ToolCall]:
        """Extract tool calls from action section"""
        tool_calls = []
        call_id_counter = 1
        
        # Try function call format first
        func_pattern = r'(\w+)\s*\(\s*([^)]*)\s*\)'
        func_matches = re.findall(func_pattern, action_text)
        
        for func_name, args_str in func_matches:
            try:
                arguments = self._parse_function_arguments(args_str)
                tool_call = ToolCall(
                    call_id=f"call_{call_id_counter}",
                    tool_name=func_name,
                    arguments=arguments
                )
                tool_calls.append(tool_call)
                call_id_counter += 1
            except Exception as e:
                self.logger.debug(f"Failed to parse function call {func_name}: {e}")
        
        # Try JSON format if no function calls found
        if not tool_calls:
            json_pattern = r'\{[^}]*"tool"[^}]*\}'
            json_matches = re.findall(json_pattern, action_text)
            
            for match in json_matches:
                try:
                    data = json.loads(match)
                    if 'tool' in data:
                        tool_call = ToolCall(
                            call_id=f"call_{call_id_counter}",
                            tool_name=data['tool'],
                            arguments=data.get('arguments', data.get('args', {}))
                        )
                        tool_calls.append(tool_call)
                        call_id_counter += 1
                except json.JSONDecodeError:
                    continue
        
        return tool_calls
    
    def _parse_function_arguments(self, args_str: str) -> Dict[str, Any]:
        """Parse function-style arguments into a dictionary"""
        arguments = {}
        
        if not args_str.strip():
            return arguments
        
        # Simple key=value parsing
        for arg in args_str.split(','):
            if '=' in arg:
                key, value = arg.split('=', 1)
                key = key.strip().strip('"\'')
                value = value.strip().strip('"\'')
                
                # Try to parse value types
                if value.startswith(('{', '[')):
                    try:
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        pass
                elif value.lower() in ('true', 'false'):
                    value = value.lower() == 'true'
                elif value.replace('.', '').replace('-', '').isdigit():
                    value = float(value) if '.' in value else int(value)
                
                arguments[key] = value
        
        return arguments
    
    def _count_sections(self, text: str) -> int:
        """Count how many ReAct sections are found in the text"""
        count = 0
        for section_type in self.section_patterns:
            if self._extract_section(text, section_type):
                count += 1
        return count