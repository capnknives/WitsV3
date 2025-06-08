"""
JSON format parser for LLM responses
"""

import re
import json
import logging
from typing import Dict, Any, List, Optional

from ..schemas import ToolCall
from .base_parser import BaseParser, ParsedResponse, ResponseType

logger = logging.getLogger(__name__)


class JSONParser(BaseParser):
    """Parser for JSON-formatted LLM responses"""
    
    def __init__(self):
        super().__init__()
        self.json_patterns = [
            # JSON code blocks
            r'```json\s*(\{.*?\})\s*```',
            r'```\s*(\{.*?\})\s*```',
            # Inline JSON objects
            r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})',
        ]
    
    def can_parse(self, response: str) -> bool:
        """Check if response contains JSON patterns"""
        for pattern in self.json_patterns:
            if re.search(pattern, response, re.DOTALL | re.IGNORECASE):
                return True
        return False
    
    def parse_response(self, response: str, context: Optional[Dict[str, Any]] = None) -> ParsedResponse:
        """Parse JSON-formatted response"""
        try:
            response = response.strip()
            if not response:
                return ParsedResponse(
                    response_type=ResponseType.ERROR,
                    content="Empty response",
                    tool_calls=[],
                    confidence=0.0
                )
            
            # Extract JSON objects
            json_objects = self._extract_json_objects(response)
            
            # Parse tool calls from JSON
            tool_calls = self._parse_json_tool_calls(json_objects)
            
            # Extract other information
            reasoning = self._extract_reasoning_from_json(json_objects)
            final_answer = self._extract_final_answer_from_json(json_objects)
            
            # Determine response type
            if tool_calls:
                response_type = ResponseType.TOOL_CALL
                content = reasoning or response
            elif final_answer:
                response_type = ResponseType.FINAL_ANSWER
                content = final_answer
            elif reasoning:
                response_type = ResponseType.REASONING
                content = reasoning
            else:
                response_type = ResponseType.REASONING
                content = response
            
            # Calculate confidence
            confidence = self.calculate_confidence(response, ParsedResponse(
                response_type=response_type,
                content=content,
                tool_calls=tool_calls,
                reasoning=reasoning
            ))
            
            return ParsedResponse(
                response_type=response_type,
                content=content,
                tool_calls=tool_calls,
                reasoning=reasoning,
                confidence=confidence,
                metadata={
                    "original_response": response,
                    "json_objects_found": len(json_objects),
                    "parser": "JSONParser"
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing JSON response: {e}")
            return ParsedResponse(
                response_type=ResponseType.ERROR,
                content=f"JSON parse error: {str(e)}",
                tool_calls=[],
                confidence=0.0,
                metadata={"error": str(e), "original_response": response}
            )
    
    def _extract_json_objects(self, text: str) -> List[Dict[str, Any]]:
        """Extract all JSON objects from text"""
        json_objects = []
        
        for pattern in self.json_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            
            for match in matches:
                try:
                    obj = json.loads(match)
                    if isinstance(obj, dict):
                        json_objects.append(obj)
                except json.JSONDecodeError:
                    continue
        
        return json_objects
    
    def _parse_json_tool_calls(self, json_objects: List[Dict[str, Any]]) -> List[ToolCall]:
        """Parse tool calls from JSON objects"""
        tool_calls = []
        call_id_counter = 1
        
        for obj in json_objects:
            # Check for tool call formats
            if 'tool' in obj or 'tool_name' in obj:
                tool_name = obj.get('tool') or obj.get('tool_name')
                arguments = obj.get('arguments', obj.get('args', obj.get('parameters', {})))
                
                tool_call = ToolCall(
                    call_id=f"call_{call_id_counter}",
                    tool_name=tool_name,
                    arguments=arguments
                )
                tool_calls.append(tool_call)
                call_id_counter += 1
                
            # Check for action format
            elif 'action' in obj and isinstance(obj['action'], dict):
                action = obj['action']
                if 'name' in action:
                    tool_call = ToolCall(
                        call_id=f"call_{call_id_counter}",
                        tool_name=action['name'],
                        arguments=action.get('parameters', action.get('args', {}))
                    )
                    tool_calls.append(tool_call)
                    call_id_counter += 1
        
        return tool_calls
    
    def _extract_reasoning_from_json(self, json_objects: List[Dict[str, Any]]) -> Optional[str]:
        """Extract reasoning from JSON objects"""
        for obj in json_objects:
            if 'thought' in obj:
                return obj['thought']
            elif 'reasoning' in obj:
                return obj['reasoning']
            elif 'thinking' in obj:
                return obj['thinking']
        return None
    
    def _extract_final_answer_from_json(self, json_objects: List[Dict[str, Any]]) -> Optional[str]:
        """Extract final answer from JSON objects"""
        for obj in json_objects:
            if 'final_answer' in obj:
                return obj['final_answer']
            elif 'answer' in obj:
                return obj['answer']
            elif 'conclusion' in obj:
                return obj['conclusion']
        return None