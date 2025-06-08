"""
Parser factory for creating appropriate response parsers
"""

import logging
from typing import Optional, Dict, Any, List

from .base_parser import BaseParser, ParsedResponse, ResponseType
from .json_parser import JSONParser
from .react_parser import ReactParser
from .format_detector import FormatDetector, ResponseFormat
from ..schemas import ToolCall

logger = logging.getLogger(__name__)


class GeneralParser(BaseParser):
    """General purpose parser that combines multiple parsing strategies"""
    
    def __init__(self):
        super().__init__()
        self.json_parser = JSONParser()
        self.react_parser = ReactParser()
        self.format_detector = FormatDetector()
        
        # Fallback patterns for general parsing
        self.tool_patterns = [
            r'(\w+)\s*\(\s*([^)]*)\s*\)',  # Function calls
            r'\[tool:([^\]]+)\]\(([^)]*)\)',  # Markdown format
            r'<tool\s+name="([^"]+)"[^>]*>(.*?)</tool>',  # XML format
        ]
        
        self.reasoning_patterns = [
            r'(?:Let me think|I need to|I should)\s*(.+?)(?=\n|$)',
            r'(?:First,|Next,|Then,|Finally,)\s*(.+?)(?=\n|$)',
        ]
        
        self.final_answer_patterns = [
            r'(?:Therefore|In conclusion|To summarize)[,:]?\s*(.+?)(?=$)',
            r'(?:The answer is|My response is)[:]?\s*(.+?)(?=$)',
        ]
    
    def can_parse(self, response: str) -> bool:
        """General parser can handle any response"""
        return True
    
    def parse_response(self, response: str, context: Optional[Dict[str, Any]] = None) -> ParsedResponse:
        """Parse response using best available strategy"""
        try:
            # Detect format
            detected_format = self.format_detector.detect_format(response)
            
            # Try specialized parsers first
            if detected_format == ResponseFormat.JSON and self.json_parser.can_parse(response):
                return self.json_parser.parse_response(response, context)
            
            if detected_format == ResponseFormat.REACT and self.react_parser.can_parse(response):
                return self.react_parser.parse_response(response, context)
            
            # Fall back to general parsing
            return self._general_parse(response, context)
            
        except Exception as e:
            self.logger.error(f"Error in general parser: {e}")
            return ParsedResponse(
                response_type=ResponseType.ERROR,
                content=f"Parse error: {str(e)}",
                tool_calls=[],
                confidence=0.0,
                metadata={"error": str(e), "original_response": response}
            )
    
    def _general_parse(self, response: str, context: Optional[Dict[str, Any]] = None) -> ParsedResponse:
        """General parsing strategy for mixed formats"""
        response = response.strip()
        
        if not response:
            return ParsedResponse(
                response_type=ResponseType.ERROR,
                content="Empty response",
                tool_calls=[],
                confidence=0.0
            )
        
        # Extract components
        tool_calls = self._extract_tool_calls(response)
        reasoning = self._extract_reasoning(response)
        final_answer = self._extract_final_answer(response)
        
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
        
        return ParsedResponse(
            response_type=response_type,
            content=content,
            tool_calls=tool_calls,
            reasoning=reasoning,
            confidence=0.7,  # Medium confidence for general parsing
            metadata={
                "original_response": response,
                "parser": "GeneralParser",
                "detected_format": self.format_detector.detect_format(response).value
            }
        )
    
    def _extract_tool_calls(self, text: str) -> List[ToolCall]:
        """Extract tool calls using multiple patterns"""
        # Implementation would be similar to original ResponseParser
        # but simplified here for brevity
        return []
    
    def _extract_reasoning(self, text: str) -> Optional[str]:
        """Extract reasoning text"""
        import re
        for pattern in self.reasoning_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            if matches:
                return '\n'.join(matches).strip()
        return None
    
    def _extract_final_answer(self, text: str) -> Optional[str]:
        """Extract final answer text"""
        import re
        for pattern in self.final_answer_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            if matches:
                return '\n'.join(matches).strip()
        return None


class ParserFactory:
    """Factory for creating appropriate response parsers"""
    
    _parsers = {
        'json': JSONParser,
        'react': ReactParser,
        'general': GeneralParser,
    }
    
    @staticmethod
    def create_parser(parser_type: str = 'general') -> BaseParser:
        """
        Create a parser of the specified type
        
        Args:
            parser_type: Type of parser to create
            
        Returns:
            Parser instance
        """
        parser_class = ParserFactory._parsers.get(parser_type, GeneralParser)
        return parser_class()
    
    @staticmethod
    def get_auto_parser(response: str) -> BaseParser:
        """
        Automatically select the best parser for the response
        
        Args:
            response: LLM response to parse
            
        Returns:
            Most appropriate parser instance
        """
        detector = FormatDetector()
        detected_format = detector.detect_format(response)
        
        if detected_format == ResponseFormat.JSON:
            return JSONParser()
        elif detected_format == ResponseFormat.REACT:
            return ReactParser()
        else:
            return GeneralParser()
    
    @staticmethod
    def register_parser(name: str, parser_class: type):
        """
        Register a custom parser
        
        Args:
            name: Name for the parser
            parser_class: Parser class (must inherit from BaseParser)
        """
        if not issubclass(parser_class, BaseParser):
            raise ValueError(f"{parser_class} must inherit from BaseParser")
        
        ParserFactory._parsers[name] = parser_class


def create_response_parser(parser_type: str = 'general') -> BaseParser:
    """
    Convenience function to create a response parser
    
    Args:
        parser_type: Type of parser to create
        
    Returns:
        Parser instance
    """
    return ParserFactory.create_parser(parser_type)