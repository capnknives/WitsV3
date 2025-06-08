"""
WitsV3 Response Parsing Module

This module provides robust parsing of LLM responses for tool calls, 
structured data, and error handling.
"""

from .base_parser import ResponseType, ParsedResponse, BaseParser
from .json_parser import JSONParser
from .react_parser import ReactParser
from .format_detector import FormatDetector
from .parser_factory import ParserFactory, create_response_parser
from .prompt_builder import create_structured_prompt

# Main parser for backward compatibility
ResponseParser = ParserFactory.create_parser

__all__ = [
    # Types
    'ResponseType',
    'ParsedResponse',
    
    # Base classes
    'BaseParser',
    
    # Parsers
    'JSONParser',
    'ReactParser',
    
    # Factory
    'ParserFactory',
    'create_response_parser',
    'ResponseParser',
    
    # Utilities
    'FormatDetector',
    'create_structured_prompt',
]