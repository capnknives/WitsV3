"""
Base classes and types for response parsing
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, List, Optional

from ..schemas import ToolCall

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


class BaseParser(ABC):
    """Abstract base class for all response parsers"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def parse_response(self, response: str, context: Optional[Dict[str, Any]] = None) -> ParsedResponse:
        """
        Parse an LLM response and extract structured information
        
        Args:
            response: Raw LLM response text
            context: Optional context for parsing hints
            
        Returns:
            ParsedResponse with extracted information
        """
        pass
    
    @abstractmethod
    def can_parse(self, response: str) -> bool:
        """
        Check if this parser can handle the given response
        
        Args:
            response: Raw LLM response text
            
        Returns:
            True if this parser can handle the response
        """
        pass
    
    def calculate_confidence(self, response: str, parsed: ParsedResponse) -> float:
        """
        Calculate confidence score for parsing result
        
        Args:
            response: Original response text
            parsed: Parsed response
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        confidence = 0.5  # Base confidence
        
        # Boost confidence for successful extractions
        if parsed.tool_calls:
            confidence += 0.3
        if parsed.reasoning:
            confidence += 0.1
        if parsed.response_type == ResponseType.FINAL_ANSWER:
            confidence += 0.1
        
        return min(confidence, 1.0)