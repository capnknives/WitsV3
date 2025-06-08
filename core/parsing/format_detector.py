"""
Format detection utilities for LLM responses
"""

import re
import logging
from typing import List, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class ResponseFormat(Enum):
    """Detected response formats"""
    JSON = "json"
    REACT = "react"
    FUNCTION_CALL = "function_call"
    XML = "xml"
    MARKDOWN = "markdown"
    PLAIN_TEXT = "plain_text"
    MIXED = "mixed"


class FormatDetector:
    """Detects the format of LLM responses"""
    
    def __init__(self):
        self.format_patterns = {
            ResponseFormat.JSON: [
                r'```json',
                r'\{[^}]*"tool"[^}]*\}',
                r'\{[^}]*"action"[^}]*\}',
                r'\{[^}]*"thought"[^}]*\}',
            ],
            ResponseFormat.REACT: [
                r'(?:Thought|Reasoning):\s*',
                r'(?:Action|Tool):\s*',
                r'(?:Observation|Result):\s*',
            ],
            ResponseFormat.FUNCTION_CALL: [
                r'\w+\s*\([^)]*\)',
            ],
            ResponseFormat.XML: [
                r'<tool[^>]*>',
                r'<action[^>]*>',
                r'<thought[^>]*>',
            ],
            ResponseFormat.MARKDOWN: [
                r'\[tool:[^\]]+\]',
                r'\[action:[^\]]+\]',
            ],
        }
    
    def detect_format(self, response: str) -> ResponseFormat:
        """
        Detect the primary format of the response
        
        Args:
            response: LLM response text
            
        Returns:
            Detected format type
        """
        if not response or not response.strip():
            return ResponseFormat.PLAIN_TEXT
        
        format_scores = self._calculate_format_scores(response)
        
        # Get the format with highest score
        if format_scores:
            best_format = max(format_scores, key=format_scores.get)
            
            # Check if multiple formats have high scores
            high_score_formats = [
                fmt for fmt, score in format_scores.items() 
                if score > 0.3 and score >= format_scores[best_format] * 0.7
            ]
            
            if len(high_score_formats) > 1:
                return ResponseFormat.MIXED
            
            return best_format
        
        return ResponseFormat.PLAIN_TEXT
    
    def detect_all_formats(self, response: str) -> List[ResponseFormat]:
        """
        Detect all formats present in the response
        
        Args:
            response: LLM response text
            
        Returns:
            List of detected formats
        """
        detected_formats = []
        
        for format_type, patterns in self.format_patterns.items():
            for pattern in patterns:
                if re.search(pattern, response, re.IGNORECASE | re.MULTILINE):
                    detected_formats.append(format_type)
                    break
        
        return detected_formats or [ResponseFormat.PLAIN_TEXT]
    
    def get_format_confidence(self, response: str, format_type: ResponseFormat) -> float:
        """
        Get confidence score for a specific format
        
        Args:
            response: LLM response text
            format_type: Format to check
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if format_type not in self.format_patterns:
            return 0.0
        
        patterns = self.format_patterns[format_type]
        matches = 0
        total_weight = 0
        
        for i, pattern in enumerate(patterns):
            # Give higher weight to more specific patterns (earlier in list)
            weight = 1.0 / (i + 1)
            total_weight += weight
            
            if re.search(pattern, response, re.IGNORECASE | re.MULTILINE):
                matches += weight
        
        return matches / total_weight if total_weight > 0 else 0.0
    
    def _calculate_format_scores(self, response: str) -> dict:
        """Calculate confidence scores for all formats"""
        scores = {}
        
        for format_type in self.format_patterns:
            score = self.get_format_confidence(response, format_type)
            if score > 0:
                scores[format_type] = score
        
        return scores
    
    def extract_format_regions(self, response: str) -> List[Tuple[int, int, ResponseFormat]]:
        """
        Extract regions of different formats in the response
        
        Returns:
            List of (start, end, format) tuples
        """
        regions = []
        
        # JSON regions
        for match in re.finditer(r'```json\s*(.*?)\s*```', response, re.DOTALL):
            regions.append((match.start(), match.end(), ResponseFormat.JSON))
        
        # ReAct sections
        react_pattern = r'((?:Thought|Action|Observation):\s*.*?)(?=\n(?:Thought|Action|Observation):|$)'
        for match in re.finditer(react_pattern, response, re.IGNORECASE | re.MULTILINE | re.DOTALL):
            regions.append((match.start(), match.end(), ResponseFormat.REACT))
        
        # Function calls
        func_pattern = r'\b\w+\s*\([^)]*\)'
        for match in re.finditer(func_pattern, response):
            # Check if it's not inside another region
            if not any(start <= match.start() < end for start, end, _ in regions):
                regions.append((match.start(), match.end(), ResponseFormat.FUNCTION_CALL))
        
        # Sort by start position
        regions.sort(key=lambda x: x[0])
        
        return regions