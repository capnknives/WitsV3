"""
Relationship and concept type analysis for neural memory
"""

import logging
from typing import Optional

from ..memory_manager import MemorySegment

logger = logging.getLogger(__name__)


class RelationshipAnalyzer:
    """Analyzes relationships between concepts and determines concept types"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def determine_concept_type(self, segment: MemorySegment) -> str:
        """
        Determine the type of concept based on segment content.
        
        Args:
            segment: Memory segment to analyze
            
        Returns:
            Concept type string
        """
        try:
            if not segment or not segment.content:
                return "unknown"
                
            content_text = segment.content.text or ""
            content_lower = content_text.lower()
            
            # Simple heuristics for concept type determination
            type_indicators = {
                "question": ["question", "?", "how", "what", "why", "when", "where"],
                "task": ["task", "todo", "action", "do", "complete", "finish"],
                "fact": ["fact", "is", "are", "definition", "means"],
                "thought": ["thought", "think", "believe", "opinion"],
                "problem": ["error", "problem", "issue", "bug"],
            }
            
            for concept_type, indicators in type_indicators.items():
                if any(word in content_lower for word in indicators):
                    return concept_type
            
            return "information"
                
        except Exception as e:
            self.logger.debug(f"Error determining concept type: {e}")
            return "unknown"
    
    def determine_relationship_type(self, content1: str, content2: str) -> str:
        """
        Determine the type of relationship between two pieces of content.
        
        Args:
            content1: First content to compare
            content2: Second content to compare
            
        Returns:
            Relationship type string
        """
        try:
            if not content1 or not content2:
                return "related"
                
            c1_lower = content1.lower()
            c2_lower = content2.lower()
            
            # Relationship type indicators
            relationship_indicators = {
                "causal": ["because", "caused", "due to", "result", "leads to", "triggers"],
                "temporal": ["before", "after", "then", "next", "previously", "following"],
                "similarity": ["similar", "like", "same", "resembles", "comparable"],
                "contrast": ["different", "unlike", "however", "but", "opposite"],
                "hierarchical": ["part of", "contains", "includes", "subset", "belongs to"],
                "dependency": ["requires", "depends on", "needs", "prerequisite"],
                "example": ["example", "instance", "such as", "for example", "e.g."],
            }
            
            # Check each relationship type
            for rel_type, indicators in relationship_indicators.items():
                if any(word in c1_lower for word in indicators) or \
                   any(word in c2_lower for word in indicators):
                    return rel_type
            
            # Default to semantic relationship
            return "semantic"
            
        except Exception as e:
            self.logger.debug(f"Error determining relationship type: {e}")
            return "related"
    
    def calculate_content_similarity(self, content1: str, content2: str) -> float:
        """
        Calculate simple content-based similarity between two texts.
        
        Args:
            content1: First content
            content2: Second content
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        try:
            if not content1 or not content2:
                return 0.0
            
            # Simple word-based similarity (Jaccard index)
            words1 = set(content1.lower().split())
            words2 = set(content2.lower().split())
            
            if not words1 or not words2:
                return 0.0
            
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            self.logger.debug(f"Error calculating content similarity: {e}")
            return 0.0
    
    def is_strong_relationship(
        self, 
        content1: str, 
        content2: str,
        threshold: float = 0.3
    ) -> bool:
        """
        Determine if two contents have a strong relationship.
        
        Args:
            content1: First content
            content2: Second content
            threshold: Minimum similarity threshold
            
        Returns:
            True if relationship is strong
        """
        similarity = self.calculate_content_similarity(content1, content2)
        rel_type = self.determine_relationship_type(content1, content2)
        
        # Some relationship types are inherently stronger
        strong_types = {"causal", "dependency", "hierarchical"}
        
        if rel_type in strong_types:
            # Lower threshold for strong relationship types
            return similarity >= threshold * 0.7
        
        return similarity >= threshold