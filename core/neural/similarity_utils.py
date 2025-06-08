"""
Similarity calculation utilities for neural memory
"""

import logging
from typing import List, Optional, Tuple
import math

logger = logging.getLogger(__name__)


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors with robust error handling.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity between 0.0 and 1.0
    """
    try:
        if not vec1 or not vec2:
            return 0.0
            
        if len(vec1) != len(vec2):
            logger.debug(f"Vector length mismatch: {len(vec1)} vs {len(vec2)}")
            return 0.0
        
        # Calculate dot product and magnitudes
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(a * a for a in vec2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        similarity = dot_product / (magnitude1 * magnitude2)
        return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
        
    except Exception as e:
        logger.debug(f"Error calculating cosine similarity: {e}")
        return 0.0


class SimilarityCalculator:
    """Advanced similarity calculations for neural memory"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity (delegates to function)"""
        return cosine_similarity(vec1, vec2)
    
    def euclidean_distance(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate Euclidean distance between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Euclidean distance (0 means identical)
        """
        try:
            if not vec1 or not vec2:
                return float('inf')
            
            if len(vec1) != len(vec2):
                return float('inf')
            
            squared_diff = sum((a - b) ** 2 for a, b in zip(vec1, vec2))
            return math.sqrt(squared_diff)
            
        except Exception as e:
            self.logger.debug(f"Error calculating Euclidean distance: {e}")
            return float('inf')
    
    def euclidean_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Convert Euclidean distance to similarity score.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        distance = self.euclidean_distance(vec1, vec2)
        if distance == float('inf'):
            return 0.0
        
        # Convert distance to similarity using exponential decay
        # Smaller distances yield higher similarity
        return math.exp(-distance)
    
    def manhattan_distance(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate Manhattan (L1) distance between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Manhattan distance
        """
        try:
            if not vec1 or not vec2:
                return float('inf')
            
            if len(vec1) != len(vec2):
                return float('inf')
            
            return sum(abs(a - b) for a, b in zip(vec1, vec2))
            
        except Exception as e:
            self.logger.debug(f"Error calculating Manhattan distance: {e}")
            return float('inf')
    
    def dot_product(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate dot product between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Dot product value
        """
        try:
            if not vec1 or not vec2:
                return 0.0
            
            if len(vec1) != len(vec2):
                return 0.0
            
            return sum(a * b for a, b in zip(vec1, vec2))
            
        except Exception as e:
            self.logger.debug(f"Error calculating dot product: {e}")
            return 0.0
    
    def find_most_similar(
        self,
        query_vector: List[float],
        candidate_vectors: List[Tuple[str, List[float]]],
        method: str = "cosine",
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find the most similar vectors to a query vector.
        
        Args:
            query_vector: Query vector
            candidate_vectors: List of (id, vector) tuples
            method: Similarity method ("cosine", "euclidean", "manhattan")
            top_k: Number of top results to return
            
        Returns:
            List of (id, similarity_score) tuples sorted by similarity
        """
        try:
            if not query_vector or not candidate_vectors:
                return []
            
            # Calculate similarities
            similarities = []
            
            for candidate_id, candidate_vec in candidate_vectors:
                if method == "cosine":
                    score = self.cosine_similarity(query_vector, candidate_vec)
                elif method == "euclidean":
                    score = self.euclidean_similarity(query_vector, candidate_vec)
                elif method == "manhattan":
                    # Convert distance to similarity
                    distance = self.manhattan_distance(query_vector, candidate_vec)
                    score = 1.0 / (1.0 + distance) if distance != float('inf') else 0.0
                else:
                    # Default to cosine
                    score = self.cosine_similarity(query_vector, candidate_vec)
                
                if score > 0:
                    similarities.append((candidate_id, score))
            
            # Sort by similarity (descending) and return top k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            self.logger.error(f"Error finding similar vectors: {e}")
            return []