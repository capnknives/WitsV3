"""
WitsV3 Neural Memory Module

This module provides neural web-powered memory backend with graph-based 
knowledge representation and advanced memory management.
"""

from .memory_backend import NeuralMemoryBackend
from .concept_manager import ConceptManager
from .connection_manager import ConnectionManager
from .persistence_manager import PersistenceManager
from .similarity_utils import SimilarityCalculator, cosine_similarity
from .relationship_analyzer import RelationshipAnalyzer

__all__ = [
    # Main backend
    'NeuralMemoryBackend',
    
    # Managers
    'ConceptManager',
    'ConnectionManager', 
    'PersistenceManager',
    
    # Utilities
    'SimilarityCalculator',
    'cosine_similarity',
    'RelationshipAnalyzer',
]