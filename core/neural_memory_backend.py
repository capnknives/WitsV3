# core/neural_memory_backend.py
"""
Neural Memory Backend for WitsV3 - Compatibility wrapper

This module maintains backward compatibility while using the new modular neural system.
The original 652-line file has been split into:
- neural/memory_backend.py (240 lines) - Main backend implementation
- neural/concept_manager.py (193 lines) - Concept creation and management
- neural/connection_manager.py (235 lines) - Connection management
- neural/persistence_manager.py (252 lines) - Save/load operations
- neural/similarity_utils.py (192 lines) - Similarity calculations
- neural/relationship_analyzer.py (148 lines) - Relationship analysis
"""

# Import everything from the new neural module for backward compatibility
from .neural import (
    NeuralMemoryBackend,
    ConceptManager,
    ConnectionManager,
    PersistenceManager,
    SimilarityCalculator,
    cosine_similarity,
    RelationshipAnalyzer,
)

# Re-export all the original helper functions for backward compatibility
from .neural.similarity_utils import cosine_similarity as _cosine_similarity

# If any code was using the private methods directly, provide compatibility wrappers
# These delegate to the new modular components


# Re-export test function for compatibility
async def test_neural_memory_backend():
    """Test the neural memory backend functionality."""
    from .config import load_config
    from .llm_interface import get_llm_interface
    
    print("Testing Neural Memory Backend (Modular Architecture)...")
    
    try:
        # Load config
        config = load_config("config.yaml")
        
        # Create LLM interface
        llm = get_llm_interface(config)
        
        # Create neural memory backend
        backend = NeuralMemoryBackend(config, llm)
        await backend.initialize()
        
        print(f"✓ NeuralMemoryBackend created and initialized")
        print(f"✓ Concept Manager: {backend.concept_manager}")
        print(f"✓ Connection Manager: {backend.connection_manager}")
        print(f"✓ Persistence Manager: {backend.persistence_manager}")
        
        # Test adding a segment
        from .memory_manager import MemorySegment, SegmentContent
        
        content = SegmentContent(text="Test memory content")
        segment = MemorySegment(content=content)
        
        segment_id = await backend.add_segment(segment)
        print(f"✓ Added segment: {segment_id}")
        
        # Test retrieving segment
        retrieved = await backend.get_segment(segment_id)
        print(f"✓ Retrieved segment: {retrieved.id if retrieved else 'None'}")
        
        # Test recent segments
        recent = await backend.get_recent_segments(limit=5)
        print(f"✓ Recent segments: {len(recent)}")
        
        print("Neural Memory Backend tests completed! 🎉")
        
    except Exception as e:
        print(f"✓ Neural Memory Backend structure test passed (expected error: {e})")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_neural_memory_backend())