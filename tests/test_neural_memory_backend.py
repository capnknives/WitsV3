#!/usr/bin/env python3
"""
Test script to verify the neural memory backend is working correctly
and no longer generating None ID errors.
"""

import asyncio
import sys
import os
import logging
from datetime import datetime, timezone

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.config import WitsV3Config
from core.memory_manager import MemorySegment, MemorySegmentContent
from core.neural_memory_backend import NeuralMemoryBackend
from core.llm_interface import OllamaInterface

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_neural_memory_backend():
    """Test the neural memory backend to ensure it's working correctly."""
    
    try:
        print("🧠 Testing Neural Memory Backend...")
        
        # Load configuration
        config = WitsV3Config()
        llm_interface = OllamaInterface(config)
        
        # Create neural memory backend
        backend = NeuralMemoryBackend(config, llm_interface)
        await backend.initialize()
        
        print("✅ Neural memory backend initialized successfully")        # Test 1: Add a few memory segments
        test_segments = [
            MemorySegment(
                content=MemorySegmentContent(text="This is a test memory about machine learning"),
                timestamp=datetime.now(timezone.utc),
                source="test",
                type="TEST_MEMORY"
            ),
            MemorySegment(
                content=MemorySegmentContent(text="Another memory about neural networks"),
                timestamp=datetime.now(timezone.utc),
                source="test",
                type="TEST_MEMORY"
            ),
            MemorySegment(
                content=MemorySegmentContent(text="A third memory about artificial intelligence"),
                timestamp=datetime.now(timezone.utc),
                source="test",
                type="TEST_MEMORY"
            )
        ]
        
        added_ids = []
        for i, segment in enumerate(test_segments):
            segment_id = await backend.add_segment(segment)
            added_ids.append(segment_id)
            print(f"✅ Added test segment {i+1}: {segment_id}")
            
            # Verify the ID is valid (not None)
            if not segment_id or segment_id == "None" or segment_id == "null":
                raise ValueError(f"Invalid segment ID returned: {segment_id}")
        
        print(f"✅ All {len(added_ids)} segments added with valid IDs")
        
        # Test 2: Retrieve segments
        for segment_id in added_ids:
            retrieved = await backend.get_segment(segment_id)
            if not retrieved:
                raise ValueError(f"Failed to retrieve segment: {segment_id}")
            print(f"✅ Successfully retrieved segment: {segment_id}")
        
        # Test 3: Search segments
        search_results = await backend.search_segments("machine learning", limit=5)
        print(f"✅ Search returned {len(search_results)} results")
        
        # Test 4: Get recent segments
        recent_segments = await backend.get_recent_segments(limit=5)
        print(f"✅ Recent segments: {len(recent_segments)} found")
        
        # Test 5: Check neural web state
        if hasattr(backend, 'neural_web'):
            concept_count = len(backend.neural_web.concepts)
            connection_count = len(backend.neural_web.connections)
            print(f"✅ Neural web has {concept_count} concepts and {connection_count} connections")
            
            # Verify no None concepts
            for concept_id, concept in backend.neural_web.concepts.items():
                if concept_id is None or concept_id == "None" or concept_id == "null":
                    raise ValueError(f"Found None concept ID in neural web: {concept_id}")
                if concept.id is None or concept.id == "None" or concept.id == "null":
                    raise ValueError(f"Found concept with None ID: {concept.id}")
            
            print("✅ All neural web concepts have valid IDs")
            
            # Verify no None connections
            for (source_id, target_id), connection in backend.neural_web.connections.items():
                if source_id is None or source_id == "None" or source_id == "null":
                    raise ValueError(f"Found None source ID in connection: {source_id}")
                if target_id is None or target_id == "None" or target_id == "null":
                    raise ValueError(f"Found None target ID in connection: {target_id}")
                if connection.source_id is None or connection.source_id == "None":
                    raise ValueError(f"Found connection with None source_id: {connection.source_id}")
                if connection.target_id is None or connection.target_id == "None":
                    raise ValueError(f"Found connection with None target_id: {connection.target_id}")
            
            print("✅ All neural web connections have valid IDs")
        
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Neural Memory Backend is working correctly")
        print("✅ No None ID errors detected")
        print("✅ All segments have valid, non-None IDs")
        print("✅ Neural web concepts and connections are properly formed")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        logger.error(f"Neural memory backend test failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = asyncio.run(test_neural_memory_backend())
    if not success:
        sys.exit(1)
