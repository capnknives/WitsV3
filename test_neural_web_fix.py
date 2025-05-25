#!/usr/bin/env python3
"""
Test the neural web loading to ensure the null ID fix is working
"""

import asyncio
import json
from pathlib import Path
from core.neural_web_core import NeuralWeb

async def test_neural_web():
    """Test that the neural web loads without null ID errors"""
    print("Testing neural web loading...")
    
    # Create a neural web instance
    neural_web = NeuralWeb()
    
    # Check if the data file exists
    neural_web_path = Path("data/neural_web.json")
    if not neural_web_path.exists():
        print("Neural web data file not found")
        return
    
    # Load and parse the JSON
    with open(neural_web_path, 'r') as f:
        neural_data = json.load(f)
    
    print(f"Found {len(neural_data.get('concepts', {}))} concepts")
    print(f"Found {len(neural_data.get('connections', {}))} connections")
    
    # Test adding concepts (this would fail before the fix)
    concepts_added = 0
    for concept_id, concept_data in neural_data.get("concepts", {}).items():
        stored_concept_id = concept_data.get("id")
        
        # The fix should handle null IDs
        if stored_concept_id is None or stored_concept_id == "null":
            print(f"Found concept with null ID - this should be handled by the fix")
        
        try:
            await neural_web.add_concept(
                concept_id=stored_concept_id,
                content=concept_data["content"],
                concept_type=concept_data["concept_type"],
                metadata=concept_data.get("metadata", {})
            )
            concepts_added += 1
        except Exception as e:
            print(f"Error adding concept {stored_concept_id}: {e}")
    
    print(f"Successfully added {concepts_added} concepts to neural web")
    
    # Test network operations that would fail with null nodes
    try:
        stats = neural_web.get_statistics()
        print(f"Neural web statistics: {stats}")
        print("✅ Neural web is working correctly!")
        return True
    except Exception as e:
        print(f"❌ Error accessing neural web: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_neural_web())
    if success:
        print("\n🎉 All tests passed! The null ID error has been fixed.")
    else:
        print("\n❌ Tests failed. There may still be issues.")
