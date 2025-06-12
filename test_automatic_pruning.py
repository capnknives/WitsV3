#!/usr/bin/env python3

import asyncio
import os
from core.config import load_config
from core.memory_manager import MemoryManager
from core.llm_interface import get_llm_interface

async def test_automatic_pruning():
    """Test that automatic pruning triggers when adding segments."""
    print("🔧 Testing Automatic Memory Pruning...")

    # Load config
    config = load_config('config.yaml')

    # Set very aggressive pruning settings for testing
    config.memory_manager.enable_auto_pruning = True
    config.memory_manager.max_memory_size_mb = 2  # Very small limit
    config.memory_manager.pruning_threshold = 0.8  # Prune at 80% of max size
    config.memory_manager.max_memory_segments = 30  # Also limit by count

    print(f"Max size: {config.memory_manager.max_memory_size_mb} MB")
    print(f"Max segments: {config.memory_manager.max_memory_segments}")

    # Initialize LLM interface and memory manager
    llm_interface = get_llm_interface(config)
    memory_manager = MemoryManager(config, llm_interface)
    await memory_manager.initialize()

    # Check current state
    initial_count = len(memory_manager.backend.segments)
    initial_size = await memory_manager.backend._get_memory_size_mb()
    print(f"Initial: {initial_count} segments, {initial_size:.2f} MB")

    # Add some new segments that should trigger automatic pruning
    for i in range(5):
        await memory_manager.add_memory(
            type="TEST_AUTO_PRUNE",
            source="test_script",
            content_text=f"This is test segment {i} with some content to see if automatic pruning works. " * 20,  # Make it large
            importance=0.9,  # High importance to see if it gets kept
            metadata={"test_batch": i}
        )

        # Check if pruning happened
        current_count = len(memory_manager.backend.segments)
        current_size = await memory_manager.backend._get_memory_size_mb()
        print(f"Added segment {i}: {current_count} segments, {current_size:.2f} MB")

    final_count = len(memory_manager.backend.segments)
    final_size = await memory_manager.backend._get_memory_size_mb()

    print(f"\nFinal result:")
    print(f"  Segments: {initial_count} → {final_count} (change: {final_count - initial_count})")
    print(f"  Size: {initial_size:.2f} MB → {final_size:.2f} MB")

    if final_count <= config.memory_manager.max_memory_segments:
        print("✅ Automatic count-based pruning worked!")
    else:
        print("❌ Count-based pruning may not have triggered")

    if final_size <= config.memory_manager.max_memory_size_mb:
        print("✅ Automatic size-based pruning worked!")
    else:
        print("❌ Size-based pruning may not have triggered")

if __name__ == "__main__":
    asyncio.run(test_automatic_pruning())
