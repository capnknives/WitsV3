#!/usr/bin/env python3

import asyncio
import os
from core.config import load_config
from core.memory_manager import MemoryManager
from core.llm_interface import get_llm_interface

async def test_memory_pruning():
    """Test the enhanced memory pruning functionality."""
    print("🔧 Testing Enhanced Memory Pruning...")

    # Load config
    config = load_config('config.yaml')

    # Override settings for testing
    config.memory_manager.enable_auto_pruning = True
    config.memory_manager.max_memory_size_mb = 5  # Very small limit to trigger pruning
    config.memory_manager.pruning_threshold = 0.8  # Prune at 80% of max size
    config.memory_manager.max_memory_segments = 50  # Also limit by count

    print(f"Config: max_memory_size_mb={config.memory_manager.max_memory_size_mb}")
    print(f"Config: pruning_threshold={config.memory_manager.pruning_threshold}")
    print(f"Config: enable_auto_pruning={config.memory_manager.enable_auto_pruning}")

    # Initialize LLM interface and memory manager
    llm_interface = get_llm_interface(config)
    memory_manager = MemoryManager(config, llm_interface)
    await memory_manager.initialize()

    # Check current memory size
    if hasattr(memory_manager.backend, '_get_memory_size_mb'):
        current_size = await memory_manager.backend._get_memory_size_mb()
        print(f"Current memory size: {current_size:.2f} MB")

    # Check if auto-pruning would trigger
    if hasattr(memory_manager.backend, '_should_prune_by_size'):
        should_prune = await memory_manager.backend._should_prune_by_size()
        print(f"Should prune by size: {should_prune}")

    # Manual test of pruning
    segment_count_before = len(memory_manager.backend.segments)
    print(f"Segments before pruning: {segment_count_before}")

    # Trigger manual pruning
    await memory_manager.backend.prune_memory()

    segment_count_after = len(memory_manager.backend.segments)
    print(f"Segments after pruning: {segment_count_after}")
    print(f"Segments removed: {segment_count_before - segment_count_after}")

    # Check size after pruning
    if hasattr(memory_manager.backend, '_get_memory_size_mb'):
        size_after = await memory_manager.backend._get_memory_size_mb()
        print(f"Memory size after pruning: {size_after:.2f} MB")

if __name__ == "__main__":
    asyncio.run(test_memory_pruning())
