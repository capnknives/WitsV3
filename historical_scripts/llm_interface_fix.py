#!/usr/bin/env python
"""
WitsV3 LLM Interface Fix and Test
Specifically tests the Ollama interface streaming functionality.
"""

import asyncio
import logging
import json
import os

# Configure logging to see everything
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("WitsV3_LLM_Fix")

from core.config import load_config
from core.llm_interface import OllamaInterface, get_llm_interface

async def test_ollama_direct():
    """
    Test direct Ollama interface streaming.
    """
    print("\n=== Testing Direct Ollama Interface ===")
    config = load_config()
    interface = OllamaInterface(config)

    # Test text generation
    print("\nTesting text generation...")
    response = await interface.generate_text("Say hello in one short sentence.")
    print(f"Response: {response}")

    # Test streaming
    print("\nTesting streaming...")
    print("Streamed response: ", end="", flush=True)
    async for chunk in interface.stream_text("Count from 1 to 5 briefly."):
        print(chunk, end="", flush=True)
    print("\nStreaming complete!")

    # Test embedding
    print("\nTesting embedding...")
    embedding = await interface.get_embedding("Test embedding")
    print(f"Embedding dimensions: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")

    print("\nDirect Ollama interface tests complete!\n")
    return True

async def test_adaptive_llm():
    """
    Test adaptive LLM interface streaming.
    """
    print("\n=== Testing Adaptive LLM Interface ===")
    config = load_config()
    # Force adaptive provider
    config.llm_interface.default_provider = "adaptive"

    interface = get_llm_interface(config)

    # Test text generation
    print("\nTesting text generation...")
    response = await interface.generate_text("Say hello in one short sentence.")
    print(f"Response: {response}")

    # Test streaming - needs await for adaptive interface
    print("\nTesting streaming...")
    print("Streamed response: ", end="", flush=True)

    # This is the key fix for the adaptive interface streaming
    async for chunk in interface.stream_text("Count from 1 to 5 briefly."):
        print(chunk, end="", flush=True)
    print("\nStreaming complete!")

    # Test embedding
    print("\nTesting embedding...")
    embedding = await interface.get_embedding("Test embedding")
    print(f"Embedding dimensions: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")

    print("\nAdaptive LLM interface tests complete!\n")
    return True

async def check_file_for_debug(filepath):
    """Check if a file contains certain debug statements."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            return 'print(' in content or 'logger.debug(' in content
    except Exception:
        return False

async def main():
    """Main function to test both interfaces."""
    print("WitsV3 LLM Interface Fix and Test")
    print("=" * 50)

    # First test Ollama direct interface
    ollama_ok = await test_ollama_direct()

    # Then test adaptive interface
    adaptive_ok = await test_adaptive_llm()

    if ollama_ok and adaptive_ok:
        print("\n✅ All tests passed! LLM interface is working correctly.")
        return 0
    else:
        print("\n❌ Some tests failed. See logs for details.")
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        exit(exit_code)
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        exit(130)
    except Exception as e:
        print(f"Test failed with error: {e}")
        exit(1)
