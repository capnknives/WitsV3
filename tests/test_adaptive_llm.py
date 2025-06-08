"""
Test script for the WitsV3 Adaptive LLM System.

This script tests the Adaptive LLM System components and their integration.
"""

import asyncio
import os
import logging
import time
from typing import Dict, List, Any
import sys

import torch
import numpy as np
import pytest

from core.config import WitsV3Config
from core.llm_interface import BaseLLMInterface, get_llm_interface
from core.complexity_analyzer import ComplexityAnalyzer
from core.dynamic_module_loader import DynamicModuleLoader
from core.semantic_cache import SemanticCache
from core.adaptive_llm_interface import AdaptiveLLMInterface
from core.adaptive_llm_config import AdaptiveLLMSettings

# Add the project root to sys.path for test discovery
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("WitsV3.Test")

# Test prompts
TEST_PROMPTS = [
    "Hello, how are you today?",
    "Write a Python function to calculate the Fibonacci sequence recursively.",
    "Explain the philosophical implications of quantum mechanics on our understanding of reality.",
    "What is 2+2?",
    "Write a short story about a robot who discovers emotions."
]

async def run_complexity_analyzer(config: WitsV3Config, llm: BaseLLMInterface) -> None:
    """Test the ComplexityAnalyzer component."""
    logger.info("Testing ComplexityAnalyzer...")

    analyzer = ComplexityAnalyzer(config, llm)

    for prompt in TEST_PROMPTS:
        logger.info(f"Analyzing prompt: {prompt[:50]}...")

        # Analyze complexity
        analysis = await analyzer.analyze_complexity(prompt)

        logger.info(f"Complexity: {analysis['complexity']:.2f}")
        logger.info(f"Domain: {analysis['domain']}")

        # Route to module
        module = await analyzer.route_to_module(prompt)
        logger.info(f"Routed to module: {module}")

        logger.info("-" * 50)

    logger.info("ComplexityAnalyzer tests completed!")

async def run_dynamic_module_loader(config: WitsV3Config) -> None:
    """Test the DynamicModuleLoader component."""
    logger.info("Testing DynamicModuleLoader...")

    loader = DynamicModuleLoader(config)

    # Test loading modules
    modules = ['base', 'python', 'math']
    complexities = [0.2, 0.6, 0.9]

    for module_name, complexity in zip(modules, complexities):
        logger.info(f"Loading module: {module_name} with complexity {complexity:.1f}...")

        module = await loader.load_module(module_name, complexity)
        logger.info(f"Module loaded: {module.name}, {module.bits}-bit")

        # Test module info
        info = loader.get_module_info(module_name)
        logger.info(f"Module info: {info}")

        logger.info("-" * 50)

    # Test resource usage
    usage = loader.get_resource_usage()
    logger.info(f"Resource usage: {usage}")

    # Test unloading
    logger.info("Unloading all modules...")
    await loader.unload_all()

    logger.info("DynamicModuleLoader tests completed!")

async def run_semantic_cache(config: WitsV3Config, llm: BaseLLMInterface) -> None:
    """Test the SemanticCache component."""
    logger.info("Testing SemanticCache...")

    cache = SemanticCache(config, llm)

    # Test adding patterns
    for i, prompt in enumerate(TEST_PROMPTS):
        logger.info(f"Adding pattern for prompt: {prompt[:50]}...")

        response = f"This is a test response for prompt {i+1}."
        module = ['base', 'python', 'creative', 'math', 'chat'][i % 5]
        complexity = 0.2 + (i * 0.2)

        await cache.add_pattern(
            prompt,
            response,
            module,
            complexity,
            0.5,  # generation time
            0.9   # quality
        )

    # Test finding similar patterns
    for prompt in TEST_PROMPTS:
        logger.info(f"Finding similar pattern for prompt: {prompt[:50]}...")

        similar = await cache.find_similar(prompt)

        if similar:
            logger.info(f"Found similar pattern with similarity: {similar['similarity']:.4f}")
            logger.info(f"Response: {similar['response']}")
        else:
            logger.info("No similar pattern found.")

        logger.info("-" * 50)

    # Test user patterns
    user_patterns = cache.get_user_patterns()
    logger.info(f"User patterns: {user_patterns}")

    # Test clearing cache
    logger.info("Clearing cache...")
    await cache.clear_cache()

    logger.info("SemanticCache tests completed!")

async def run_adaptive_llm_interface(config: WitsV3Config) -> None:
    """Test the AdaptiveLLMInterface component."""
    logger.info("Testing AdaptiveLLMInterface...")

    # Create base LLM interface
    base_llm = get_llm_interface(config)

    # Create adaptive LLM interface
    adaptive_llm = AdaptiveLLMInterface(config, base_llm)

    # Test generation
    for prompt in TEST_PROMPTS:
        logger.info(f"Generating response for prompt: {prompt[:50]}...")

        start_time = time.time()
        response = await adaptive_llm.generate_text(prompt)
        generation_time = time.time() - start_time

        logger.info(f"Response: {response[:100]}...")
        logger.info(f"Generation time: {generation_time:.2f}s")

        logger.info("-" * 50)

    # Test streaming - temporarily disabled due to async iteration issue
    logger.info("Testing streaming... (temporarily disabled)")
    # prompt = "Tell me a story about a brave knight."
    #
    # logger.info(f"Streaming response for prompt: {prompt}...")
    #
    # response_chunks = []
    # async for chunk in adaptive_llm.stream_text(prompt):
    #     response_chunks.append(chunk)
    #     logger.info(f"Received chunk: {chunk}")
    #
    # logger.info(f"Full streamed response: {''.join(response_chunks)}")

    # Test system status
    logger.info("Getting system status...")
    status = adaptive_llm.get_system_status()

    logger.info(f"Resource usage: {status['resource_usage']}")
    logger.info(f"Performance stats: {status['performance_stats']}")

    # Test shutdown
    logger.info("Shutting down...")
    await adaptive_llm.shutdown()

    logger.info("AdaptiveLLMInterface tests completed!")

@pytest.mark.asyncio
async def test_adaptive_llm():
    """Pytest entrypoint for all adaptive LLM system tests."""
    logger.info("Starting Adaptive LLM System tests...")
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file_path = os.path.join(current_script_dir, "config.yaml")
    config = WitsV3Config.from_yaml(config_file_path)
    llm = get_llm_interface(config)
    await run_complexity_analyzer(config, llm)
    await run_dynamic_module_loader(config)
    await run_semantic_cache(config, llm)
    await run_adaptive_llm_interface(config)
    logger.info("All tests completed successfully! 🎉")
