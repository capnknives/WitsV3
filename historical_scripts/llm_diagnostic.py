#!/usr/bin/env python3
"""
WitsV3 LLM Interface Diagnostic
This script tests the LLM interface directly without the full system
"""

import asyncio
# Fix Unicode encoding issues
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

import sys
import codecs
# Set UTF-8 encoding for stdout/stderr
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

# Fix Unicode encoding issues
import sys
sys.stdout.reconfigure(encoding='utf-8')

import logging
import sys
import time
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime,
    encoding='utf-8'
)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("WitsV3.LLMDiagnostic")

# Add project root to path if needed
current_dir = Path(__file__).parent
if current_dir not in sys.path:
    sys.path.insert(0, str(current_dir))

async def test_basic_llm():
    """Test basic LLM functionality with OllamaInterface."""
    from core.config import load_config
    from core.llm_interface import OllamaInterface

    logger.info("Testing basic LLM functionality with OllamaInterface...")

    # Load configuration
    config = load_config()
    # Access config values safely
    model = getattr(config, "model", "unknown")
    if hasattr(config, "llm"):
        model = getattr(config.llm, "model", model)
    logger.info(f"Loaded configuration with model: {model}")

    # Initialize LLM interface
    llm = OllamaInterface(config=config)

    # Get host and port safely
    ollama_host = "localhost"
    ollama_port = "11434"
    if hasattr(config, "llm"):
        ollama_host = getattr(config.llm, "ollama_host", ollama_host)
        ollama_port = getattr(config.llm, "ollama_port", ollama_port)
    logger.info(f"Initialized OllamaInterface with host: {ollama_host}:{ollama_port}")

    # Test simple generation
    try:
        logger.info("Testing simple generation...")
        start_time = time.time()
        response = await llm.generate_text("Say hello in one short sentence.")
        elapsed = time.time() - start_time
        logger.info(f"Response received in {elapsed:.2f}s: {response}")
    except Exception as e:
        logger.error(f"Simple generation failed: {e}")
        return False

    # Test streaming
    try:
        logger.info("Testing streaming...")
        start_time = time.time()
        chunks = []

        async for chunk in llm.stream_text("Count from 1 to 5."):
            chunks.append(chunk)
            logger.info(f"Received chunk: {chunk}")

        elapsed = time.time() - start_time
        logger.info(f"Streaming completed in {elapsed:.2f}s")
        logger.info(f"Received {len(chunks)} chunks")
        logger.info(f"Full response: {''.join(chunks)}")
    except Exception as e:
        logger.error(f"Streaming failed: {e}")
        return False

    logger.info("Basic LLM tests passed!")
    return True

async def test_adaptive_llm():
    """Test adaptive LLM functionality."""
    try:
        from core.config import load_config

        # Try to import from different possible locations
        try:
            from core.adaptive.adaptive_llm_interface import AdaptiveLLMInterface
        except ImportError:
            try:
                from core.adaptive_llm_interface import AdaptiveLLMInterface
            except ImportError:
                logger.warning("AdaptiveLLMInterface not found in expected locations")
                return None

        logger.info("Testing adaptive LLM functionality...")

        # Load configuration
        config = load_config()

        # Initialize LLM interface
        llm = AdaptiveLLMInterface(config=config)
        logger.info(f"Initialized AdaptiveLLMInterface")

        # Test simple generation
        logger.info("Testing simple generation with adaptive LLM...")
        start_time = time.time()
        response = await llm.generate_text("Say hello in one short sentence.")
        elapsed = time.time() - start_time
        logger.info(f"Response received in {elapsed:.2f}s: {response}")

        # Test streaming
        logger.info("Testing streaming with adaptive LLM...")
        start_time = time.time()
        chunks = []

        async for chunk in llm.stream_text("Count from 1 to 5."):
            chunks.append(chunk)
            logger.info(f"Received chunk: {chunk}")

        elapsed = time.time() - start_time
        logger.info(f"Streaming completed in {elapsed:.2f}s")
        logger.info(f"Received {len(chunks)} chunks")
        logger.info(f"Full response: {''.join(chunks)}")

        logger.info("Adaptive LLM tests passed!")
        return True
    except ImportError as e:
        logger.warning(f"Adaptive LLM module not available: {e}")
        return None
    except Exception as e:
        logger.error(f"Adaptive LLM tests failed: {e}")
        return False

async def main():
    """Main entry point."""
    logger.info("WitsV3 LLM Interface Diagnostic")
    logger.info("=" * 50)

    basic_result = await test_basic_llm()

    if not basic_result:
        logger.error("❌ Basic LLM tests failed - check Ollama is running and configured correctly")
        return 1

    # Test adaptive LLM if basic tests pass
    adaptive_result = await test_adaptive_llm()

    if adaptive_result is None:
        logger.info("ℹ️ Adaptive LLM tests skipped - module not available")
    elif adaptive_result:
        logger.info("✅ All LLM interface tests passed!")
    else:
        logger.error("❌ Adaptive LLM tests failed")
        return 1

    return 0

if __name__ == "__main__":
    # Set environment variable to indicate diagnostic mode
    os.environ['WITSV3_DIAGNOSTIC_MODE'] = '1'

    # Run the diagnostic
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
