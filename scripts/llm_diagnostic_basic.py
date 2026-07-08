#!/usr/bin/env python3
"""
WitsV3 Basic LLM Interface Diagnostic
Tests only the core OllamaInterface without adaptive LLM
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
    try:
        from core.config import load_config
        from core.llm_interface import OllamaInterface

        logger.info("Testing basic LLM functionality with OllamaInterface...")

        # Load configuration
        config = load_config()
        logger.info(f"Loaded configuration")

        # Initialize LLM interface
        llm = OllamaInterface(config=config)
        logger.info(f"Initialized OllamaInterface")

        # Test simple generation
        logger.info("Testing simple generation...")
        start_time = time.time()
        response = await llm.generate_text("Say hello in one short sentence.")
        elapsed = time.time() - start_time
        logger.info(f"Response received in {elapsed:.2f}s: {response}")

        # Test streaming
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

        logger.info("Basic LLM tests passed!")
        return True

    except Exception as e:
        logger.error(f"LLM test failed: {e}")
        return False

async def main():
    """Main entry point."""
    logger.info("WitsV3 Basic LLM Interface Diagnostic")
    logger.info("=" * 50)

    success = await test_basic_llm()

    if success:
        logger.info("✅ LLM interface tests passed - system is functional!")
        return 0
    else:
        logger.error("❌ LLM interface tests failed - check Ollama configuration")
        return 1

if __name__ == "__main__":
    # Set environment variable to indicate diagnostic mode
    os.environ['WITSV3_DIAGNOSTIC_MODE'] = '1'

    # Run the diagnostic
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
