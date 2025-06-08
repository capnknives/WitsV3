#!/usr/bin/env python3
"""
Script to run the WitsV3 background agent
"""

import os
import asyncio
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.absolute()
os.environ["PYTHONPATH"] = str(project_root)

# Set environment variables
os.environ["WITSV3_BACKGROUND_MODE"] = "true"
os.environ["CURSOR_INTEGRATION"] = "true"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/background_agent.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("WitsV3.BackgroundAgent")

async def main():
    """Main entry point for background agent"""
    try:
        from agents.background_agent import main as run_agent
        await run_agent()
    except KeyboardInterrupt:
        logger.info("Background agent stopped by user")
    except Exception as e:
        logger.error(f"Error running background agent: {e}")
        raise

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Run the agent
    asyncio.run(main())
