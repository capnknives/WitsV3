#!/usr/bin/env python3
"""
Script to run the WitsV3 background agent
"""

import os
import asyncio
import logging
from pathlib import Path

from core.runtime_paths import ensure_runtime_layout, logs_dir

# Add project root to Python path
project_root = Path(__file__).parent.absolute()
os.environ["PYTHONPATH"] = str(project_root)

# Set environment variables
os.environ["WITSV3_BACKGROUND_MODE"] = "true"
os.environ["CURSOR_INTEGRATION"] = "true"

ensure_runtime_layout()
logs_dir().mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(logs_dir() / "background_agent.log")),
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
    ensure_runtime_layout()
    asyncio.run(main())
