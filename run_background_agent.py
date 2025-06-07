"""
Script to run the WitsV3 background agent
"""

import os
import asyncio
from agents.background_agent import main

if __name__ == "__main__":
    # Set environment variables
    os.environ["WITSV3_BACKGROUND_MODE"] = "true"
    os.environ["CURSOR_INTEGRATION"] = "true"
    
    # Run the agent
    asyncio.run(main()) 