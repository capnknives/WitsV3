"""
Main entry point for WitsV3 Matrix GUI Application.

This module serves as the entry point for the WitsV3 Matrix GUI application.
It initializes the application and starts the main window.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add parent directory to path to import WitsV3 modules
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Print Python version for debugging
print(f"Python version: {sys.version}")

# Check for Qt implementations
print("Available Qt implementations:")
for qt_impl in ["PyQt6", "PyQt5", "PySide6", "PySide2"]:
    try:
        __import__(qt_impl)
        print(f"- {qt_impl} is available")
    except ImportError:
        print(f"- {qt_impl} is NOT available")

# Try to import qasync
try:
    import qasync
    print("qasync imported successfully")
except ImportError as e:
    print(f"Error importing qasync: {e}")
    print("\nTo fix this issue, please run: pip install qasync")
    print("If you've already installed qasync, make sure it's installed in the correct Python environment.")
    sys.exit(1)

# Import GUI modules
from gui.matrix_ui import main as matrix_main

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "witsv3_matrix.log"))
    ]
)

logger = logging.getLogger("WitsV3.GUI.Main")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="WitsV3 Matrix GUI")
    
    # Add arguments
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--minimize",
        action="store_true",
        help="Start minimized to system tray"
    )
    
    return parser.parse_args()

def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Configure logging level
    if args.debug:
        logging.getLogger("WitsV3").setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Log startup
    logger.info("Starting WitsV3 Matrix GUI")
    
    # Start matrix UI
    matrix_main()

if __name__ == "__main__":
    main()
