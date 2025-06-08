"""
Matrix UI for WitsV3 Matrix GUI Application.

This module provides the entry point for the matrix-style interface for the WitsV3 Matrix GUI.
The implementation has been refactored into separate modules for better maintainability.
"""

import sys
import logging

# Debug information
print("Python version:", sys.version)
print("Matrix UI module loading...")

# Import main window and run function
from .matrix_window import MatrixMainWindow, main

# Re-export for backward compatibility
__all__ = ['MatrixMainWindow', 'main', 'COLOR_SCHEMES']

# Import color schemes for external use
from .color_schemes import COLOR_SCHEMES

logger = logging.getLogger("WitsV3.GUI.MatrixUI")
logger.info("Matrix UI module loaded successfully")

if __name__ == "__main__":
    main()
