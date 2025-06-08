"""
Color schemes for the Matrix UI.

This module defines color schemes for the matrix-style interface.
"""

from PyQt6.QtGui import QColor

# Matrix color schemes
COLOR_SCHEMES = {
    "matrix_green": {
        "background": QColor(0, 0, 0),
        "text": QColor(0, 255, 0),
        "user_text": QColor(0, 255, 0),
        "assistant_text": QColor(0, 200, 0),
        "thinking_text": QColor(0, 150, 0),
        "action_text": QColor(0, 255, 100),
        "observation_text": QColor(100, 255, 100),
        "error_text": QColor(255, 0, 0),
        "status_text": QColor(0, 150, 0),
        "input_background": QColor(0, 40, 0),
        "input_text": QColor(0, 255, 0),
        "falling_chars": QColor(0, 150, 0, 150),
    },
    "cyberpunk_blue": {
        "background": QColor(0, 0, 20),
        "text": QColor(0, 150, 255),
        "user_text": QColor(0, 200, 255),
        "assistant_text": QColor(0, 150, 255),
        "thinking_text": QColor(0, 100, 200),
        "action_text": QColor(0, 200, 255),
        "observation_text": QColor(100, 200, 255),
        "error_text": QColor(255, 50, 50),
        "status_text": QColor(0, 100, 200),
        "input_background": QColor(0, 20, 40),
        "input_text": QColor(0, 200, 255),
        "falling_chars": QColor(0, 100, 200, 150),
    },
    "amber_terminal": {
        "background": QColor(0, 0, 0),
        "text": QColor(255, 180, 0),
        "user_text": QColor(255, 200, 0),
        "assistant_text": QColor(255, 150, 0),
        "thinking_text": QColor(200, 120, 0),
        "action_text": QColor(255, 200, 0),
        "observation_text": QColor(255, 220, 100),
        "error_text": QColor(255, 0, 0),
        "status_text": QColor(200, 120, 0),
        "input_background": QColor(40, 20, 0),
        "input_text": QColor(255, 180, 0),
        "falling_chars": QColor(200, 120, 0, 150),
    },
    "high_contrast": {
        "background": QColor(0, 0, 0),
        "text": QColor(255, 255, 255),
        "user_text": QColor(255, 255, 255),
        "assistant_text": QColor(200, 200, 200),
        "thinking_text": QColor(150, 150, 150),
        "action_text": QColor(255, 255, 255),
        "observation_text": QColor(220, 220, 220),
        "error_text": QColor(255, 100, 100),
        "status_text": QColor(150, 150, 150),
        "input_background": QColor(40, 40, 40),
        "input_text": QColor(255, 255, 255),
        "falling_chars": QColor(150, 150, 150, 150),
    }
}