"""
Matrix rain animation widget.

This module implements the falling character animation for the matrix-style interface.
"""

import random
from typing import List, Tuple

from PyQt6.QtCore import Qt, QTimer, QRect
from PyQt6.QtGui import QFont, QColor, QPainter
from PyQt6.QtWidgets import QWidget

from .color_schemes import COLOR_SCHEMES


class MatrixRainWidget(QWidget):
    """
    Widget that displays the matrix rain animation.

    This widget renders falling characters in a matrix-style animation.
    """

    def __init__(self, parent=None):
        """
        Initialize the MatrixRainWidget.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)

        # Set attributes
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        # Animation settings
        self.color_scheme = "matrix_green"
        self.char_color = COLOR_SCHEMES[self.color_scheme]["falling_chars"]
        self.chars = []
        self.columns: List[List[float]] = []
        self.column_count = 0
        self.char_size = 14
        self.speed = 1.0
        self.density = 0.5
        self.enabled = True

        # Characters to use
        self.char_set = "".join([chr(i) for i in range(33, 127)] +
                               [chr(i) for i in range(0x30A0, 0x30FF)] +
                               [chr(i) for i in range(0x3040, 0x309F)])

        # Font
        self.matrix_font = QFont("Courier New", self.char_size)
        self.matrix_font.setStyleHint(QFont.StyleHint.Monospace)

        # Timer for animation
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(50)  # 20 FPS

        # Initialize columns
        self.init_columns()

    def init_columns(self):
        """Initialize the columns for the animation."""
        # Calculate number of columns based on width
        self.column_count = self.width() // (self.char_size + 2)

        # Initialize columns
        self.columns = []
        for i in range(self.column_count):
            # Each column has: [position, speed, length, start_delay]
            self.columns.append([
                random.randint(-20, 0),  # Position (negative = not yet visible)
                random.uniform(0.2, 1.0) * self.speed,  # Speed
                random.randint(5, 30),  # Length
                random.randint(0, 100)  # Start delay
            ])

    def update_animation(self):
        """Update the animation state."""
        if not self.enabled:
            return

        # Update column positions
        for i in range(len(self.columns)):
            # Check if column should start moving yet
            if self.columns[i][3] > 0:
                self.columns[i][3] -= 1
                continue

            # Move column down
            self.columns[i][0] += self.columns[i][1]

            # Reset column if it's gone off the bottom
            if self.columns[i][0] - self.columns[i][2] > self.height() // self.char_size:
                # Only reset some columns to maintain density
                if random.random() < self.density:
                    self.columns[i][0] = random.randint(-20, 0)
                    self.columns[i][1] = random.uniform(0.2, 1.0) * self.speed
                    self.columns[i][2] = random.randint(5, 30)
                else:
                    self.columns[i][0] = -100  # Hide column

        # Trigger repaint
        self.update()

    def set_color_scheme(self, scheme_name: str):
        """
        Set the color scheme.

        Args:
            scheme_name: Name of the color scheme
        """
        if scheme_name in COLOR_SCHEMES:
            self.color_scheme = scheme_name
            self.char_color = COLOR_SCHEMES[scheme_name]["falling_chars"]
            self.update()

    def set_speed(self, speed: float):
        """
        Set the animation speed.

        Args:
            speed: Animation speed (0.1 to 2.0)
        """
        self.speed = max(0.1, min(2.0, speed))

        # Update column speeds
        for i in range(len(self.columns)):
            self.columns[i][1] = random.uniform(0.2, 1.0) * self.speed

    def set_density(self, density: float):
        """
        Set the animation density.

        Args:
            density: Animation density (0.1 to 1.0)
        """
        self.density = max(0.1, min(1.0, density))

    def set_char_size(self, size: int):
        """
        Set the character size.

        Args:
            size: Character size in pixels
        """
        self.char_size = max(8, min(24, size))
        self.matrix_font = QFont("Courier New", self.char_size)
        self.matrix_font.setStyleHint(QFont.StyleHint.Monospace)

        # Reinitialize columns
        self.init_columns()

    def set_enabled(self, enabled: bool):
        """
        Enable or disable the animation.

        Args:
            enabled: Whether the animation is enabled
        """
        self.enabled = enabled

        if enabled and not self.timer.isActive():
            self.timer.start(50)
        elif not enabled and self.timer.isActive():
            self.timer.stop()

    def resizeEvent(self, event):
        """Handle resize events."""
        super().resizeEvent(event)

        # Reinitialize columns
        self.init_columns()

    def paintEvent(self, event):
        """Paint the animation."""
        if not self.enabled:
            return

        # Create painter
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Clear background
        painter.fillRect(event.rect(), COLOR_SCHEMES[self.color_scheme]["background"])

        # Set font
        painter.setFont(self.matrix_font)

        # Draw falling characters
        for i, column in enumerate(self.columns):
            # Skip if column is not visible yet
            if column[3] > 0 or column[0] < 0:
                continue

            # Calculate x position
            x = i * (self.char_size + 2)

            # Draw characters in column
            for j in range(int(column[2])):
                # Calculate y position
                y = int((column[0] - j) * self.char_size)

                # Skip if character is not visible
                if y < -self.char_size or y > self.height():
                    continue

                # Get random character
                char = random.choice(self.char_set)

                # Calculate color (fade out towards the end of the column)
                fade = 1.0 - (j / column[2])
                color = QColor(
                    self.char_color.red(),
                    self.char_color.green(),
                    self.char_color.blue(),
                    int(self.char_color.alpha() * fade)
                )

                # Draw character
                painter.setPen(color)
                painter.drawText(QRect(x, y, self.char_size, self.char_size),
                                Qt.AlignmentFlag.AlignCenter, char)