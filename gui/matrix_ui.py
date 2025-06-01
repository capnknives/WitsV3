"""
Matrix UI for WitsV3 Matrix GUI Application.

This module implements the matrix-style interface for the WitsV3 Matrix GUI.
It provides a customizable matrix-style chat interface with falling code animation.
"""

import os
import sys
import random
import asyncio
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

# Add debug information before importing qasync
print("Python version:", sys.version)
print("Available Qt implementations:")
for qt_impl in ["PyQt6", "PyQt5", "PySide6", "PySide2"]:
    try:
        __import__(qt_impl)
        print(f"- {qt_impl} is available")
    except ImportError:
        print(f"- {qt_impl} is NOT available")

try:
    import qasync
    print("qasync imported successfully")
except ImportError as e:
    print(f"Error importing qasync: {e}")

from PyQt6.QtCore import (
    Qt, QSize, QTimer, QRect, QPoint, QPropertyAnimation,
    QEasingCurve, pyqtSignal, pyqtSlot, QObject, QEvent
)
from PyQt6.QtGui import (
    QFont, QColor, QPalette, QPainter, QPixmap, QIcon,
    QTextCursor, QTextCharFormat, QTextDocument, QAction,
    QKeySequence, QFontMetrics, QTextOption, QTextBlockFormat
)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QLabel, QScrollArea,
    QSplitter, QFrame, QMenu, QMenuBar, QStatusBar, QSystemTrayIcon,
    QDialog, QTabWidget, QComboBox, QCheckBox, QSpinBox, QSlider,
    QColorDialog, QFileDialog, QMessageBox, QToolBar, QToolButton,
    QSizePolicy, QStyle, QStyleFactory, QGraphicsOpacityEffect,
    QGroupBox
)

# Add parent directory to path to import WitsV3 modules
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import WitsV3 modules
from core.schemas import StreamData
from gui.wits_connector import WitsConnector
from gui.startup_manager import StartupManager

logger = logging.getLogger("WitsV3.GUI.MatrixUI")

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
        self.columns = []
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

    def set_color_scheme(self, scheme_name):
        """
        Set the color scheme.

        Args:
            scheme_name: Name of the color scheme
        """
        if scheme_name in COLOR_SCHEMES:
            self.color_scheme = scheme_name
            self.char_color = COLOR_SCHEMES[scheme_name]["falling_chars"]
            self.update()

    def set_speed(self, speed):
        """
        Set the animation speed.

        Args:
            speed: Animation speed (0.1 to 2.0)
        """
        self.speed = max(0.1, min(2.0, speed))

        # Update column speeds
        for i in range(len(self.columns)):
            self.columns[i][1] = random.uniform(0.2, 1.0) * self.speed

    def set_density(self, density):
        """
        Set the animation density.

        Args:
            density: Animation density (0.1 to 1.0)
        """
        self.density = max(0.1, min(1.0, density))

    def set_char_size(self, size):
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

    def set_enabled(self, enabled):
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

class ChatMessageWidget(QWidget):
    """
    Widget that displays a chat message.

    This widget renders a single chat message with proper formatting.
    """

    def __init__(self, message, role, color_scheme="matrix_green", parent=None):
        """
        Initialize the ChatMessageWidget.

        Args:
            message: Message text
            role: Role of the message sender ("user" or "assistant")
            color_scheme: Color scheme name
            parent: Parent widget
        """
        super().__init__(parent)

        # Set attributes
        self.message = message
        self.role = role
        self.color_scheme = color_scheme

        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Create header
        header_layout = QHBoxLayout()

        # Create role label
        self.role_label = QLabel(role.capitalize())
        self.role_label.setFont(QFont("Courier New", 10, QFont.Weight.Bold))

        # Create timestamp label
        self.timestamp_label = QLabel(datetime.now().strftime("%H:%M:%S"))
        self.timestamp_label.setFont(QFont("Courier New", 8))
        self.timestamp_label.setAlignment(Qt.AlignmentFlag.AlignRight)

        # Add labels to header layout
        header_layout.addWidget(self.role_label)
        header_layout.addWidget(self.timestamp_label)

        # Create message text edit
        self.message_text = QTextEdit()
        self.message_text.setReadOnly(True)
        self.message_text.setFont(QFont("Courier New", 10))
        self.message_text.setFrameStyle(QFrame.Shape.NoFrame)
        self.message_text.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.message_text.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.message_text.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)

        # Set text
        self.message_text.setPlainText(message)

        # Add widgets to layout
        layout.addLayout(header_layout)
        layout.addWidget(self.message_text)

        # Apply color scheme
        self.apply_color_scheme(color_scheme)

    def apply_color_scheme(self, scheme_name):
        """
        Apply a color scheme to the widget.

        Args:
            scheme_name: Name of the color scheme
        """
        if scheme_name not in COLOR_SCHEMES:
            return

        self.color_scheme = scheme_name
        scheme = COLOR_SCHEMES[scheme_name]

        # Set role label color
        if self.role == "user":
            self.role_label.setStyleSheet(f"color: {scheme['user_text'].name()}")
        else:
            self.role_label.setStyleSheet(f"color: {scheme['assistant_text'].name()}")

        # Set timestamp label color
        self.timestamp_label.setStyleSheet(f"color: {scheme['status_text'].name()}")

        # Set message text colors
        palette = self.message_text.palette()
        if self.role == "user":
            palette.setColor(QPalette.ColorRole.Text, scheme["user_text"])
        else:
            palette.setColor(QPalette.ColorRole.Text, scheme["assistant_text"])
        palette.setColor(QPalette.ColorRole.Base, scheme["background"])
        self.message_text.setPalette(palette)

class ChatDisplayWidget(QScrollArea):
    """
    Widget that displays the chat history.

    This widget renders the chat history with proper formatting.
    """

    def __init__(self, color_scheme="matrix_green", parent=None):
        """
        Initialize the ChatDisplayWidget.

        Args:
            color_scheme: Color scheme name
            parent: Parent widget
        """
        super().__init__(parent)

        # Set attributes
        self.color_scheme = color_scheme
        
        # Message widgets
        self.message_widgets = []

        # Create container widget
        self.container = QWidget()
        self.container.setObjectName("chatContainer")

        # Create layout
        self.chat_layout = QVBoxLayout(self.container)
        self.chat_layout.setContentsMargins(10, 10, 10, 10)
        self.chat_layout.setSpacing(10)
        self.chat_layout.addStretch()

        # Set scroll area properties
        self.setWidget(self.container)
        self.setWidgetResizable(True)
        self.setFrameStyle(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Apply color scheme
        self.apply_color_scheme(color_scheme)

    def apply_color_scheme(self, scheme_name):
        """
        Apply a color scheme to the widget.

        Args:
            scheme_name: Name of the color scheme
        """
        if scheme_name not in COLOR_SCHEMES:
            return

        self.color_scheme = scheme_name
        scheme = COLOR_SCHEMES[scheme_name]

        # Set background color
        self.container.setStyleSheet(f"QWidget#chatContainer {{ background-color: {scheme['background'].name()} }}")

        # Update message widgets
        for widget in self.message_widgets:
            widget.apply_color_scheme(scheme_name)

    def add_message(self, message, role):
        """
        Add a message to the chat history.

        Args:
            message: Message text
            role: Role of the message sender ("user" or "assistant")
        """
        # Create message widget
        message_widget = ChatMessageWidget(message, role, self.color_scheme)

        # Add to layout (before the stretch)
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, message_widget)

        # Add to list
        self.message_widgets.append(message_widget)

        # Scroll to bottom
        scrollbar = self.verticalScrollBar()
        if scrollbar:
            scrollbar.setValue(scrollbar.maximum())

    def clear(self):
        """Clear the chat history."""
        # Remove all message widgets
        for widget in self.message_widgets:
            self.chat_layout.removeWidget(widget)
            widget.deleteLater()

        # Clear list
        self.message_widgets = []

class MatrixMainWindow(QMainWindow):
    """
    Main window for the WitsV3 Matrix GUI.

    This class implements the main window for the WitsV3 Matrix GUI,
    including the matrix-style interface and all UI components.
    """

    def __init__(self):
        """Initialize the MatrixMainWindow."""
        super().__init__()

        # Set attributes
        self.color_scheme = "matrix_green"
        self.animation_enabled = True
        self.animation_speed = 1.0
        self.animation_density = 0.5
        self.show_thinking = True
        self.always_on_top = False
        self.opacity = 0.95
        self.processing = False

        # Create WitsConnector
        self.connector = WitsConnector()

        # Initialize UI
        self.init_ui()

        # Set window properties
        self.setWindowTitle("WitsV3 Matrix")
        self.setMinimumSize(800, 600)
        self.resize(1000, 800)

        # Set window opacity
        self.set_opacity(self.opacity)

    def init_ui(self):
        """Initialize the UI."""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Create matrix widget
        self.matrix_widget = MatrixRainWidget(self)
        self.matrix_widget.setGeometry(0, 0, self.width(), self.height())

        # Create chat display
        self.chat_display = ChatDisplayWidget(self.color_scheme)

        # Create input layout
        input_layout = QHBoxLayout()
        input_layout.setContentsMargins(10, 10, 10, 10)
        input_layout.setSpacing(10)

        # Create input field
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Type your message here...")
        self.input_field.returnPressed.connect(self.send_message)

        # Create send button
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)

        # Add widgets to input layout
        input_layout.addWidget(self.input_field)
        input_layout.addWidget(self.send_button)

        # Create status bar
        self.status_bar = QStatusBar()
        self.status_bar.showMessage("Ready")

        # Add widgets to main layout
        main_layout.addWidget(self.chat_display)
        main_layout.addLayout(input_layout)
        main_layout.addWidget(self.status_bar)

        # Apply color scheme
        self.apply_color_scheme(self.color_scheme)

    def init_system_tray(self):
        """Initialize the system tray icon."""
        # Create system tray icon
        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setIcon(QIcon.fromTheme("dialog-information"))
        self.tray_icon.setToolTip("WitsV3 Matrix")

        # Create tray menu
        tray_menu = QMenu()

        # Create actions
        show_action = QAction("Show", self)
        show_action.triggered.connect(self.show)

        hide_action = QAction("Hide", self)
        hide_action.triggered.connect(self.hide)

        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.close)

        # Add actions to menu
        tray_menu.addAction(show_action)
        tray_menu.addAction(hide_action)
        tray_menu.addSeparator()
        tray_menu.addAction(quit_action)

        # Set tray menu
        self.tray_icon.setContextMenu(tray_menu)

        # Show tray icon
        self.tray_icon.show()

    async def init_connector(self):
        """Initialize the WitsConnector."""
        try:
            # Initialize connector
            await self.connector.initialize()

            # Register callbacks
            self.connector.register_callback("on_thinking", self.on_thinking)
            self.connector.register_callback("on_action", self.on_action)
            self.connector.register_callback("on_observation", self.on_observation)
            self.connector.register_callback("on_result", self.on_result)
            self.connector.register_callback("on_error", self.on_error)
            self.connector.register_callback("on_status_change", self.on_status_change)
            self.connector.register_callback("on_model_change", self.on_model_change)
            self.connector.register_callback("on_restart", self.on_restart)

            # Update status
            self.status_bar.showMessage("Connected")
        except Exception as e:
            # Update status
            self.status_bar.showMessage(f"Error: {e}")
            logger.error(f"Error initializing connector: {e}")

    def apply_color_scheme(self, scheme_name):
        """
        Apply a color scheme to the window.

        Args:
            scheme_name: Name of the color scheme
        """
        if scheme_name not in COLOR_SCHEMES:
            return

        self.color_scheme = scheme_name
        scheme = COLOR_SCHEMES[scheme_name]

        # Apply to matrix widget
        self.matrix_widget.set_color_scheme(scheme_name)

        # Apply to chat display
        self.chat_display.apply_color_scheme(scheme_name)

        # Apply to input field
        self.input_field.setStyleSheet(
            f"background-color: {scheme['input_background'].name()}; "
            f"color: {scheme['input_text'].name()}; "
            f"border: 1px solid {scheme['text'].name()}; "
            f"padding: 5px;"
        )

        # Apply to send button
        self.send_button.setStyleSheet(
            f"background-color: {scheme['input_background'].name()}; "
            f"color: {scheme['input_text'].name()}; "
            f"border: 1px solid {scheme['text'].name()}; "
            f"padding: 5px;"
        )

        # Apply to status bar
        self.status_bar.setStyleSheet(
            f"background-color: {scheme['background'].name()}; "
            f"color: {scheme['status_text'].name()}; "
            f"border-top: 1px solid {scheme['text'].name()};"
        )

    def set_opacity(self, opacity):
        """
        Set the window opacity.

        Args:
            opacity: Window opacity (0.0 to 1.0)
        """
        self.opacity = max(0.1, min(1.0, opacity))
        self.setWindowOpacity(self.opacity)

    def set_always_on_top(self, always_on_top):
        """
        Set whether the window is always on top.

        Args:
            always_on_top: Whether the window is always on top
        """
        self.always_on_top = always_on_top

        # Set window flags
        if always_on_top:
            self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        else:
            self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowStaysOnTopHint)

        # Show window
        self.show()

    def send_message(self):
        """Send a message."""
        # Get message
        message = self.input_field.text().strip()

        # Check if message is empty
        if not message:
            return

        # Clear input field
        self.input_field.clear()

        # Add message to chat display
        self.chat_display.add_message(message, "user")

        # Process message
        asyncio.create_task(self.process_message(message))

    async def process_message(self, message):
        """
        Process a message.

        Args:
            message: Message to process
        """
        # Set processing flag
        self.processing = True

        # Update status
        self.status_bar.showMessage("Processing...")

        # Disable input
        self.input_field.setEnabled(False)
        self.send_button.setEnabled(False)

        try:
            # Process message with streaming
            async for stream_data in self.connector.process_message_stream(message):
                # Handle stream data
                if stream_data.type == "thinking":
                    # Show thinking message
                    if self.show_thinking:
                        self.chat_display.add_message(stream_data.content, "thinking")
                elif stream_data.type == "action":
                    # Show action message
                    self.chat_display.add_message(stream_data.content, "action")
                elif stream_data.type == "observation":
                    # Show observation message
                    self.chat_display.add_message(stream_data.content, "observation")
                elif stream_data.type == "result":
                    # Show result message
                    self.chat_display.add_message(stream_data.content, "assistant")
                elif stream_data.type == "error":
                    # Show error message
                    self.chat_display.add_message(stream_data.content, "error")
        except Exception as e:
            # Show error message
            self.chat_display.add_message(f"Error: {e}", "error")
            logger.error(f"Error processing message: {e}")
        finally:
            # Clear processing flag
            self.processing = False

            # Update status
            self.status_bar.showMessage("Ready")

            # Enable input
            self.input_field.setEnabled(True)
            self.send_button.setEnabled(True)

    def on_thinking(self, content):
        """
        Handle thinking event.

        Args:
            content: Thinking content
        """
        # Show thinking message
        if self.show_thinking:
            self.chat_display.add_message(content, "thinking")

    def on_action(self, content):
        """
        Handle action event.

        Args:
            content: Action content
        """
        # Show action message
        self.chat_display.add_message(content, "action")

    def on_observation(self, content):
        """
        Handle observation event.

        Args:
            content: Observation content
        """
        # Show observation message
        self.chat_display.add_message(content, "observation")

    def on_result(self, content):
        """
        Handle result event.

        Args:
            content: Result content
        """
        # Show result message
        self.chat_display.add_message(content, "assistant")

    def on_error(self, content):
        """
        Handle error event.

        Args:
            content: Error content
        """
        # Show error message
        self.chat_display.add_message(content, "error")

    def on_status_change(self, status):
        """
        Handle status change event.

        Args:
            status: New status
        """
        # Update status
        self.status_bar.showMessage(status)

    def on_model_change(self, model):
        """
        Handle model change event.
        
        Args:
            model: New model
        """
        # Update window title
        self.setWindowTitle(f"WitsV3 Matrix - {model}")
        
    def on_restart(self, reason):
        """
        Handle restart event.
        
        Args:
            reason: Reason for restart
        """
        # Show restart message
        self.chat_display.add_message(f"System is restarting: {reason}", "system")
        self.status_bar.showMessage(f"Restarting: {reason}")

    def closeEvent(self, event):
        """
        Handle close event.

        Args:
            event: Close event
        """
        # Shutdown connector
        asyncio.create_task(self.shutdown_connector())

        # Accept event
        event.accept()

    async def shutdown_connector(self):
        """Shutdown the WitsConnector."""
        try:
            # Shutdown connector
            await self.connector.shutdown()
        except Exception as e:
            logger.error(f"Error shutting down connector: {e}")

def main():
    """Main entry point for the Matrix UI."""
    # Create application
    app = QApplication(sys.argv)
    
    # Create event loop
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)
    
    # Create main window
    window = MatrixMainWindow()
    
    # Initialize system tray
    window.init_system_tray()
    
    # Show window
    window.show()
    
    # Schedule connector initialization
    asyncio.ensure_future(window.init_connector())
    
    # Run event loop
    with loop:
        loop.run_forever()

if __name__ == "__main__":
    main()
