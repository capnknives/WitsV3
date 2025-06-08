"""
Chat widgets for the Matrix UI.

This module implements the chat message and display widgets.
"""

from datetime import datetime
from typing import List, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QPalette
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLabel, 
    QScrollArea, QFrame
)

from .color_schemes import COLOR_SCHEMES


class ChatMessageWidget(QWidget):
    """
    Widget that displays a chat message.

    This widget renders a single chat message with proper formatting.
    """

    def __init__(self, message: str, role: str, color_scheme: str = "matrix_green", parent=None):
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

    def apply_color_scheme(self, scheme_name: str):
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

    def __init__(self, color_scheme: str = "matrix_green", parent=None):
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
        self.message_widgets: List[ChatMessageWidget] = []

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

    def apply_color_scheme(self, scheme_name: str):
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
        self.container.setStyleSheet(
            f"QWidget#chatContainer {{ background-color: {scheme['background'].name()} }}"
        )

        # Update message widgets
        for widget in self.message_widgets:
            widget.apply_color_scheme(scheme_name)

    def add_message(self, message: str, role: str):
        """
        Add a message to the chat history.

        Args:
            message: Message text
            role: Role of the message sender ("user" or "assistant")
        """
        # Create message widget
        message_widget = ChatMessageWidget(message, role, self.color_scheme)
        
        # Add widget to layout (before the stretch)
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, message_widget)
        
        # Store reference
        self.message_widgets.append(message_widget)

        # Scroll to bottom
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

    def clear(self):
        """Clear all messages from the chat history."""
        # Remove all message widgets
        for widget in self.message_widgets:
            widget.deleteLater()

        # Clear the list
        self.message_widgets.clear()