"""
Main window for the Matrix UI.

This module implements the main window of the matrix-style interface.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import Optional

import qasync

from PyQt6.QtCore import Qt, QSize, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QFont, QIcon, QAction, QKeySequence
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QLabel, QSplitter, QFrame, QMenuBar,
    QStatusBar, QSystemTrayIcon, QMenu, QMessageBox
)

# Add parent directory to path to import WitsV3 modules
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import WitsV3 modules
from core.schemas import StreamData
from gui.wits_connector import WitsConnector
from gui.startup_manager import StartupManager

# Import Matrix UI modules
from .color_schemes import COLOR_SCHEMES
from .matrix_rain import MatrixRainWidget
from .chat_widgets import ChatDisplayWidget

logger = logging.getLogger("WitsV3.GUI.MatrixWindow")


class MatrixMainWindow(QMainWindow):
    """
    Main window for the Matrix-style WitsV3 GUI.

    This window provides a matrix-style interface with falling code animation.
    """

    def __init__(self):
        """Initialize the MatrixMainWindow."""
        super().__init__()

        # Set window properties
        self.setWindowTitle("WitsV3 Matrix Interface")
        self.setGeometry(100, 100, 1200, 800)

        # State
        self.color_scheme = "matrix_green"
        self.opacity = 0.95
        self.always_on_top = False

        # Connectors
        self.connector: Optional[WitsConnector] = None
        self.startup_manager: Optional[StartupManager] = None

        # Widgets
        self.matrix_rain: Optional[MatrixRainWidget] = None
        self.chat_display: Optional[ChatDisplayWidget] = None
        self.input_field: Optional[QLineEdit] = None
        self.send_button: Optional[QPushButton] = None
        self.status_label: Optional[QLabel] = None
        self.model_label: Optional[QLabel] = None

        # Initialize UI
        self.init_ui()
        self.init_system_tray()

        # Set opacity
        self.setWindowOpacity(self.opacity)

        # Apply color scheme
        self.apply_color_scheme(self.color_scheme)

    def init_ui(self):
        """Initialize the user interface."""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Create matrix rain widget
        self.matrix_rain = MatrixRainWidget()
        
        # Create chat display widget
        self.chat_display = ChatDisplayWidget(self.color_scheme)

        # Stack matrix rain behind chat display
        self.matrix_rain.lower()

        # Create input area
        input_widget = QWidget()
        input_layout = QHBoxLayout(input_widget)
        input_layout.setContentsMargins(10, 10, 10, 10)

        # Create input field
        self.input_field = QLineEdit()
        self.input_field.setFont(QFont("Courier New", 12))
        self.input_field.setPlaceholderText("Enter your message...")
        self.input_field.returnPressed.connect(self.send_message)

        # Create send button
        self.send_button = QPushButton("Send")
        self.send_button.setFont(QFont("Courier New", 12))
        self.send_button.clicked.connect(self.send_message)
        self.send_button.setFixedWidth(100)

        # Add to input layout
        input_layout.addWidget(self.input_field)
        input_layout.addWidget(self.send_button)

        # Add widgets to main layout
        main_layout.addWidget(self.matrix_rain)
        main_layout.addWidget(self.chat_display)
        main_layout.addWidget(input_widget)

        # Position matrix rain to cover entire window
        self.matrix_rain.setGeometry(0, 0, self.width(), self.height())

        # Create menu bar
        self.create_menu_bar()

        # Create status bar
        self.create_status_bar()

    def create_menu_bar(self):
        """Create the menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu('&File')

        # Clear chat action
        clear_action = QAction('&Clear Chat', self)
        clear_action.setShortcut(QKeySequence.StandardKey.New)
        clear_action.triggered.connect(self.chat_display.clear)
        file_menu.addAction(clear_action)

        file_menu.addSeparator()

        # Exit action
        exit_action = QAction('E&xit', self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View menu
        view_menu = menubar.addMenu('&View')

        # Color scheme submenu
        color_menu = view_menu.addMenu('&Color Scheme')
        for scheme_name in COLOR_SCHEMES:
            action = QAction(scheme_name.replace('_', ' ').title(), self)
            action.triggered.connect(lambda checked, s=scheme_name: self.apply_color_scheme(s))
            color_menu.addAction(action)

        # Always on top action
        always_on_top_action = QAction('&Always on Top', self)
        always_on_top_action.setCheckable(True)
        always_on_top_action.triggered.connect(self.set_always_on_top)
        view_menu.addAction(always_on_top_action)

        # Matrix rain toggle
        matrix_toggle = QAction('&Matrix Rain', self)
        matrix_toggle.setCheckable(True)
        matrix_toggle.setChecked(True)
        matrix_toggle.triggered.connect(lambda checked: self.matrix_rain.set_enabled(checked))
        view_menu.addAction(matrix_toggle)

    def create_status_bar(self):
        """Create the status bar."""
        statusbar = self.statusBar()

        # Create status label
        self.status_label = QLabel("Disconnected")
        self.status_label.setFont(QFont("Courier New", 10))
        statusbar.addWidget(self.status_label)

        # Create model label
        self.model_label = QLabel("Model: None")
        self.model_label.setFont(QFont("Courier New", 10))
        statusbar.addPermanentWidget(self.model_label)

    def init_system_tray(self):
        """Initialize the system tray icon."""
        # Check if system tray is available
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return

        # Create system tray icon
        self.tray_icon = QSystemTrayIcon(self)
        
        # Set icon (use a simple circle for now)
        icon = self.style().standardIcon(self.style().StandardPixmap.SP_ComputerIcon)
        self.tray_icon.setIcon(icon)

        # Create tray menu
        tray_menu = QMenu()

        # Show/Hide action
        show_action = QAction("Show/Hide", self)
        show_action.triggered.connect(self.toggle_visibility)
        tray_menu.addAction(show_action)

        tray_menu.addSeparator()

        # Exit action
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        tray_menu.addAction(exit_action)

        # Set menu
        self.tray_icon.setContextMenu(tray_menu)

        # Connect double-click
        self.tray_icon.activated.connect(self.tray_icon_activated)

        # Show tray icon
        self.tray_icon.show()

    def toggle_visibility(self):
        """Toggle window visibility."""
        if self.isVisible():
            self.hide()
        else:
            self.show()
            self.raise_()
            self.activateWindow()

    def tray_icon_activated(self, reason):
        """Handle tray icon activation."""
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self.toggle_visibility()

    async def init_connector(self):
        """Initialize the WitsV3 connector."""
        logger.info("Initializing WitsV3 connector...")

        # Create startup manager
        self.startup_manager = StartupManager()

        # Start WitsV3 backend
        success, message = await self.startup_manager.start_witsv3()
        if not success:
            logger.error(f"Failed to start WitsV3: {message}")
            self.on_error(f"Failed to start WitsV3: {message}")
            return

        # Create connector
        self.connector = WitsConnector()

        # Connect signals
        self.connector.thinking_signal.connect(self.on_thinking)
        self.connector.action_signal.connect(self.on_action)
        self.connector.observation_signal.connect(self.on_observation)
        self.connector.result_signal.connect(self.on_result)
        self.connector.error_signal.connect(self.on_error)
        self.connector.status_signal.connect(self.on_status_change)
        self.connector.model_signal.connect(self.on_model_change)
        self.connector.restart_signal.connect(self.on_restart)

        # Connect to WitsV3
        await self.connector.connect()

    def apply_color_scheme(self, scheme_name: str):
        """
        Apply a color scheme to the window.

        Args:
            scheme_name: Name of the color scheme
        """
        if scheme_name not in COLOR_SCHEMES:
            return

        self.color_scheme = scheme_name
        scheme = COLOR_SCHEMES[scheme_name]

        # Update matrix rain
        if self.matrix_rain:
            self.matrix_rain.set_color_scheme(scheme_name)

        # Update chat display
        if self.chat_display:
            self.chat_display.apply_color_scheme(scheme_name)

        # Update input field
        if self.input_field:
            self.input_field.setStyleSheet(f"""
                QLineEdit {{
                    background-color: {scheme['input_background'].name()};
                    color: {scheme['input_text'].name()};
                    border: 1px solid {scheme['input_text'].name()};
                    padding: 5px;
                }}
            """)

        # Update send button
        if self.send_button:
            self.send_button.setStyleSheet(f"""
                QPushButton {{
                    background-color: {scheme['input_background'].name()};
                    color: {scheme['input_text'].name()};
                    border: 1px solid {scheme['input_text'].name()};
                    padding: 5px;
                }}
                QPushButton:hover {{
                    background-color: {scheme['input_text'].name()};
                    color: {scheme['input_background'].name()};
                }}
            """)

        # Update status bar
        statusbar = self.statusBar()
        if statusbar:
            statusbar.setStyleSheet(f"""
                QStatusBar {{
                    background-color: {scheme['background'].name()};
                    color: {scheme['status_text'].name()};
                }}
            """)

        # Update window
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {scheme['background'].name()};
            }}
        """)

    def set_opacity(self, opacity: float):
        """
        Set the window opacity.

        Args:
            opacity: Opacity value (0.0 to 1.0)
        """
        self.opacity = max(0.1, min(1.0, opacity))
        self.setWindowOpacity(self.opacity)

    def set_always_on_top(self, always_on_top: bool):
        """
        Set whether the window is always on top.

        Args:
            always_on_top: Whether the window should be always on top
        """
        self.always_on_top = always_on_top

        if always_on_top:
            self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        else:
            self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowStaysOnTopHint)

        # Re-show the window to apply the flag change
        self.show()

    def send_message(self):
        """Send a message to WitsV3."""
        message = self.input_field.text().strip()
        if not message:
            return

        # Clear input
        self.input_field.clear()

        # Add to chat display
        self.chat_display.add_message(message, "user")

        # Send to WitsV3
        if self.connector:
            asyncio.create_task(self.process_message(message))

    async def process_message(self, message: str):
        """
        Process a message asynchronously.

        Args:
            message: Message to process
        """
        try:
            if not self.connector:
                self.on_error("Not connected to WitsV3")
                return

            await self.connector.send_message(message)
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self.on_error(f"Error processing message: {str(e)}")

    def resizeEvent(self, event):
        """Handle resize events."""
        super().resizeEvent(event)

        # Resize matrix rain to cover entire window
        if self.matrix_rain:
            self.matrix_rain.setGeometry(0, 0, self.width(), self.height())

    @pyqtSlot(str)
    def on_thinking(self, content: str):
        """
        Handle thinking content.

        Args:
            content: Thinking content
        """
        # Add thinking message
        self.chat_display.add_message(f"🤔 {content}", "assistant")

    @pyqtSlot(str)
    def on_action(self, content: str):
        """
        Handle action content.

        Args:
            content: Action content
        """
        # Add action message
        self.chat_display.add_message(f"⚡ {content}", "assistant")

    @pyqtSlot(str)
    def on_observation(self, content: str):
        """
        Handle observation content.

        Args:
            content: Observation content
        """
        # Add observation message
        self.chat_display.add_message(f"👁️ {content}", "assistant")

    @pyqtSlot(str)
    def on_result(self, content: str):
        """
        Handle result content.

        Args:
            content: Result content
        """
        # Add result message
        self.chat_display.add_message(content, "assistant")

    @pyqtSlot(str)
    def on_error(self, content: str):
        """
        Handle error content.

        Args:
            content: Error content
        """
        # Add error message
        self.chat_display.add_message(f"❌ Error: {content}", "assistant")

    @pyqtSlot(str)
    def on_status_change(self, status: str):
        """
        Handle status change.

        Args:
            status: New status
        """
        self.status_label.setText(status)

    @pyqtSlot(str)
    def on_model_change(self, model: str):
        """
        Handle model change.

        Args:
            model: New model name
        """
        self.model_label.setText(f"Model: {model}")

    @pyqtSlot(str)
    def on_restart(self, reason: str):
        """
        Handle restart notification.

        Args:
            reason: Restart reason
        """
        self.chat_display.add_message(f"🔄 Restarting: {reason}", "assistant")

    def closeEvent(self, event):
        """Handle close event."""
        # Hide to tray if available
        if hasattr(self, 'tray_icon') and self.tray_icon.isVisible():
            event.ignore()
            self.hide()
            self.tray_icon.showMessage(
                "WitsV3 Matrix",
                "Application minimized to tray",
                QSystemTrayIcon.MessageIcon.Information,
                2000
            )
        else:
            # Shutdown connector
            if self.connector:
                asyncio.create_task(self.shutdown_connector())
            event.accept()

    async def shutdown_connector(self):
        """Shutdown the connector and backend."""
        if self.connector:
            await self.connector.disconnect()

        if self.startup_manager:
            await self.startup_manager.stop_witsv3()


def main():
    """Main entry point for the Matrix UI."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("WitsV3 Matrix")
    app.setOrganizationName("WitsV3")

    # Create event loop
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)

    # Create main window
    window = MatrixMainWindow()
    window.show()

    # Initialize connector
    asyncio.create_task(window.init_connector())

    # Run event loop
    with loop:
        loop.run_forever()


if __name__ == "__main__":
    main()