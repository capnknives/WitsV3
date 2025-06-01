# WitsV3 Matrix GUI Application

A Windows application that provides a matrix-style chat interface for the WitsV3 system. The application starts with Windows and ensures that Wits, the main personality of the AI system, is always accessible and runs on the strongest available model.

## Key Features

1. **Matrix-Style Interface**: A visually appealing interface with customizable falling code animation that resembles the Matrix movie aesthetic.

2. **Windows Autostart**: The application can be configured to start automatically with Windows, ensuring the AI system is always available.

3. **System Tray Integration**: Runs in the background with system tray access, allowing for quick access without cluttering the desktop.

4. **Multiple Color Schemes**: Choose between Matrix Green, Cyberpunk Blue, Amber Terminal, and High Contrast themes.

5. **Customizable Settings**: Adjust animation speed, density, window opacity, and other visual preferences.

6. **Always-on-Top Mode**: Keep the chat window visible over other applications when needed.

7. **WitsV3 Integration**: Seamlessly connects to the WitsV3 system, providing access to all its capabilities.

## Project Structure

- `gui/matrix_ui.py`: Main UI implementation with matrix-style interface
- `gui/wits_connector.py`: Connector to the WitsV3 system
- `gui/startup_manager.py`: Handles Windows startup integration
- `gui/main.py`: Entry point for the application
- `gui/requirements.txt`: Dependencies for the GUI
- `gui/README.md`: Documentation for the application

## How to Use

1. Install the required dependencies:
   ```
   pip install -r gui/requirements.txt
   ```

2. Run the application:
   ```
   python gui/main.py
   ```

3. To enable autostart with Windows, open the application and go to File > Preferences > Behavior and check "Start with Windows".

4. The application will connect to the WitsV3 system automatically and display the current model in the status bar.

5. Type your messages in the input field at the bottom of the window and press Enter or click Send to interact with Wits.

The application is designed to be always accessible, running in the background even when closed (minimized to system tray), and providing a visually engaging way to interact with the WitsV3 AI system.

## Implementation Details

### MatrixRainWidget

This widget renders the falling characters in a matrix-style animation. It supports:
- Customizable animation speed and density
- Multiple color schemes
- Adjustable character size

### ChatDisplayWidget

This widget displays the chat history with proper formatting for different message types:
- User messages
- Assistant responses
- Thinking messages
- Action messages
- Observation messages
- Error messages

### MatrixMainWindow

The main window of the application, which includes:
- Matrix animation background
- Chat display
- Input field
- Status bar
- Menu bar
- System tray integration

### WitsConnector

Connects the GUI to the WitsV3 system, providing:
- Message processing
- Streaming responses
- System status information
- Model information

### StartupManager

Manages the autostart of the application with Windows, using the Windows registry.
