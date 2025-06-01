# WitsV3 Matrix GUI

A matrix-style chat interface for the WitsV3 system. This application provides a customizable matrix-style chat interface with falling code animation, allowing you to interact with the WitsV3 AI system in a visually appealing way.

## Features

- Matrix-style falling code animation
- Multiple color schemes (Matrix Green, Cyberpunk Blue, Amber Terminal, High Contrast)
- Customizable animation settings (speed, density, character size)
- System tray integration for background operation
- Autostart with Windows option
- Always-on-top mode
- Adjustable window opacity
- Chat history saving

## Installation

1. Make sure you have Python 3.8+ installed
2. Install the required dependencies:

```bash
pip install -r gui/requirements.txt
```

## Usage

To start the application:

```bash
python gui/main.py
```

### Command Line Options

- `--debug`: Enable debug logging
- `--minimize`: Start minimized to system tray

## Configuration

The application can be configured through the preferences dialog, accessible from the Settings menu.

### Appearance Settings

- **Color Scheme**: Choose between Matrix Green, Cyberpunk Blue, Amber Terminal, and High Contrast
- **Animation**: Enable/disable animation, adjust speed and density
- **Window**: Configure always-on-top mode and window opacity

### Behavior Settings

- **Chat**: Show/hide thinking messages
- **Startup**: Enable/disable autostart with Windows

## System Requirements

- Windows 10/11
- Python 3.8+
- PyQt6
- WitsV3 system installed and configured

## License

This project is licensed under the same license as the WitsV3 system.
