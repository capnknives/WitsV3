"""
Startup Manager for WitsV3 Matrix GUI Application.

This module provides functionality to manage the autostart of the WitsV3 Matrix GUI application.
It handles creating and removing startup entries in the Windows registry.
"""

import os
import sys
import logging
import winreg
from pathlib import Path

logger = logging.getLogger("WitsV3.GUI.StartupManager")

class StartupManager:
    """
    Startup Manager for WitsV3 Matrix GUI Application.
    
    This class provides functionality to manage the autostart of the
    WitsV3 Matrix GUI application.
    """
    
    def __init__(self):
        """Initialize the StartupManager."""
        # Set attributes
        self.app_name = "WitsV3Matrix"
        self.app_path = str(Path(__file__).resolve().parent / "main.py")
        self.python_path = sys.executable
        self.startup_command = f'"{self.python_path}" "{self.app_path}" --minimize'
        
        # Registry key
        self.registry_key = r"Software\Microsoft\Windows\CurrentVersion\Run"
    
    def is_autostart_enabled(self) -> bool:
        """
        Check if autostart is enabled.
        
        Returns:
            Whether autostart is enabled
        """
        try:
            # Open registry key
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, self.registry_key)
            
            # Get value
            value, _ = winreg.QueryValueEx(key, self.app_name)
            
            # Close key
            winreg.CloseKey(key)
            
            # Check if value matches
            return value == self.startup_command
        except FileNotFoundError:
            # Key not found
            return False
        except Exception as e:
            # Error
            logger.error(f"Error checking autostart: {e}")
            return False
    
    def enable_autostart(self) -> bool:
        """
        Enable autostart.
        
        Returns:
            Whether autostart was enabled successfully
        """
        try:
            # Open registry key
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, self.registry_key, 0, winreg.KEY_WRITE)
            
            # Set value
            winreg.SetValueEx(key, self.app_name, 0, winreg.REG_SZ, self.startup_command)
            
            # Close key
            winreg.CloseKey(key)
            
            logger.info("Autostart enabled")
            return True
        except Exception as e:
            # Error
            logger.error(f"Error enabling autostart: {e}")
            return False
    
    def disable_autostart(self) -> bool:
        """
        Disable autostart.
        
        Returns:
            Whether autostart was disabled successfully
        """
        try:
            # Open registry key
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, self.registry_key, 0, winreg.KEY_WRITE)
            
            # Delete value
            winreg.DeleteValue(key, self.app_name)
            
            # Close key
            winreg.CloseKey(key)
            
            logger.info("Autostart disabled")
            return True
        except FileNotFoundError:
            # Key not found, already disabled
            return True
        except Exception as e:
            # Error
            logger.error(f"Error disabling autostart: {e}")
            return False
