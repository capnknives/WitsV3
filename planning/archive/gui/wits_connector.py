"""
WitsV3 Connector for Matrix GUI Application.

This module provides a connector to the WitsV3 system for the Matrix GUI application.
It handles communication with the WitsV3 system and provides callbacks for events.
"""

import os
import sys
import logging
import asyncio
import subprocess
from typing import Dict, Any, Callable, List, Optional, AsyncGenerator
from pathlib import Path

# Add parent directory to path to import WitsV3 modules
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import WitsV3 modules
from core.schemas import StreamData
from core.config import WitsV3Config, load_config
from agents.wits_control_center_agent import WitsControlCenterAgent

# Import file watcher if available
try:
    from core.file_watcher import FileWatcher
    HAS_FILE_WATCHER = True
except ImportError:
    HAS_FILE_WATCHER = False
    logging.getLogger("WitsV3.GUI.WitsConnector").warning(
        "File watcher not available. Install watchdog package for auto-restart functionality."
    )

logger = logging.getLogger("WitsV3.GUI.WitsConnector")

# Flag to indicate if this is a restart
IS_RESTART = "--restart" in sys.argv

class WitsConnector:
    """
    Connector to the WitsV3 system.
    
    This class handles communication with the WitsV3 system and provides
    callbacks for events.
    """
    
    def __init__(self):
        """Initialize the WitsConnector."""
        # Set attributes
        self.initialized = False
        self.config = None
        self.wits_system = None
        self.session_id = None
        self.file_watcher = None
        
        # Callbacks
        self.callbacks = {
            "on_thinking": [],
            "on_action": [],
            "on_observation": [],
            "on_result": [],
            "on_error": [],
            "on_status_change": [],
            "on_model_change": [],
            "on_restart": []
        }
    
    def register_callback(self, event_type: str, callback: Callable):
        """
        Register a callback for an event.
        
        Args:
            event_type: Type of event to register for
            callback: Callback function
        """
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
    
    def unregister_callback(self, event_type: str, callback: Callable):
        """
        Unregister a callback for an event.
        
        Args:
            event_type: Type of event to unregister for
            callback: Callback function
        """
        if event_type in self.callbacks and callback in self.callbacks[event_type]:
            self.callbacks[event_type].remove(callback)
    
    def trigger_callback(self, event_type: str, content: Any):
        """
        Trigger callbacks for an event.
        
        Args:
            event_type: Type of event to trigger
            content: Event content
        """
        if event_type in self.callbacks:
            for callback in self.callbacks[event_type]:
                try:
                    callback(content)
                except Exception as e:
                    logger.error(f"Error in callback for {event_type}: {e}")
    
    async def initialize(self):
        """Initialize the WitsConnector."""
        try:
            # Try to load the real WitsV3 system
            try:
                # Load config
                self.config = WitsV3Config()
                
                # Create LLM interface
                from core.llm_interface import get_llm_interface
                llm_interface = get_llm_interface(self.config)
                
                # Create WitsControlCenterAgent
                self.wits_system = WitsControlCenterAgent(
                    agent_name="WitsGUI",
                    config=self.config,
                    llm_interface=llm_interface
                )
            except Exception as e:
                # Log the error but continue with a mock implementation
                logger.warning(f"Failed to initialize real WitsV3 system: {e}")
                logger.warning("Using mock implementation instead")
                
                # Create a mock implementation
                from core.schemas import StreamData
                
                class MockWitsSystem:
                    def get_model_name(self):
                        return "mock-model"
                    
                    async def run(self, input_data, **kwargs):
                        # Simulate thinking
                        yield StreamData(
                            type="thinking",
                            content="Thinking about your request...",
                            source="MockWitsSystem"
                        )
                        
                        # Simulate action
                        yield StreamData(
                            type="action",
                            content="Processing your request",
                            source="MockWitsSystem"
                        )
                        
                        # Simulate result
                        yield StreamData(
                            type="result",
                            content=f"This is a mock response to: {input_data}",
                            source="MockWitsSystem"
                        )
                
                self.wits_system = MockWitsSystem()
                self.config = None
            
            # Create a unique session ID
            import uuid
            self.session_id = str(uuid.uuid4())
            
            # Set initialized flag
            self.initialized = True
            
            # Initialize file watcher if available
            if HAS_FILE_WATCHER and self.config and self.config.auto_restart_on_file_change:
                await self._initialize_file_watcher()
            
            # Trigger status change callback
            self.trigger_callback("on_status_change", "Initialized")
            
            # Trigger model change callback
            system_status = self.get_system_status()
            if "llm_info" in system_status and "model" in system_status["llm_info"]:
                self.trigger_callback("on_model_change", system_status["llm_info"]["model"])
            
            logger.info("WitsConnector initialized")
        except Exception as e:
            logger.error(f"Error initializing WitsConnector: {e}")
            self.trigger_callback("on_error", f"Error initializing WitsConnector: {e}")
            raise
    
    async def _initialize_file_watcher(self):
        """Initialize the file watcher for auto-restart functionality."""
        try:
            # Create file watcher
            self.file_watcher = FileWatcher(
                restart_callback=self._restart_system,
                patterns=["*.py"]  # Watch Python files
            )
            
            # Start file watcher
            watch_paths = [
                os.path.join(os.getcwd(), "agents"),
                os.path.join(os.getcwd(), "core"),
                os.path.join(os.getcwd(), "tools"),
                os.path.join(os.getcwd(), "gui")
            ]
            self.file_watcher.start(watch_paths)
            
            logger.info(f"File watcher initialized and watching for changes in Python files")
        except Exception as e:
            logger.error(f"Failed to initialize file watcher: {e}")
    
    def _restart_system(self):
        """Restart the system when a file changes."""
        logger.info("Restarting WitsV3 GUI system due to file changes...")
        
        try:
            # Trigger restart callback
            self.trigger_callback("on_restart", "Restarting due to file changes")
            
            # Get the current script path
            script_path = sys.argv[0]
            
            # Build the restart command
            cmd = [sys.executable, script_path, "--restart"]
            
            # Add any other command line arguments
            for arg in sys.argv[1:]:
                if arg != "--restart":  # Avoid duplicate restart flags
                    cmd.append(arg)
            
            # Start the new process
            subprocess.Popen(cmd)
            
            # Exit the current process
            logger.info("New WitsV3 GUI process started, shutting down current process...")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Failed to restart system: {e}")
    
    async def shutdown(self):
        """Shutdown the WitsConnector."""
        if self.initialized:
            try:
                # Stop file watcher if running
                if self.file_watcher:
                    self.file_watcher.stop()
                    logger.info("File watcher shutdown complete")
                
                # Reset attributes
                self.initialized = False
                self.config = None
                self.wits_system = None
                self.session_id = None
                
                # Trigger status change callback
                self.trigger_callback("on_status_change", "Shutdown")
                
                logger.info("WitsConnector shutdown")
            except Exception as e:
                logger.error(f"Error shutting down WitsConnector: {e}")
                self.trigger_callback("on_error", f"Error shutting down WitsConnector: {e}")
                raise
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get the system status.
        
        Returns:
            System status
        """
        if not self.initialized:
            return {"status": "Not initialized"}
        
        try:
            # Get basic system status
            model = self.wits_system.get_model_name() if self.wits_system else "unknown"
            return {
                "status": "Initialized",
                "llm_info": {
                    "model": model
                }
            }
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            self.trigger_callback("on_error", f"Error getting system status: {e}")
            return {"status": "Error", "error": str(e)}
    
    async def process_message(self, message: str) -> str:
        """
        Process a message.
        
        Args:
            message: Message to process
            
        Returns:
            Response message
        """
        if not self.initialized:
            raise RuntimeError("WitsConnector not initialized")
        
        if not self.wits_system:
            raise RuntimeError("WitsSystem not initialized")
        
        try:
            # Process message using run method
            result = ""
            async for stream_data in self.wits_system.run(message, session_id=self.session_id):
                if stream_data.type == "result":
                    result = stream_data.content
            
            # Return response
            return result
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self.trigger_callback("on_error", f"Error processing message: {e}")
            raise
    
    async def process_message_stream(self, message: str) -> AsyncGenerator[StreamData, None]:
        """
        Process a message with streaming.
        
        Args:
            message: Message to process
            
        Yields:
            Stream data
        """
        if not self.initialized:
            raise RuntimeError("WitsConnector not initialized")
        
        if not self.wits_system:
            raise RuntimeError("WitsSystem not initialized")
        
        try:
            # Process message with streaming using run method
            async for stream_data in self.wits_system.run(message, session_id=self.session_id):
                # Trigger callbacks based on stream data type
                if stream_data.type == "thinking":
                    self.trigger_callback("on_thinking", stream_data.content)
                elif stream_data.type == "action":
                    self.trigger_callback("on_action", stream_data.content)
                elif stream_data.type == "observation":
                    self.trigger_callback("on_observation", stream_data.content)
                elif stream_data.type == "result":
                    self.trigger_callback("on_result", stream_data.content)
                elif stream_data.type == "error":
                    self.trigger_callback("on_error", stream_data.content)
                
                # Yield stream data
                yield stream_data
        except Exception as e:
            logger.error(f"Error processing message stream: {e}")
            self.trigger_callback("on_error", f"Error processing message stream: {e}")
            raise
