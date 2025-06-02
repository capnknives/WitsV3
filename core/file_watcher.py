"""
File watcher module for WitsV3.

This module provides file watching functionality for WitsV3, allowing the system
to automatically restart when Python files are changed.
"""

import os
import sys
import time
import logging
import asyncio
from typing import List, Callable, Optional, Set, Any
from pathlib import Path

# Import watchdog if available
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileSystemEvent, PatternMatchingEventHandler
    HAS_WATCHDOG = True
except ImportError:
    HAS_WATCHDOG = False
    logging.getLogger("WitsV3.FileWatcher").warning(
        "Watchdog package not available. Install with 'pip install watchdog' for file watching functionality."
    )

logger = logging.getLogger("WitsV3.FileWatcher")

class FileWatcher:
    """
    File watcher class for synchronous file watching.
    
    This class provides file watching functionality using the watchdog package.
    It is used by the GUI version of WitsV3.
    """
    
    def __init__(self, restart_callback: Callable[[], None], patterns: Optional[List[str]] = None):
        """
        Initialize the FileWatcher.
        
        Args:
            restart_callback: Callback function to call when a file change is detected
            patterns: List of file patterns to watch (e.g. ["*.py"])
        """
        if not HAS_WATCHDOG:
            raise ImportError("Watchdog package not available. Install with 'pip install watchdog'.")
        
        self.restart_callback = restart_callback
        self.patterns = patterns if patterns is not None else ["*.py"]
        self.observer = None
        self.handler = None
        self.watched_paths = set()
        self.debounce_time = 1.0  # Debounce time in seconds
        self.last_event_time = 0
        
    def start(self, paths: List[str]):
        """
        Start watching the specified paths.
        
        Args:
            paths: List of paths to watch
        """
        if self.observer:
            logger.warning("FileWatcher already started")
            return
        
        # Create observer
        self.observer = Observer()
        
        # Create event handler
        self.handler = PatternMatchingEventHandler(
            patterns=self.patterns,
            ignore_directories=True,
            case_sensitive=False
        )
        
        # Set event handlers
        self.handler.on_modified = self._on_file_changed
        self.handler.on_created = self._on_file_changed
        
        # Schedule paths
        for path in paths:
            if os.path.exists(path):
                self.observer.schedule(self.handler, path, recursive=True)
                self.watched_paths.add(path)
                logger.info(f"Watching path: {path}")
            else:
                logger.warning(f"Path does not exist: {path}")
          # Start observer
        self.observer.start()
        logger.info(f"FileWatcher started with patterns: {self.patterns}")
    
    def stop(self):
        """Stop watching."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
            self.handler = None
            self.watched_paths = set()
            logger.info("FileWatcher stopped")
    
    def _on_file_changed(self, event):  # : FileSystemEvent
        """
        Handle file change event.
        
        Args:
            event: File system event
        """
        # Debounce events
        current_time = time.time()
        if current_time - self.last_event_time < self.debounce_time:
            return
        
        self.last_event_time = current_time
        
        # Log event
        logger.info(f"File changed: {event.src_path}")
        
        # Call restart callback
        try:
            self.restart_callback()
        except Exception as e:
            logger.error(f"Error in restart callback: {e}")


class AsyncFileWatcher:
    """
    Asynchronous file watcher class.
    
    This class provides asynchronous file watching functionality using the watchdog package.
    It is used by the CLI version of WitsV3.
    """
    
    def __init__(self, restart_callback: Callable[[], None], patterns: Optional[List[str]] = None):
        """
        Initialize the AsyncFileWatcher.
        
        Args:
            restart_callback: Callback function to call when a file change is detected
            patterns: List of file patterns to watch (e.g. ["*.py"])
        """
        if not HAS_WATCHDOG:
            raise ImportError("Watchdog package not available. Install with 'pip install watchdog'.")
        
        self.restart_callback = restart_callback
        self.patterns = patterns if patterns is not None else ["*.py"]
        self.observer = None
        self.handler = None
        self.watched_paths = set()
        self.debounce_time = 1.0  # Debounce time in seconds
        self.last_event_time = 0
        self._running = False
        self._loop = None
    
    async def start(self, paths: List[str]):
        """
        Start watching the specified paths.
        
        Args:
            paths: List of paths to watch
        """
        if self.observer:
            logger.warning("AsyncFileWatcher already started")
            return
        
        # Store event loop
        self._loop = asyncio.get_event_loop()
        
        # Create observer
        self.observer = Observer()
        
        # Create event handler
        self.handler = PatternMatchingEventHandler(
            patterns=self.patterns,
            ignore_directories=True,
            case_sensitive=False
        )
        
        # Set event handlers
        self.handler.on_modified = self._on_file_changed
        self.handler.on_created = self._on_file_changed
        
        # Schedule paths
        for path in paths:
            if os.path.exists(path):
                self.observer.schedule(self.handler, path, recursive=True)
                self.watched_paths.add(path)
                logger.info(f"Watching path: {path}")
            else:
                logger.warning(f"Path does not exist: {path}")
          # Start observer
        self.observer.start()
        self._running = True
        logger.info(f"AsyncFileWatcher started with patterns: {self.patterns}")
    
    async def stop(self):
        """Stop watching."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
            self.handler = None
            self.watched_paths = set()
            self._running = False
            logger.info("AsyncFileWatcher stopped")
    
    def _on_file_changed(self, event):  # : FileSystemEvent
        """
        Handle file change event.
        
        Args:
            event: File system event
        """
        # Debounce events
        current_time = time.time()
        if current_time - self.last_event_time < self.debounce_time:
            return
        
        self.last_event_time = current_time
        
        # Log event
        logger.info(f"File changed: {event.src_path}")
        
        # Call restart callback
        try:
            self.restart_callback()
        except Exception as e:
            logger.error(f"Error in restart callback: {e}")


# Test function
def test_file_watcher():
    """Test the file watcher."""
    import tempfile
    import shutil
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create test file
        test_file = os.path.join(temp_dir, "test.py")
        with open(test_file, "w") as f:
            f.write("# Test file")
        
        # Create restart callback
        def restart_callback():
            print("Restart callback called")
        
        # Create file watcher
        watcher = FileWatcher(restart_callback)
        
        # Start watching
        watcher.start([temp_dir])
        
        # Modify file
        time.sleep(1)
        with open(test_file, "a") as f:
            f.write("\n# Modified")
        
        # Wait for event to be processed
        time.sleep(2)
        
        # Stop watching
        watcher.stop()
    
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test file watcher
    test_file_watcher()
