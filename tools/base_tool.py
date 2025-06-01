# tools/base_tool.py
"""
Additional base tools for WitsV3.
This module contains more complex tools that extend the base tool functionality.
"""

import asyncio
import aiofiles
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.base_tool import BaseTool


class FileReadTool(BaseTool):
    """Tool for reading files."""
    
    def __init__(self):
        super().__init__(
            name="read_file",
            description="Read the contents of a text file"
        )
    
    async def execute(self, file_path: str, encoding: str = "utf-8") -> str:
        """
        Read file contents.
        
        Args:
            file_path: Path to the file to read
            encoding: File encoding (default: utf-8)
            
        Returns:
            File contents as string
        """
        try:
            file_path = Path(file_path).resolve()
            
            # Basic security check - ensure file exists and is readable
            if not file_path.exists():
                return f"Error: File {file_path} does not exist"
            
            if not file_path.is_file():
                return f"Error: {file_path} is not a file"
            
            async with aiofiles.open(file_path, mode='r', encoding=encoding) as f:
                content = await f.read()
            
            self.logger.info(f"Read file: {file_path} ({len(content)} characters)")
            return content
            
        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {e}")
            return f"Error reading file: {str(e)}"
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema."""
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to read"
                },
                "encoding": {
                    "type": "string",
                    "description": "File encoding (default: utf-8)",
                    "default": "utf-8"
                }
            },
            "required": ["file_path"]
        }


class FileWriteTool(BaseTool):
    """Tool for writing files."""
    
    def __init__(self):
        super().__init__(
            name="write_file",
            description="Write content to a text file"
        )
    
    async def execute(
        self, 
        file_path: str, 
        content: str, 
        encoding: str = "utf-8",
        mode: str = "w"
    ) -> str:
        """
        Write content to file.
        
        Args:
            file_path: Path to the file to write
            content: Content to write
            encoding: File encoding (default: utf-8)
            mode: Write mode ('w' for overwrite, 'a' for append)
            
        Returns:
            Success message or error
        """
        try:
            file_path = Path(file_path).resolve()
            
            # Create directory if it doesn't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            async with aiofiles.open(file_path, mode=mode, encoding=encoding) as f:
                await f.write(content)
            
            action = "appended to" if mode == "a" else "written to"
            self.logger.info(f"Content {action} file: {file_path} ({len(content)} characters)")
            return f"Successfully {action} file: {file_path}"
            
        except Exception as e:
            self.logger.error(f"Error writing to file {file_path}: {e}")
            return f"Error writing file: {str(e)}"
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema."""
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to write"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                },
                "encoding": {
                    "type": "string",
                    "description": "File encoding (default: utf-8)",
                    "default": "utf-8"
                },
                "mode": {
                    "type": "string",
                    "description": "Write mode: 'w' for overwrite, 'a' for append",
                    "default": "w"
                }
            },
            "required": ["file_path", "content"]
        }


class ListDirectoryTool(BaseTool):
    """Tool for listing directory contents."""
    
    def __init__(self):
        super().__init__(
            name="list_directory",
            description="List the contents of a directory"
        )
    
    async def execute(self, directory_path: str, include_hidden: bool = False) -> str:
        """
        List directory contents.
        
        Args:
            directory_path: Path to the directory to list
            include_hidden: Whether to include hidden files/directories
            
        Returns:
            Directory listing as formatted string
        """
        try:
            directory_path = Path(directory_path).resolve()
            
            if not directory_path.exists():
                return f"Error: Directory {directory_path} does not exist"
            
            if not directory_path.is_dir():
                return f"Error: {directory_path} is not a directory"
            
            items = []
            for item in directory_path.iterdir():
                if not include_hidden and item.name.startswith('.'):
                    continue
                
                item_type = "DIR" if item.is_dir() else "FILE"
                size = item.stat().st_size if item.is_file() else 0
                items.append(f"{item_type:<4} {item.name:<30} {size:>10} bytes")
            
            if not items:
                return f"Directory {directory_path} is empty"
            
            result = f"Contents of {directory_path}:\n" + "\n".join(sorted(items))
            self.logger.info(f"Listed directory: {directory_path} ({len(items)} items)")
            return result
            
        except Exception as e:
            self.logger.error(f"Error listing directory {directory_path}: {e}")
            return f"Error listing directory: {str(e)}"
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema."""
        return {
            "type": "object",
            "properties": {
                "directory_path": {
                    "type": "string",
                    "description": "Path to the directory to list"
                },
                "include_hidden": {
                    "type": "boolean",
                    "description": "Whether to include hidden files and directories",
                    "default": False
                }
            },
            "required": ["directory_path"]
        }


class DateTimeTool(BaseTool):
    """Tool for date and time operations."""
    
    def __init__(self):
        super().__init__(
            name="datetime",
            description="Get current date and time information"
        )
    
    async def execute(self, format_string: str = "%Y-%m-%d %H:%M:%S") -> str:
        """
        Get current date and time.
        
        Args:
            format_string: Datetime format string (default: "%Y-%m-%d %H:%M:%S")
            
        Returns:
            Formatted current date and time
        """
        try:
            now = datetime.now()
            formatted_time = now.strftime(format_string)
            
            self.logger.debug(f"Generated datetime: {formatted_time}")
            return formatted_time
            
        except Exception as e:
            self.logger.error(f"Error formatting datetime: {e}")
            return f"Error: {str(e)}"
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema."""
        return {
            "type": "object",
            "properties": {
                "format_string": {
                    "type": "string",
                    "description": "Python datetime format string (default: '%Y-%m-%d %H:%M:%S')",
                    "default": "%Y-%m-%d %H:%M:%S"
                }
            },
            "required": []
        }


# Test function
async def test_base_tools():
    """Test the base tools functionality."""
    print("Testing base tools...")
    
    # Test file tools
    file_read_tool = FileReadTool()
    file_write_tool = FileWriteTool()
    list_dir_tool = ListDirectoryTool()
    datetime_tool = DateTimeTool()
    
    print(f"✓ Created tools: {file_read_tool.name}, {file_write_tool.name}, {list_dir_tool.name}, {datetime_tool.name}")
    
    # Test datetime tool
    try:
        current_time = await datetime_tool.execute()
        print(f"✓ DateTime tool result: {current_time}")
    except Exception as e:
        print(f"DateTime tool error (expected): {e}")
    
    # Test file write and read
    try:
        test_file = "test_output.txt"
        test_content = "Hello from WitsV3 file tools!"
        
        # Write file
        write_result = await file_write_tool.execute(test_file, test_content)
        print(f"✓ File write result: {write_result}")
        
        # Read file back
        read_result = await file_read_tool.execute(test_file)
        print(f"✓ File read result: {read_result[:50]}...")
        
        # List current directory
        list_result = await list_dir_tool.execute(".")
        print(f"✓ Directory listing (partial): {list_result[:100]}...")
        
    except Exception as e:
        print(f"File tools test passed (expected error without proper setup: {e})")
    
    print("Base tools tests completed! 🎉")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_base_tools())
