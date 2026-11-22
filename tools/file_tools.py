"""
File-related tools for WitsV3.
This module contains tools for file operations.
"""

from datetime import datetime
from typing import Any

import aiofiles

from core.base_tool import BaseTool
from core.tool_context import tool_user_role


class FileReadTool(BaseTool):
    """Tool for reading files."""

    def __init__(self):
        super().__init__(
            name="read_file",
            description=(
                "Read a text file and return its contents. Use it when the owner names a "
                "file to read, quote or explain. Not for searching the owner's indexed "
                "documents (use document_search) or listing a folder (use list_directory). "
                'Examples: {"file_path": "README.md"}, {"file_path": "docs/notes.txt"}.'
            ),
        )

    async def execute(
        self, file_path: str, encoding: str = "utf-8", user_role: str = "owner"
    ) -> str:
        """
        Read file contents.

        Args:
            file_path: Path to the file to read
            encoding: File encoding (default: utf-8)
            user_role: owner | guest | RBAC role for read-root policy

        Returns:
            File contents as string
        """
        from core.config import load_config
        from core.path_access import check

        try:
            config = load_config()
            resolved = await check(file_path, "read", role=tool_user_role(user_role), config=config)
        except PermissionError as e:
            self.logger.error(str(e))
            return f"Error: {e}"

        try:
            if not resolved.exists():
                return f"Error: File {resolved} does not exist"

            if not resolved.is_file():
                return f"Error: {resolved} is not a file"

            async with aiofiles.open(resolved, encoding=encoding) as f:
                content = await f.read()

            self.logger.info(f"Read file: {resolved} ({len(content)} characters)")
            return content

        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {e}")
            return f"Error reading file: {str(e)}"

    def get_schema(self) -> dict[str, Any]:
        """Get tool schema."""
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path of the file to read, relative to the project "
                    "folder or absolute, such as 'README.md'",
                },
                "encoding": {
                    "type": "string",
                    "description": "Text encoding of the file; leave it out for the "
                    "usual 'utf-8'",
                    "default": "utf-8",
                },
            },
            "required": ["file_path"],
        }


class FileWriteTool(BaseTool):
    """Tool for writing files."""

    def __init__(self):
        super().__init__(
            name="write_file",
            description=(
                "Create or overwrite a text file with the given content. Call it whenever "
                "the owner asks to create, save or write a script, file, page or website: "
                "write the file, do not just print the code in your reply. Relative paths "
                "are fine and missing folders are created. Not for reading files. "
                'Examples: {"file_path": "scripts/rename.py", "content": "print(1)"}, '
                '{"file_path": "site/index.html", "content": "<h1>Hi</h1>"}.'
            ),
        )

    async def execute(
        self, file_path: str, content: str, encoding: str = "utf-8", mode: str = "w"
    ) -> str:
        """
        Write content to file. Relative paths land in the projects folder;
        absolute paths must pass core.path_access.check(write), so the repo is refused.

        Args:
            file_path: Path to the file to write
            content: Content to write
            encoding: File encoding (default: utf-8)
            mode: Write mode ('w' for overwrite, 'a' for append)

        Returns:
            Success message or error
        """
        from core.config import load_config
        from core.path_access import check, resolve_write_path

        try:
            config = load_config()
            resolved = await check(resolve_write_path(file_path, config), "write", config=config)
        except PermissionError as e:
            self.logger.error(str(e))
            return f"Error: {e}"

        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)

            async with aiofiles.open(resolved, mode=mode, encoding=encoding) as f:
                await f.write(content)

            action = "appended to" if mode == "a" else "written to"
            self.logger.info(f"Content {action} file: {resolved} ({len(content)} characters)")
            return f"Successfully {action} file: {resolved}"

        except PermissionError as e:
            error_msg = f"Permission error writing to file {resolved}: {e}"
            self.logger.error(error_msg)
            return f"Error: {error_msg}"
        except FileNotFoundError as e:
            error_msg = f"File not found error: {e}"
            self.logger.error(error_msg)
            return f"Error: {error_msg}"
        except Exception as e:
            self.logger.error(f"Error writing to file {resolved}: {e}")
            return f"Error writing file: {str(e)}"

    def get_schema(self) -> dict[str, Any]:
        """Get tool schema."""
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path of the file to write; relative paths such as "
                    "'scripts/backup.sh' are fine",
                },
                "content": {
                    "type": "string",
                    "description": "The full text to put in the file, such as the "
                    "complete script or page",
                },
                "encoding": {
                    "type": "string",
                    "description": "Text encoding to write with; leave it out for the "
                    "usual 'utf-8'",
                    "default": "utf-8",
                },
                "mode": {
                    "type": "string",
                    "description": "'w' replaces the file (the default), 'a' appends " "to its end",
                    "default": "w",
                },
            },
            "required": ["file_path", "content"],
        }


class ListDirectoryTool(BaseTool):
    """Tool for listing directory contents."""

    def __init__(self):
        super().__init__(
            name="list_directory",
            description=(
                "List the files and folders in a directory with their sizes. Use it when "
                "the owner asks what is in a folder or you need a file's exact name. Not "
                "for reading a file's contents (use read_file). "
                'Examples: {"directory_path": "docs"}, {"directory_path": "."}.'
            ),
        )

    async def execute(
        self, directory_path: str, include_hidden: bool = False, user_role: str = "owner"
    ) -> str:
        """
        List directory contents.

        Args:
            directory_path: Path to the directory to list
            include_hidden: Whether to include hidden files/directories
            user_role: owner | guest | RBAC role for read-root policy

        Returns:
            Directory listing as formatted string
        """
        from core.config import load_config
        from core.path_access import check

        try:
            config = load_config()
            directory_path = await check(
                directory_path, "read", role=tool_user_role(user_role), config=config
            )
        except PermissionError as e:
            self.logger.error(str(e))
            return f"Error: {e}"

        try:
            if not directory_path.exists():
                return f"Error: Directory {directory_path} does not exist"

            if not directory_path.is_dir():
                return f"Error: {directory_path} is not a directory"

            items = []
            for item in directory_path.iterdir():
                if not include_hidden and item.name.startswith("."):
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

    def get_schema(self) -> dict[str, Any]:
        """Get tool schema."""
        return {
            "type": "object",
            "properties": {
                "directory_path": {
                    "type": "string",
                    "description": "Path of the folder to list, relative to the project "
                    "folder, such as 'docs' or '.'",
                },
                "include_hidden": {
                    "type": "boolean",
                    "description": "true to also list hidden entries whose names start "
                    "with a dot",
                    "default": False,
                },
            },
            "required": ["directory_path"],
        }


class DateTimeTool(BaseTool):
    """Tool for date and time operations."""

    def __init__(self):
        super().__init__(
            name="datetime",
            description=(
                "Get the current local date and time. Use it whenever the answer depends "
                "on today's date, the day of the week or the time now. Not for date maths "
                'on dates the owner gives. Examples: {}, {"format_string": "%A %d %B %Y"}.'
            ),
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

    def get_schema(self) -> dict[str, Any]:
        """Get tool schema."""
        return {
            "type": "object",
            "properties": {
                "format_string": {
                    "type": "string",
                    "description": "Python strftime format; leave it out for "
                    "'%Y-%m-%d %H:%M:%S'",
                    "default": "%Y-%m-%d %H:%M:%S",
                }
            },
        }
