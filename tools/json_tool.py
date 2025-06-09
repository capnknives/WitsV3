"""
JSON Manipulation Tool for WitsV3.
Provides operations for manipulating JSON data.
"""

import logging
import json
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from core.base_tool import BaseTool
from core.schemas import ToolCall

logger = logging.getLogger(__name__)

class JSONTool(BaseTool):
    """Tool for manipulating JSON data."""

    def __init__(self):
        """Initialize the JSON tool."""
        super().__init__(
            name="json_manipulate",
            description="Manipulate JSON data with various operations like get, set, merge, etc."
        )

    async def execute(
        self,
        operation: str,
        data: Union[str, Dict[str, Any], List[Any]],
        path: Optional[str] = None,
        value: Optional[Any] = None,
        file_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a JSON manipulation operation.

        Args:
            operation: The operation to perform (get, set, merge, validate, format)
            data: The JSON data to operate on
            path: Optional JSONPath for get/set operations
            value: Optional value for set operations
            file_path: Optional file path for read/write operations

        Returns:
            Dictionary containing the operation result
        """
        try:
            # Parse data if it's a string (except for validation and file operations which need the raw string)
            if isinstance(data, str) and operation not in ["validate", "read_file"]:
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    return {
                        "success": False,
                        "error": "Invalid JSON string"
                    }

            if operation == "get":
                return await self._get_value(data, path)
            elif operation == "set":
                return await self._set_value(data, path, value)
            elif operation == "merge":
                return await self._merge_data(data, value)
            elif operation == "validate":
                return await self._validate_json(data)
            elif operation == "format":
                return await self._format_json(data)
            elif operation == "read_file":
                return await self._read_json_file(file_path)
            elif operation == "write_file":
                return await self._write_json_file(data, file_path)
            else:
                return {
                    "success": False,
                    "error": f"Unknown operation: {operation}"
                }

        except Exception as e:
            logger.error(f"Error in JSON operation: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _get_value(self, data: Any, path: Optional[str]) -> Dict[str, Any]:
        """Get a value from JSON data using a path."""
        if not path:
            return {
                "success": True,
                "value": data
            }

        try:
            # Handle array notation like values[0]
            current = data
            parts = []

            # Split path and handle array indices
            for part in path.split("."):
                if "[" in part and "]" in part:
                    # Handle array access like "values[0]"
                    key, index_part = part.split("[", 1)
                    index = index_part.rstrip("]")
                    if key:
                        parts.append(key)
                    parts.append(f"[{index}]")
                else:
                    parts.append(part)

            for part in parts:
                if part.startswith("[") and part.endswith("]"):
                    # Array index
                    try:
                        index = int(part[1:-1])
                        if isinstance(current, list):
                            current = current[index]
                        else:
                            return {
                                "success": False,
                                "error": f"Cannot index non-array with: {part}"
                            }
                    except (ValueError, IndexError):
                        return {
                            "success": False,
                            "error": f"Invalid array index: {part}"
                        }
                else:
                    # Dictionary key
                    if isinstance(current, dict):
                        current = current.get(part)
                    else:
                        return {
                            "success": False,
                            "error": f"Cannot access key '{part}' on non-object"
                        }

            return {
                "success": True,
                "value": current
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def _set_value(self, data: Any, path: Optional[str], value: Any) -> Dict[str, Any]:
        """Set a value in JSON data using a path."""
        if not path:
            return {
                "success": True,
                "value": value
            }

        try:
            # Create a copy to avoid modifying the original
            result = json.loads(json.dumps(data))
            current = result
            parts = path.split(".")

            # Navigate to the parent of the target
            for key in parts[:-1]:
                if isinstance(current, dict):
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                elif isinstance(current, list):
                    try:
                        current = current[int(key)]
                    except (ValueError, IndexError):
                        return {
                            "success": False,
                            "error": f"Invalid array index: {key}"
                        }
                else:
                    return {
                        "success": False,
                        "error": f"Invalid path: {path}"
                    }

            # Set the value
            last_key = parts[-1]
            if isinstance(current, dict):
                current[last_key] = value
            elif isinstance(current, list):
                try:
                    current[int(last_key)] = value
                except (ValueError, IndexError):
                    return {
                        "success": False,
                        "error": f"Invalid array index: {last_key}"
                    }
            else:
                return {
                    "success": False,
                    "error": f"Invalid path: {path}"
                }

            return {
                "success": True,
                "data": json.dumps(result)
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def _merge_data(self, data1: Any, data2: Any) -> Dict[str, Any]:
        """Merge two JSON data structures."""
        try:
            if isinstance(data1, dict) and isinstance(data2, dict):
                result = data1.copy()
                for key, value in data2.items():
                    if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                        result[key] = (await self._merge_data(result[key], value))["value"]
                    else:
                        result[key] = value
                return {
                    "success": True,
                    "data": json.dumps(result)
                }
            elif isinstance(data1, list) and isinstance(data2, list):
                return {
                    "success": True,
                    "value": data1 + data2
                }
            else:
                return {
                    "success": True,
                    "value": data2
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def _validate_json(self, data: Any) -> Dict[str, Any]:
        """Validate JSON data structure."""
        try:
            if isinstance(data, str):
                # Try to parse the string as JSON
                json.loads(data)
            else:
                # Try to serialize and deserialize to validate
                json_str = json.dumps(data)
                json.loads(json_str)
            return {
                "success": True,
                "is_valid": True
            }
        except Exception as e:
            return {
                "success": True,
                "is_valid": False,
                "error": str(e)
            }

    async def _format_json(self, data: Any) -> Dict[str, Any]:
        """Format JSON data with proper indentation."""
        try:
            formatted = json.dumps(data, indent=4)
            return {
                "success": True,
                "formatted": formatted
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def _read_json_file(self, file_path: Optional[str]) -> Dict[str, Any]:
        """Read JSON data from a file."""
        if not file_path:
            return {
                "success": False,
                "error": "No file path provided"
            }

        try:
            path = Path(file_path)
            if not path.exists():
                return {
                    "success": False,
                    "error": f"File not found: {file_path}"
                }

            with open(path, "r") as f:
                data = json.load(f)
                return {
                    "success": True,
                    "value": data
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    
    
    
    async def _write_json_file(self, data: Any, file_path: Optional[str]) -> Dict[str, Any]:
        """Write JSON data to a file."""
        if not file_path:
            return {
                "success": False,
                "error": "No file path provided"
            }

        try:
            path = Path(file_path)
            abs_path = path.resolve()

            # Security check - only write within project directory
            if not str(abs_path).startswith(os.path.abspath(".")):
                return {
                    "success": False,
                    "error": f"Security error: Cannot write outside project directory: {file_path}"
                }

            # Create parent directories if they don't exist
            path.parent.mkdir(parents=True, exist_ok=True)

            self.logger.info(f"Writing JSON data to {abs_path}")

            with open(abs_path, "w") as f:
                json.dump(data, f, indent=2)
                return {
                    "success": True,
                    "message": f"Data written to {file_path}"
                }

        except PermissionError as e:
            error_msg = f"Permission error writing to file {file_path}: {e}"
            self.logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg
            }
        except Exception as e:
            self.logger.error(f"Error writing JSON to {file_path}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    def get_schema(self) -> Dict[str, Any]:
        """Get the tool's schema for LLM consumption."""
        return {
            "name": "json_manipulate",
            "description": "Manipulate JSON data with various operations",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "The operation to perform (get, set, merge, validate, format, read_file, write_file)",
                        "enum": ["get", "set", "merge", "validate", "format", "read_file", "write_file"]
                    },
                    "data": {
                        "type": ["string", "object", "array"],
                        "description": "The JSON data to operate on"
                    },
                    "path": {
                        "type": "string",
                        "description": "Optional JSONPath for get/set operations"
                    },
                    "value": {
                        "type": ["string", "object", "array", "number", "boolean"],
                        "description": "Optional value for set/merge operations"
                    },
                    "file_path": {
                        "type": "string",
                        "description": "Optional file path for read/write operations"
                    }
                },
                "required": ["operation", "data"]
            }
        }
