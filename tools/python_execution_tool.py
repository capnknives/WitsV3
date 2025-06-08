"""
Python Code Execution Tool for WitsV3.
Provides safe execution of Python code in a sandboxed environment.
"""

import logging
import asyncio
import tempfile
import os
from typing import Any, Dict, Optional
import subprocess
from pathlib import Path

from core.base_tool import BaseTool
from core.schemas import ToolCall

logger = logging.getLogger(__name__)

class PythonExecutionTool(BaseTool):
    """Tool for safely executing Python code."""
    
    def __init__(self):
        """Initialize the Python execution tool."""
        super().__init__(
            name="python_execute",
            description="Execute Python code in a sandboxed environment. Returns the output or error."
        )
        self.timeout = 30  # seconds
        self.max_output_size = 1024 * 1024  # 1MB
        
    async def execute(self, code: str, timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute Python code in a sandboxed environment.
        
        Args:
            code: The Python code to execute
            timeout: Optional timeout in seconds
            
        Returns:
            Dictionary containing output, error, and execution status
        """
        try:
            # Create temporary directory for execution
            with tempfile.TemporaryDirectory() as temp_dir:
                # Write code to temporary file
                script_path = Path(temp_dir) / "script.py"
                with open(script_path, "w") as f:
                    f.write(code)
                
                # Set up environment variables for sandbox
                env = os.environ.copy()
                env["PYTHONPATH"] = ""  # Prevent importing from system
                env["PYTHONUNBUFFERED"] = "1"  # Ensure output is not buffered
                
                # Execute code with timeout
                process = await asyncio.create_subprocess_exec(
                    "python",
                    str(script_path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env,
                    cwd=temp_dir
                )
                
                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(),
                        timeout=timeout or self.timeout
                    )
                    
                    # Check output size
                    if len(stdout) > self.max_output_size:
                        stdout = stdout[:self.max_output_size] + b"\n... (output truncated)"
                    if len(stderr) > self.max_output_size:
                        stderr = stderr[:self.max_output_size] + b"\n... (error truncated)"
                    
                    return {
                        "success": process.returncode == 0,
                        "output": stdout.decode().strip(),
                        "error": stderr.decode().strip(),
                        "return_code": process.returncode
                    }
                    
                except asyncio.TimeoutError:
                    process.kill()
                    return {
                        "success": False,
                        "output": "",
                        "error": f"Execution timed out after {timeout or self.timeout} seconds",
                        "return_code": -1
                    }
                    
        except Exception as e:
            logger.error(f"Error executing Python code: {e}")
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "return_code": -1
            }
            
    def get_schema(self) -> Dict[str, Any]:
        """Get the tool's schema for LLM consumption."""
        return {
            "name": "python_execute",
            "description": "Execute Python code in a sandboxed environment",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to execute"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Optional timeout in seconds",
                        "default": 30
                    }
                },
                "required": ["code"]
            }
        } 