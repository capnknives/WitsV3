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
from core.config import load_config

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
        self.config = load_config()

    def _create_restricted_script(self, user_code: str) -> str:
        """Create a restricted script that conditionally blocks network/subprocess access."""

        # Reload configuration to get latest settings
        current_config = load_config()

        # Check configuration for network access
        allow_network = current_config.security.python_execution_network_access
        allow_subprocess = current_config.security.python_execution_subprocess_access

        if allow_network and allow_subprocess:
            # No restrictions - return user code directly
            return user_code

        restrictions = []

        if not allow_network:
            # Add network restrictions
            restrictions.append('''
# Network restriction wrapper
import socket
import urllib.request
import urllib.parse
import urllib.error
import http.client
import ssl

# Disable network modules by replacing them with restricted versions
original_socket = socket.socket

def restricted_socket(*args, **kwargs):
    raise PermissionError("Network access is not allowed in sandboxed environment")

def restricted_urlopen(*args, **kwargs):
    raise PermissionError("Network access is not allowed in sandboxed environment")

def restricted_create_connection(*args, **kwargs):
    raise PermissionError("Network access is not allowed in sandboxed environment")

def restricted_gethostbyname(*args, **kwargs):
    raise PermissionError("Network access is not allowed in sandboxed environment")

# Replace network functions
socket.socket = restricted_socket
socket.create_connection = restricted_create_connection
socket.gethostbyname = restricted_gethostbyname
socket.gethostbyname_ex = restricted_gethostbyname
socket.getaddrinfo = restricted_gethostbyname

urllib.request.urlopen = restricted_urlopen
http.client.HTTPConnection.__init__ = lambda self, *args, **kwargs: (_ for _ in ()).throw(PermissionError("Network access not allowed"))
http.client.HTTPSConnection.__init__ = lambda self, *args, **kwargs: (_ for _ in ()).throw(PermissionError("Network access not allowed"))
''')

        if not allow_subprocess:
            # Add subprocess restrictions
            restrictions.append('''
# Block subprocess for security
import subprocess
original_run = subprocess.run
original_popen = subprocess.Popen

def restricted_subprocess(*args, **kwargs):
    raise PermissionError("Subprocess execution is not allowed in sandboxed environment")

subprocess.run = restricted_subprocess
subprocess.Popen = restricted_subprocess
subprocess.call = restricted_subprocess
subprocess.check_call = restricted_subprocess
subprocess.check_output = restricted_subprocess
''')

        # Combine restrictions with user code
        restricted_script = '\n'.join(restrictions) + '\n\n# Execute user code\n' + user_code
        return restricted_script

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
                # Create restricted script
                restricted_code = self._create_restricted_script(code)

                # Write code to temporary file with UTF-8 encoding
                script_path = Path(temp_dir) / "script.py"
                with open(script_path, "w", encoding="utf-8") as f:
                    f.write(restricted_code)

                # Set up environment variables for sandbox
                env = os.environ.copy()
                env["PYTHONPATH"] = ""  # Prevent importing from system
                env["PYTHONUNBUFFERED"] = "1"  # Ensure output is not buffered
                env["PYTHONIOENCODING"] = "utf-8"  # Force UTF-8 encoding

                # Remove network-related environment variables
                network_vars = ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
                              "ftp_proxy", "FTP_PROXY", "no_proxy", "NO_PROXY"]
                for var in network_vars:
                    env.pop(var, None)

                # Additional security: Use isolated Python execution
                python_args = [
                    "python",
                    "-I",  # Isolated mode - don't add user site directory
                    "-S",  # Don't imply 'import site' on initialization
                    "-s",  # Don't add user site directory to sys.path
                    str(script_path)
                ]

                # Execute code with timeout
                process = await asyncio.create_subprocess_exec(
                    *python_args,
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

                    # Decode output with UTF-8 and error handling
                    try:
                        output = stdout.decode('utf-8', errors='replace').strip()
                        error = stderr.decode('utf-8', errors='replace').strip()
                    except UnicodeDecodeError:
                        # Fallback to latin-1 if UTF-8 fails
                        output = stdout.decode('latin-1', errors='replace').strip()
                        error = stderr.decode('latin-1', errors='replace').strip()

                    # Check output size and truncate if necessary
                    if len(output.encode('utf-8')) > self.max_output_size:
                        # Truncate safely without breaking UTF-8
                        truncated_bytes = output.encode('utf-8')[:self.max_output_size]
                        try:
                            output = truncated_bytes.decode('utf-8')
                        except UnicodeDecodeError:
                            # Find safe truncation point
                            for i in range(len(truncated_bytes)-1, -1, -1):
                                try:
                                    output = truncated_bytes[:i].decode('utf-8')
                                    break
                                except UnicodeDecodeError:
                                    continue
                        output += "\n... (output truncated)"

                    if len(error.encode('utf-8')) > self.max_output_size:
                        # Truncate error message safely
                        truncated_bytes = error.encode('utf-8')[:self.max_output_size]
                        try:
                            error = truncated_bytes.decode('utf-8')
                        except UnicodeDecodeError:
                            for i in range(len(truncated_bytes)-1, -1, -1):
                                try:
                                    error = truncated_bytes[:i].decode('utf-8')
                                    break
                                except UnicodeDecodeError:
                                    continue
                        error += "\n... (error truncated)"

                    return {
                        "success": process.returncode == 0,
                        "output": output,
                        "error": error,
                        "return_code": process.returncode
                    }

                except asyncio.TimeoutError:
                    try:
                        process.kill()
                        await process.wait()
                    except:
                        pass  # Process might already be dead
                    return {
                        "success": False,
                        "output": "",
                        "error": f"Execution timed out after {timeout or self.timeout} seconds",
                        "return_code": -1
                    }

        except UnicodeError as e:
            logger.error(f"Unicode encoding error in Python execution: {e}")
            return {
                "success": False,
                "output": "",
                "error": f"Unicode encoding error: {str(e)}",
                "return_code": -1
            }
        except Exception as e:
            logger.error(f"Error executing Python code: {e}")
            return {
                "success": False,
                "output": "",
                "error": f"Execution error: {str(e)}",
                "return_code": -1
            }

    def get_schema(self) -> Dict[str, Any]:
        """Get the tool's schema for LLM consumption."""
        # Dynamic description based on configuration
        current_config = load_config()
        network_status = "allowed" if current_config.security.python_execution_network_access else "blocked"
        subprocess_status = "allowed" if current_config.security.python_execution_subprocess_access else "blocked"

        description = f"Execute Python code in a sandboxed environment (network access: {network_status}, subprocess: {subprocess_status})"
        code_description = f"The Python code to execute (network: {network_status}, subprocess: {subprocess_status})"

        return {
            "name": "python_execute",
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": code_description
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
