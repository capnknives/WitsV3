Python Execution Tool
==================

The Python Execution Tool provides a safe way to execute Python code in a sandboxed environment.

Overview
-------

The Python Execution Tool allows you to execute Python code safely with controlled access to system resources. It provides a sandboxed environment with configurable timeouts and resource limits.

Installation
----------

The Python Execution Tool is included in the core WitsV3 package. No additional installation is required.

Usage
----

Basic Usage
~~~~~~~~~

.. code-block:: python

   from witsv3.tools import PythonExecutionTool

   # Initialize the tool
   tool = PythonExecutionTool()

   # Execute Python code
   result = await tool.execute(
       code="print('Hello, World!')"
   )

Parameters
~~~~~~~~~

* **code** (str, required): The Python code to execute
* **timeout** (int, optional): Maximum execution time in seconds (default: 30)
* **max_output_size** (int, optional): Maximum output size in bytes (default: 1MB)

Return Value
~~~~~~~~~~

The tool returns a dictionary with the following structure:

.. code-block:: python

   {
       "success": True,
       "output": "Hello, World!",
       "error": None,
       "return_code": 0
   }

Error Handling
~~~~~~~~~~~~

The tool handles various error cases:

* Syntax errors
* Runtime errors
* Timeout errors
* Resource limit exceeded
* Security violations

Example error response:

.. code-block:: python

   {
       "success": False,
       "output": "",
       "error": "Error message",
       "return_code": 1
   }

Advanced Usage
-----------

Custom Timeout
~~~~~~~~~~~~

Set a custom timeout for code execution:

.. code-block:: python

   tool = PythonExecutionTool()
   result = await tool.execute(
       code="import time; time.sleep(10)",
       timeout=15
   )

Output Size Limit
~~~~~~~~~~~~~~

Configure maximum output size:

.. code-block:: python

   tool = PythonExecutionTool()
   result = await tool.execute(
       code="print('x' * 1000000)",
       max_output_size=1024  # 1KB
   )

Environment Variables
~~~~~~~~~~~~~~~~~~

Set environment variables for the execution:

.. code-block:: python

   tool = PythonExecutionTool()
   tool.env = {
       "PYTHONPATH": "/custom/path",
       "CUSTOM_VAR": "value"
   }

Error Handling Example
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   try:
       result = await tool.execute(code="1/0")
       if result["success"]:
           print(f"Output: {result['output']}")
       else:
           print(f"Error: {result['error']}")
   except Exception as e:
       print(f"Unexpected error: {e}")

Best Practices
-----------

1. **Code Safety**
   * Validate input code
   * Restrict dangerous operations
   * Implement resource limits

2. **Error Handling**
   * Catch specific exceptions
   * Provide clear error messages
   * Log execution details

3. **Resource Management**
   * Set appropriate timeouts
   * Limit memory usage
   * Clean up resources

4. **Security**
   * Restrict file system access
   * Limit network access
   * Control environment variables

Example Use Cases
--------------

1. **Code Evaluation**
   * Test code snippets
   * Validate algorithms
   * Debug issues

2. **Dynamic Execution**
   * Run user-provided code
   * Execute plugins
   * Process data

3. **Learning Environment**
   * Run tutorials
   * Test examples
   * Practice coding

API Reference
-----------

.. py:class:: PythonExecutionTool

   Python code execution tool with sandboxing.

   .. py:method:: execute(code: str, timeout: int = 30, max_output_size: int = 1048576) -> dict

      Execute Python code in a sandboxed environment.

      :param code: Python code to execute
      :param timeout: Maximum execution time in seconds
      :param max_output_size: Maximum output size in bytes
      :return: Dictionary containing execution results

   .. py:method:: get_schema() -> dict

      Get the tool's schema for LLM consumption.

      :return: Dictionary containing tool schema

Limitations
---------

* Restricted system access
* Limited resource usage
* No persistent storage
* No network access
* No GUI operations

Troubleshooting
------------

Common Issues
~~~~~~~~~~~

1. **Timeout Errors**
   * Optimize code performance
   * Increase timeout value
   * Check for infinite loops

2. **Memory Errors**
   * Reduce data size
   * Optimize memory usage
   * Increase memory limit

3. **Security Violations**
   * Check code permissions
   * Verify resource access
   * Review security settings

Support
------

 