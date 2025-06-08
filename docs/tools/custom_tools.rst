Custom Tools
===========

Learn how to create and integrate custom tools in WitsV3.

Overview
-------

WitsV3 provides a flexible framework for creating custom tools. This guide explains how to create, test, and integrate your own tools into the system.

Creating a Custom Tool
-------------------

Basic Structure
~~~~~~~~~~~~~

All custom tools must inherit from the BaseTool class:

.. code-block:: python

   from witsv3.tools import BaseTool

   class MyCustomTool(BaseTool):
       def __init__(self):
           super().__init__(
               name="my_custom_tool",
               description="A custom tool for specific tasks"
           )

       async def execute(self, **kwargs):
           # Your tool logic here
           return {"success": True, "result": "Custom tool result"}

Required Methods
~~~~~~~~~~~~~

1. **__init__**
   * Set tool name and description
   * Initialize any required resources
   * Configure tool settings

2. **execute**
   * Implement the main tool logic
   * Handle input parameters
   * Return results in standard format

3. **get_schema**
   * Define tool parameters
   * Specify required fields
   * Document parameter types

Example Implementation
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from witsv3.tools import BaseTool
   from typing import Dict, Any

   class FileProcessorTool(BaseTool):
       def __init__(self):
           super().__init__(
               name="file_processor",
               description="Process files with custom operations"
           )
           self.supported_formats = ["txt", "csv", "json"]

       async def execute(self, file_path: str, operation: str, **kwargs) -> Dict[str, Any]:
           try:
               if not self._validate_file(file_path):
                   return {
                       "success": False,
                       "error": "Invalid file format"
                   }

               result = await self._process_file(file_path, operation, **kwargs)
               return {
                   "success": True,
                   "result": result
               }
           except Exception as e:
               return {
                   "success": False,
                   "error": str(e)
               }

       def get_schema(self) -> Dict[str, Any]:
           return {
               "name": "file_processor",
               "description": "Process files with custom operations",
               "parameters": {
                   "type": "object",
                   "properties": {
                       "file_path": {
                           "type": "string",
                           "description": "Path to the file to process"
                       },
                       "operation": {
                           "type": "string",
                           "enum": ["read", "write", "transform"],
                           "description": "Operation to perform"
                       }
                   },
                   "required": ["file_path", "operation"]
               }
           }

       async def _process_file(self, file_path: str, operation: str, **kwargs) -> Any:
           # Implement file processing logic
           pass

       def _validate_file(self, file_path: str) -> bool:
           # Implement file validation
           pass

Tool Registration
--------------

Registering Your Tool
~~~~~~~~~~~~~~~~~~

Register your custom tool with WitsV3:

.. code-block:: python

   from witsv3 import WitsV3
   from my_tools import MyCustomTool

   wits = WitsV3()
   wits.register_tool(MyCustomTool())

Tool Configuration
~~~~~~~~~~~~~~~

Configure tool settings:

.. code-block:: python

   tool = MyCustomTool()
   tool.configure({
       "timeout": 30,
       "max_retries": 3,
       "custom_setting": "value"
   })

Testing Your Tool
--------------

Unit Tests
~~~~~~~~

Create unit tests for your tool:

.. code-block:: python

   import pytest
   from witsv3.tools import MyCustomTool

   @pytest.mark.asyncio
   async def test_my_custom_tool():
       tool = MyCustomTool()
       
       # Test successful execution
       result = await tool.execute(param1="test")
       assert result["success"] is True
       assert "result" in result

       # Test error handling
       result = await tool.execute(param1=None)
       assert result["success"] is False
       assert "error" in result

Integration Tests
~~~~~~~~~~~~~~

Test tool integration:

.. code-block:: python

   @pytest.mark.asyncio
   async def test_tool_integration():
       wits = WitsV3()
       wits.register_tool(MyCustomTool())
       
       result = await wits.execute_tool(
           "my_custom_tool",
           param1="test"
       )
       assert result["success"] is True

Best Practices
-----------

1. **Error Handling**
   * Use specific exceptions
   * Provide clear error messages
   * Implement proper logging

2. **Input Validation**
   * Validate all parameters
   * Use type hints
   * Document requirements

3. **Resource Management**
   * Clean up resources
   * Handle timeouts
   * Implement retries

4. **Documentation**
   * Document parameters
   * Provide examples
   * Include error cases

5. **Testing**
   * Write unit tests
   * Test edge cases
   * Verify integration

Example Tools
-----------

File Processing Tool
~~~~~~~~~~~~~~~~

A tool for processing files:

.. code-block:: python

   class FileProcessorTool(BaseTool):
       def __init__(self):
           super().__init__(
               name="file_processor",
               description="Process files with various operations"
           )

       async def execute(self, file_path: str, operation: str) -> Dict[str, Any]:
           # Implementation
           pass

API Integration Tool
~~~~~~~~~~~~~~~~

A tool for API integration:

.. code-block:: python

   class APIIntegrationTool(BaseTool):
       def __init__(self):
           super().__init__(
               name="api_integration",
               description="Integrate with external APIs"
           )

       async def execute(self, endpoint: str, method: str, data: Dict[str, Any]) -> Dict[str, Any]:
           # Implementation
           pass

Data Analysis Tool
~~~~~~~~~~~~~~~

A tool for data analysis:

.. code-block:: python

   class DataAnalysisTool(BaseTool):
       def __init__(self):
           super().__init__(
               name="data_analysis",
               description="Perform data analysis operations"
           )

       async def execute(self, data: List[Any], analysis_type: str) -> Dict[str, Any]:
           # Implementation
           pass

Troubleshooting
------------

Common Issues
~~~~~~~~~~~

1. **Tool Registration**
   * Check tool name
   * Verify initialization
   * Check dependencies

2. **Execution Errors**
   * Validate parameters
   * Check error handling
   * Verify resource cleanup

3. **Integration Issues**
   * Check tool registration
   * Verify configuration
   * Test communication

Support
------

For issues and feature requests, please visit the `GitHub repository <https://github.com/yourusername/witsv3>`_. 