Tools
=====

WitsV3 provides a comprehensive set of tools for various tasks.

Built-in Tools
------------

Web Search Tool
~~~~~~~~~~~~~

The Web Search Tool provides web search capabilities using DuckDuckGo:

.. code-block:: python

   from witsv3.tools import WebSearchTool

   tool = WebSearchTool()
   results = await tool.execute(
       query="Python async programming",
       max_results=5
   )

Python Execution Tool
~~~~~~~~~~~~~~~~~~

The Python Execution Tool allows safe execution of Python code:

.. code-block:: python

   from witsv3.tools import PythonExecutionTool

   tool = PythonExecutionTool()
   result = await tool.execute(
       code="print('Hello, World!')"
   )

JSON Tool
~~~~~~~~

The JSON Tool provides JSON manipulation capabilities:

.. code-block:: python

   from witsv3.tools import JSONTool

   tool = JSONTool()
   result = await tool.execute(
       operation="get",
       data='{"name": "test"}',
       path="name"
   )

Math Tool
~~~~~~~~

The Math Tool provides mathematical and statistical operations:

.. code-block:: python

   from witsv3.tools import MathTool

   tool = MathTool()
   result = await tool.execute(
       operation="basic_stats",
       data=[1, 2, 3, 4, 5]
   )

Creating Custom Tools
------------------

Base Tool
~~~~~~~~

All tools inherit from the BaseTool class:

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

Tool Registration
~~~~~~~~~~~~~~~

Register your custom tool:

.. code-block:: python

   from witsv3 import WitsV3

   wits = WitsV3()
   wits.register_tool(MyCustomTool())

Tool Schema
~~~~~~~~~

Define your tool's schema:

.. code-block:: python

   def get_schema(self):
       return {
           "name": "my_custom_tool",
           "description": "A custom tool for specific tasks",
           "parameters": {
               "type": "object",
               "properties": {
                   "param1": {"type": "string"},
                   "param2": {"type": "integer"}
               },
               "required": ["param1"]
           }
       }

Error Handling
~~~~~~~~~~~~

Handle errors in your tool:

.. code-block:: python

   async def execute(self, **kwargs):
       try:
           # Your tool logic here
           return {"success": True, "result": "Success"}
       except Exception as e:
           return {
               "success": False,
               "error": str(e)
           }

Tool Testing
----------

Write tests for your tool:

.. code-block:: python

   import pytest
   from witsv3.tools import MyCustomTool

   @pytest.mark.asyncio
   async def test_my_custom_tool():
       tool = MyCustomTool()
       result = await tool.execute(param1="test")
       assert result["success"] is True
       assert "result" in result

Tool Best Practices
----------------

1. **Error Handling**
   * Always handle exceptions
   * Return meaningful error messages
   * Use appropriate error types

2. **Input Validation**
   * Validate all input parameters
   * Use type hints
   * Document parameter requirements

3. **Performance**
   * Use async operations
   * Implement timeouts
   * Handle resource cleanup

4. **Security**
   * Sanitize inputs
   * Implement access control
   * Use secure defaults

5. **Documentation**
   * Document all parameters
   * Provide usage examples
   * Include error cases

Example Tools
-----------

See the :ref:`examples` section for complete tool examples.

API Reference
-----------

.. toctree::
   :maxdepth: 2

   web_search
   python_execution
   json
   math
   custom_tools 