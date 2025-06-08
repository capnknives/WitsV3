JSON Tool
========

The JSON Tool provides comprehensive JSON manipulation capabilities.

Overview
-------

The JSON Tool allows you to perform various operations on JSON data, including getting values, setting values, merging objects, validating structure, and formatting. It provides a safe and efficient way to work with JSON data.

Installation
----------

The JSON Tool is included in the core WitsV3 package. No additional installation is required.

Usage
----

Basic Usage
~~~~~~~~~

.. code-block:: python

   from witsv3.tools import JSONTool

   # Initialize the tool
   tool = JSONTool()

   # Get a value from JSON
   result = await tool.execute(
       operation="get",
       data='{"name": "test", "value": 42}',
       path="name"
   )

Parameters
~~~~~~~~~

* **operation** (str, required): The operation to perform
  * "get": Get a value from JSON
  * "set": Set a value in JSON
  * "merge": Merge two JSON objects
  * "validate": Validate JSON structure
  * "format": Format JSON with indentation
  * "read_file": Read JSON from file
  * "write_file": Write JSON to file

* **data** (str/dict, required): The JSON data to operate on
* **path** (str, optional): The path to the value (for get/set operations)
* **value** (any, optional): The value to set (for set operation)
* **merge_data** (str/dict, optional): The JSON data to merge (for merge operation)
* **file_path** (str, optional): The file path (for read_file/write_file operations)

Return Value
~~~~~~~~~~

The tool returns a dictionary with the following structure:

.. code-block:: python

   {
       "success": True,
       "result": "Operation result",
       "error": None
   }

Error Handling
~~~~~~~~~~~~

The tool handles various error cases:

* Invalid JSON syntax
* Invalid operation
* Path not found
* File I/O errors
* Validation errors

Example error response:

.. code-block:: python

   {
       "success": False,
       "result": None,
       "error": "Error message"
   }

Advanced Usage
-----------

Get Value
~~~~~~~~

Get a value from nested JSON:

.. code-block:: python

   result = await tool.execute(
       operation="get",
       data='{"user": {"name": "John", "age": 30}}',
       path="user.name"
   )

Set Value
~~~~~~~~

Set a value in nested JSON:

.. code-block:: python

   result = await tool.execute(
       operation="set",
       data='{"user": {"name": "John"}}',
       path="user.age",
       value=30
   )

Merge Objects
~~~~~~~~~~~

Merge two JSON objects:

.. code-block:: python

   result = await tool.execute(
       operation="merge",
       data='{"a": 1, "b": 2}',
       merge_data='{"b": 3, "c": 4}'
   )

Validate Structure
~~~~~~~~~~~~~~~

Validate JSON against a schema:

.. code-block:: python

   result = await tool.execute(
       operation="validate",
       data='{"name": "John", "age": 30}',
       schema={
           "type": "object",
           "properties": {
               "name": {"type": "string"},
               "age": {"type": "integer"}
           }
       }
   )

Format JSON
~~~~~~~~~

Format JSON with indentation:

.. code-block:: python

   result = await tool.execute(
       operation="format",
       data='{"name":"John","age":30}'
   )

File Operations
~~~~~~~~~~~~

Read JSON from file:

.. code-block:: python

   result = await tool.execute(
       operation="read_file",
       file_path="data.json"
   )

Write JSON to file:

.. code-block:: python

   result = await tool.execute(
       operation="write_file",
       data='{"name": "John", "age": 30}',
       file_path="output.json"
   )

Best Practices
-----------

1. **Data Validation**
   * Validate input JSON
   * Check data types
   * Handle missing values

2. **Path Handling**
   * Use dot notation
   * Handle nested paths
   * Validate path existence

3. **Error Handling**
   * Catch specific exceptions
   * Provide clear messages
   * Log error details

4. **Performance**
   * Use efficient operations
   * Minimize parsing
   * Cache results

Example Use Cases
--------------

1. **Configuration Management**
   * Read/write config files
   * Update settings
   * Validate configs

2. **Data Transformation**
   * Convert formats
   * Merge datasets
   * Filter data

3. **API Integration**
   * Parse responses
   * Format requests
   * Handle errors

API Reference
-----------

.. py:class:: JSONTool

   JSON manipulation tool.

   .. py:method:: execute(operation: str, data: Union[str, dict], **kwargs) -> dict

      Execute a JSON operation.

      :param operation: Operation to perform
      :param data: JSON data to operate on
      :param kwargs: Additional operation-specific parameters
      :return: Dictionary containing operation results

   .. py:method:: get_schema() -> dict

      Get the tool's schema for LLM consumption.

      :return: Dictionary containing tool schema

Limitations
---------

* No streaming support
* Limited file size
* No binary data
* No circular references
* No custom encoders

Troubleshooting
------------

Common Issues
~~~~~~~~~~~

1. **Invalid JSON**
   * Check syntax
   * Validate structure
   * Handle encoding

2. **Path Errors**
   * Verify path format
   * Check existence
   * Handle nesting

3. **File Issues**
   * Check permissions
   * Verify paths
   * Handle encoding

Support
------

For issues and feature requests, please visit the `GitHub repository <https://github.com/yourusername/witsv3>`_. 