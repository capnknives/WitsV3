Web Search Tool
=============

The Web Search Tool provides web search capabilities using DuckDuckGo.

Overview
-------

The Web Search Tool allows you to perform web searches and retrieve results programmatically. It uses the DuckDuckGo API to fetch search results in a structured format.

Installation
----------

The Web Search Tool is included in the core WitsV3 package. No additional installation is required.

Usage
----

Basic Usage
~~~~~~~~~

.. code-block:: python

   from witsv3.tools import WebSearchTool

   # Initialize the tool
   tool = WebSearchTool()

   # Perform a search
   results = await tool.execute(
       query="Python async programming",
       max_results=5
   )

Parameters
~~~~~~~~~

* **query** (str, required): The search query string
* **max_results** (int, optional): Maximum number of results to return (default: 5)

Return Value
~~~~~~~~~~

The tool returns a dictionary with the following structure:

.. code-block:: python

   {
       "success": True,
       "results": [
           {
               "title": "Result title",
               "link": "https://example.com",
               "snippet": "Result description..."
           },
           # ... more results
       ]
   }

Error Handling
~~~~~~~~~~~~

The tool handles various error cases:

* Network errors
* Invalid responses
* Rate limiting
* Timeout errors

Example error response:

.. code-block:: python

   {
       "success": False,
       "error": "Error message"
   }

Advanced Usage
-----------

Custom Headers
~~~~~~~~~~~~

You can customize the request headers:

.. code-block:: python

   tool = WebSearchTool()
   tool.headers = {
       "User-Agent": "Custom User Agent"
   }

Timeout Configuration
~~~~~~~~~~~~~~~~~~

Set a custom timeout:

.. code-block:: python

   tool = WebSearchTool()
   tool.timeout = 30  # seconds

Error Handling Example
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   try:
       results = await tool.execute(query="test")
       if results["success"]:
           for result in results["results"]:
               print(f"Title: {result['title']}")
               print(f"Link: {result['link']}")
               print(f"Snippet: {result['snippet']}")
       else:
           print(f"Error: {results['error']}")
   except Exception as e:
       print(f"Unexpected error: {e}")

Best Practices
-----------

1. **Query Optimization**
   * Use specific, targeted queries
   * Include relevant keywords
   * Avoid overly broad searches

2. **Result Processing**
   * Validate result structure
   * Handle missing fields
   * Implement result filtering

3. **Rate Limiting**
   * Implement request throttling
   * Handle rate limit errors
   * Use appropriate delays

4. **Error Recovery**
   * Implement retry logic
   * Handle network issues
   * Log error details

Example Use Cases
--------------

1. **Research Assistant**
   * Gather information on topics
   * Find relevant documentation
   * Collect reference materials

2. **Content Aggregation**
   * Collect news articles
   * Gather blog posts
   * Find related content

3. **Data Collection**
   * Gather market data
   * Collect statistics
   * Find research papers

API Reference
-----------

.. py:class:: WebSearchTool

   Web search tool using DuckDuckGo.

   .. py:method:: execute(query: str, max_results: int = 5) -> dict

      Execute a web search.

      :param query: Search query string
      :param max_results: Maximum number of results to return
      :return: Dictionary containing search results or error information

   .. py:method:: get_schema() -> dict

      Get the tool's schema for LLM consumption.

      :return: Dictionary containing tool schema

Limitations
---------

* Rate limiting by DuckDuckGo
* Limited number of results per query
* No advanced search operators
* No image or video search
* No real-time results

Troubleshooting
------------

Common Issues
~~~~~~~~~~~

1. **No Results**
   * Check query syntax
   * Verify network connection
   * Check rate limiting

2. **Timeout Errors**
   * Increase timeout value
   * Check network speed
   * Verify API availability

3. **Invalid Responses**
   * Check response format
   * Verify API changes
   * Update tool version

Support
------

For issues and feature requests, please visit the `GitHub repository <https://github.com/yourusername/witsv3>`_. 