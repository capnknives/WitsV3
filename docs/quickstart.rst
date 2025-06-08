Quickstart Guide
===============

This guide will help you get started with WitsV3 quickly.

Basic Usage
----------

Initialize WitsV3:

.. code-block:: python

   from witsv3 import WitsV3

   # Create a new instance
   wits = WitsV3()

   # Run a simple command
   result = wits.run("What is the weather in New York?")
   print(result)

Using the CLI
------------

WitsV3 provides a command-line interface:

.. code-block:: bash

   # Basic usage
   wits "What is the weather in New York?"

   # With options
   wits --model llama2 --temperature 0.8 "Tell me a story"

   # Interactive mode
   wits --interactive

   # Background agent
   wits --agent

Working with Tools
----------------

WitsV3 comes with several built-in tools:

.. code-block:: python

   from witsv3 import WitsV3
   from witsv3.tools import WebSearchTool, PythonExecutionTool

   wits = WitsV3()

   # Use web search
   results = wits.tools.web_search.execute(
       query="Python async programming",
       max_results=5
   )

   # Execute Python code
   result = wits.tools.python_execution.execute(
       code="print('Hello, World!')"
   )

Creating Custom Tools
-------------------

You can create your own tools:

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

   # Register the tool
   wits.register_tool(MyCustomTool())

Using the Neural Web
------------------

The neural web provides advanced memory capabilities:

.. code-block:: python

   from witsv3 import WitsV3

   wits = WitsV3()

   # Store information
   wits.memory.store(
       content="Python is a programming language",
       metadata={"type": "fact", "topic": "programming"}
   )

   # Search memory
   results = wits.memory.search(
       query="programming languages",
       limit=5
   )

   # Get neural connections
   connections = wits.memory.get_connections(
       node_id="python_language",
       max_depth=2
   )

Background Agent
--------------

Run a background agent for continuous processing:

.. code-block:: python

   from witsv3 import WitsV3

   wits = WitsV3()

   # Start background agent
   wits.start_agent()

   # Send tasks to agent
   wits.agent.send_task({
       "type": "process",
       "data": "Process this data"
   })

   # Stop agent
   wits.stop_agent()

Error Handling
------------

WitsV3 provides robust error handling:

.. code-block:: python

   from witsv3 import WitsV3, WitsError

   wits = WitsV3()

   try:
       result = wits.run("Complex task")
   except WitsError as e:
       print(f"Error: {e}")
   except Exception as e:
       print(f"Unexpected error: {e}")

Configuration
------------

Configure WitsV3 behavior:

.. code-block:: python

   from witsv3 import WitsV3, WitsConfig

   config = WitsConfig(
       llm_model="llama2",
       temperature=0.7,
       max_tokens=2000,
       memory_backend="supabase_neural"
   )

   wits = WitsV3(config=config)

Next Steps
---------

* Read the :ref:`architecture` documentation
* Explore the :ref:`tools/index`
* Learn about :ref:`agents/index`
* Check the :ref:`api/index`

For more examples and advanced usage, see the :ref:`examples` section. 