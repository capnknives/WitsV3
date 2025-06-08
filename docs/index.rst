Welcome to WitsV3's documentation!
================================

WitsV3 is an LLM orchestration system with a CLI-first approach, ReAct pattern, and tool registry.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   architecture
   tools/index
   agents/index
   core/index
   api/index
   development
   contributing
   changelog

Features
--------

* CLI-first approach for easy integration
* ReAct pattern for agent reasoning
* Tool registry for extensibility
* Neural web architecture
* Adaptive LLM system
* Background agent support
* Comprehensive testing suite

Installation
-----------

.. code-block:: bash

   pip install witsv3

For development installation:

.. code-block:: bash

   git clone https://github.com/yourusername/witsv3.git
   cd witsv3
   make install-dev

Quick Start
----------

.. code-block:: python

   from witsv3 import WitsV3

   # Initialize the system
   wits = WitsV3()

   # Run a command
   result = wits.run("What is the weather in New York?")

   # Get the response
   print(result)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 