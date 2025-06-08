Installation
============

Requirements
-----------

* Python 3.10 or higher
* pip (Python package installer)
* Git (for development installation)

Basic Installation
----------------

You can install WitsV3 using pip:

.. code-block:: bash

   pip install witsv3

Development Installation
----------------------

For development, clone the repository and install with development dependencies:

.. code-block:: bash

   git clone https://github.com/yourusername/witsv3.git
   cd witsv3
   make install-dev

This will install:
* All production dependencies
* Development tools (pytest, black, flake8, etc.)
* Documentation tools (Sphinx)
* Type checking tools (mypy)

Optional Dependencies
-------------------

Some features require additional dependencies:

* FAISS-GPU: For neural memory backend (requires CUDA)
* Docker: For containerized deployment
* Supabase: For cloud storage backend

You can install these using:

.. code-block:: bash

   # For FAISS-GPU (using conda)
   conda create -n faiss-gpu-env2 python=3.10
   conda activate faiss-gpu-env2
   conda install -c pytorch faiss-gpu

   # For Docker
   # Follow Docker installation instructions for your platform

   # For Supabase
   pip install supabase

Configuration
------------

After installation, you need to configure WitsV3:

1. Create a config.yaml file in your project directory
2. Set up your environment variables
3. Configure your LLM provider (Ollama)

Example config.yaml:

.. code-block:: yaml

   project_name: "WitsV3"
   version: "3.1.0"
   logging_level: "INFO"

   llm:
     provider: "ollama"
     model: "llama2"
     temperature: 0.7
     max_tokens: 2000

   memory:
     backend: "supabase_neural"
     file_path: "memory/"
     neural_web:
       enabled: true
       max_connections: 100

   supabase:
     url: "your-supabase-url"
     key: "your-supabase-key"
     enable_realtime: true

Verifying Installation
--------------------

To verify your installation:

.. code-block:: bash

   # Run tests
   make test

   # Check version
   python -c "import witsv3; print(witsv3.__version__)"

   # Run example
   python -m witsv3.cli "Hello, world!"

Troubleshooting
--------------

Common issues and solutions:

1. **ImportError: No module named 'witsv3'**
   * Make sure you're in the correct Python environment
   * Try reinstalling the package

2. **CUDA errors with FAISS-GPU**
   * Verify CUDA installation
   * Check GPU compatibility
   * Use CPU version if needed

3. **Supabase connection issues**
   * Verify credentials
   * Check network connection
   * Ensure Supabase project is active

4. **Ollama connection issues**
   * Verify Ollama is running
   * Check port configuration
   * Ensure model is downloaded

For more help, check the :ref:`troubleshooting` section or open an issue on GitHub. 