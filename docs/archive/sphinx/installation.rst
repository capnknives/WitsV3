Installation
============

.. warning::

   Prefer the root ``README.md`` Quick start. This page is a short Sphinx mirror.

Requirements
------------

* Python 3.10+
* Ollama with recommended models (``qwen3:8b``, ``qwen2.5-coder:7b``,
  ``llama3.2:3b``, ``nomic-embed-text``)
* GPU with ~8 GB VRAM recommended (optional)

From source
-----------

.. code-block:: bash

   git clone https://github.com/capnknives/WitsV3.git
   cd WitsV3
   python -m venv .venv
   # Windows: .venv\Scripts\activate
   # Unix:    source .venv/bin/activate
   pip install -r requirements.txt
   python scripts/setup_local_data.py
   copy .env.example .env
   # set WITSV3_WEB_TOKEN (and optional search / Anthropic keys)

Or run ``python install.py`` for deps + local data + auth + model pulls.

There is no supported ``pip install witsv3`` release workflow documented for
this personal assistant stack — clone the repo.

Secrets
-------

Never put credentials in ``config.yaml``. Use ``.env`` (see ``.env.example``).
