Welcome to WitsV3's documentation!
================================

.. warning::

   These Sphinx pages are **secondary** and largely historical. The canonical,
   maintained product guide is the repository root ``README.md``
   (Web UI first, Ollama local, July 2026 status). Forward work lives in
   ``planning/roadmap/suggested-features-2026-07.md``.

WitsV3 is a local-first LLM orchestration system: control center + ReAct
orchestrator + tools + memory, primarily used via ``python run_web.py``.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart

Features (current)
------------------

* Web UI (FastAPI + SSE) and CLI
* ReAct orchestrator with tool registry (~26 built-in tools)
* Document RAG, multi-provider web search, MCP discovery
* Verified code-edit pipeline (coding + self-repair agents)
* Optional neural-web memory backend and adaptive stacks (not default)

Quick run
---------

.. code-block:: bash

   python -m venv .venv
   # activate venv, then:
   pip install -r requirements.txt
   copy .env.example .env   # set WITSV3_WEB_TOKEN
   python run_web.py
