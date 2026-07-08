Quickstart Guide
================

.. warning::

   Prefer the root ``README.md``. The old packaged ``wits`` CLI API described
   elsewhere in this Sphinx tree is not the current entry point.

Web UI (recommended)
--------------------

.. code-block:: bash

   .venv\Scripts\python.exe run_web.py   # Windows
   # .venv/bin/python run_web.py         # Unix

Open ``http://localhost:8000`` and enter ``WITSV3_WEB_TOKEN``.

CLI
---

.. code-block:: bash

   python run.py
   python run.py --test

Tests
-----

.. code-block:: bash

   pytest tests/ -q --no-cov

Next reading
------------

* Root ``README.md`` — capabilities, config, self-repair
* ``AGENTS.md`` — agent hierarchy
* ``planning/roadmap/suggested-features-2026-07.md`` — what's next
