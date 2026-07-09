---
title: "WitsV3 Path Migration Guide"
created: "2025-06-10"
last_updated: "2026-07-09"
status: "historical"
---

# WitsV3 Path Migration Guide

> **July 2026 update:** Planning docs moved again from `planning/` to `docs/`.
> `planning/` is now a redirect stub; see [`docs/README.md`](docs/README.md).

> **Phase 3–4 (July 2026):** All mutable runtime data lives under `var/`.
> Legacy top-level folders auto-migrate on startup. `var/documents/` was renamed
> to `var/user_files/` (not to be confused with repo `docs/`). See
> [`docs/architecture/paths-and-layout.md`](docs/architecture/paths-and-layout.md).


This document maps previous file locations to the reorganized structure. Use it when
updating old imports or reading archived docs.

## Table of Contents

- [1. Documentation Migration](#1-documentation-migration)
- [2. Test Directory Changes](#2-test-directory-changes)
- [3. Source Code Reorganization](#3-source-code-reorganization)
- [4. Import Path Updates](#4-import-path-updates)
- [5. Configuration Changes](#5-configuration-changes)

## 1. Documentation Migration

| Previous Location             | Current Location                                         | Notes                                                    |
| ----------------------------- | -------------------------------------------------------- | -------------------------------------------------------- |
| `/*.md` (root planning docs) | `/docs/**/*.md` | Canonical tree since July 2026 |
| `/planning/**/*.md` (except README) | `/docs/` equivalents | **Deleted** — stub only |
| `/CLAUDE_EVOLUTION_PROMPT.md` | `/planning/implementation/claude-evolution-prompt.md`    |                                                          |
| `/AUTHENTICATION_STATUS.md`   | `/planning/technical-notes/consolidated-system-fixes.md` | Merged into consolidated document                        |
| `/DEBUG_*.md`                 | `/planning/technical-notes/consolidated-system-fixes.md` | All debug docs consolidated                              |
| `/IMPLEMENTATION_SUMMARY.md`  | `/planning/implementation/`                              | Split into multiple implementation docs                  |
| `/WITSV3_FIXES.md`            | `/planning/technical-notes/consolidated-system-fixes.md` | Merged into consolidated document                        |

## 2. Test Directory Changes

| Previous Pattern                  | Current Pattern                | Notes                                 |
| --------------------------------- | ------------------------------ | ------------------------------------- |
| Inline test functions             | `/tests/**/*.py`               | Tests moved to dedicated directory    |
| Module-level test files           | `/tests/module_name/test_*.py` | Tests mirror package structure        |
| `test_*.py` in module directories | `/tests/module_name/test_*.py` | Removed tests from source directories |

## 3. Source Code Reorganization

| Previous Structure     | Current Structure                | Notes                            |
| ---------------------- | -------------------------------- | -------------------------------- |
| `/src/agents/`         | `/agents/`                       | Moved to root for easier imports |
| `/src/core/`           | `/core/`                         | Moved to root for easier imports |
| `/src/tools/`          | `/tools/`                        | Moved to root for easier imports |
| `/src/schemas/`        | `/core/schemas.py`               | Consolidated into core package   |
| `/models/adapters/`    | `/core/adaptive/`                | Renamed for clarity              |
| `/models/embeddings/`  | `/core/neural_memory_backend.py` | Consolidated into single module  |
| `/config/personality/` | `/config/`                       | Simplified structure             |

## 4. Import Path Updates

If you're updating code from a previous version, use these import path mappings:

### Previous Imports

```python
from src.agents import BaseAgent
from src.core import LLMInterface
from src.tools import BaseTool
from src.schemas import StreamData
from models.adapters import ModelAdapter
```

### Updated Imports

```python
from agents import BaseAgent
from core import BaseLLMInterface
from tools import BaseTool
from core.schemas import StreamData
from core.adaptive import AdaptiveTokenizer
```

## 5. Configuration Changes

| Previous Config Structure         | Current Config Structure                   | Notes                     |
| --------------------------------- | ------------------------------------------ | ------------------------- |
| `/config/llm_config.yaml`         | `/config.yaml` (section: `llm_interface`)  | Consolidated              |
| `/config/agent_config.yaml`       | `/config.yaml` (section: `agents`)         | Consolidated              |
| `/config/tool_config.yaml`        | `/config.yaml` (section: `tool_system`)    | Consolidated              |
| `/config/memory_config.yaml`      | `/config.yaml` (section: `memory_manager`) | Consolidated              |
| `/config/personality_config.yaml` | `/config/wits_personality.yaml`            | Kept separate due to size |

## Compatibility Mapping

For backward compatibility, the following mappings are maintained:

| Legacy Feature   | Current Implementation                        | Notes                                     |
| ---------------- | --------------------------------------------- | ----------------------------------------- |
| `LLMInterface`   | `BaseLLMInterface` + provider implementations | Split into base class and implementations |
| `MemoryManager`  | `MemoryManager` + backend implementations     | Added pluggable backends                  |
| `ToolRegistry`   | `ToolRegistry` + auto-discovery               | Added dynamic tool discovery              |
| `StreamResponse` | `StreamData`                                  | Renamed for clarity                       |

This guide should help you navigate the reorganized project structure. For detailed information on the current structure, see [FILE_STRUCTURE.md](FILE_STRUCTURE.md).
