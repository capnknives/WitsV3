---
title: "WitsV3 Documentation Archive"
created: "2025-06-09"
last_updated: "2026-07-08"
status: "active"
---

# WitsV3 Documentation Archive

This directory contains archived documentation and pointers to pruned dormant code.
Several code trees were removed in July 2026 footprint work; recover via git tags
`archive-pre-prune-2026-07` (GUI) and `archive-pre-prune-2b-2026-07` (adaptive LLM,
sphinx, synthetic brain). Stub READMEs under each pruned folder document restore commands.

## Directory Structure

- **Root** - Archived documents from all categories
- **[originals/](originals/)** - Original files from before the documentation reorganization

## Archive Guidelines

1. **When to Archive**:

   - Documentation that has been superseded by newer versions
   - Information that is no longer relevant but has historical value
   - Draft documents that were never completed

2. **How to Archive**:

   - Use the archive command in the documentation maintenance tool:
     ```bash
     python scripts/doc_maintenance.py archive docs/path/to/document.md
     ```
   - This will:
     - Update the document's status to "archived"
     - Move the document to the archive directory
     - Preserve the original content

3. **Archive Organization**:
   - Documents are stored in the root of the archive directory
   - Duplicate filenames are handled by adding a date suffix

## Accessing Archived Documents

Archived documents can be referenced like any other document, but they are clearly marked as archived and should not be considered current information.

To view a list of all archived documents:

```bash
python scripts/doc_maintenance.py list
```
