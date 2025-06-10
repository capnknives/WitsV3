---
title: "WitsV3 Planning Documentation"
created: "2025-06-09"
last_updated: "2025-06-09"
status: "active"
---

# WitsV3 Planning Documentation

This directory contains all planning, architectural, and technical documentation for the WitsV3 project.

## Documentation Organization

### Key Consolidated Documents

- **[DOCUMENTATION_REORGANIZATION.md](DOCUMENTATION_REORGANIZATION.md)** - Summary of the documentation reorganization project
- **[technical-notes/consolidated-system-fixes.md](technical-notes/consolidated-system-fixes.md)** - Comprehensive technical documentation covering all system fixes, improvements, and setup details
- **[architecture/system-architecture.md](architecture/system-architecture.md)** - Complete system architecture and component design
- **[roadmap/neural-web-roadmap.md](roadmap/neural-web-roadmap.md)** - Future enhancement roadmap for neural web architecture
- **[tasks/task-management.md](tasks/task-management.md)** - Current task status and backlog

### Directory Structure

- **[architecture/](architecture/)** - System architecture and design documents
- **[implementation/](implementation/)** - Implementation details and summaries
- **[roadmap/](roadmap/)** - Future plans and enhancement roadmaps
- **[tasks/](tasks/)** - Task tracking and management
- **[technical-notes/](technical-notes/)** - Debug information, fixes, and technical notes
- **[archive/](archive/)** - Archive of original documentation files

## Documentation Maintenance

### Documentation Tools

The following scripts are available in the `scripts/` directory to help maintain documentation:

- **`doc_maintenance.py`** - Main documentation maintenance tool with the following commands:
  - `list` - Lists all documentation files with their titles
  - `create` - Creates a new document with proper metadata
  - `update` - Updates metadata in an existing document
  - `archive` - Archives a document by moving it to the archive directory

Examples:

```bash
# List all documentation
python scripts/doc_maintenance.py list

# Create a new document
python scripts/doc_maintenance.py create planning/tasks/new-task.md "New Task Title"

# Update document metadata
python scripts/doc_maintenance.py update planning/tasks/some-file.md --status "draft"

# Archive a document
python scripts/doc_maintenance.py archive planning/tasks/obsolete-task.md
```

### Other Documentation Scripts

- **`migrate_docs.py`** - Used for initial migration of documentation (historical)
- **`standardize_format.py`** - Standardizes document format (historical)
- **`fix_metadata_format.py`** - Fixes metadata format issues (historical)
- **`cleanup_backups.py`** - Cleans up backup files (historical)
- **`cleanup_originals.py`** - Cleans up original files after migration (historical)

## Documentation Standards

### Metadata Format

All documents should include the following metadata at the top:

```
---
title: "Document Title"
created: "YYYY-MM-DD"
last_updated: "YYYY-MM-DD"
status: "active|archived|draft"
---
```

### Document Guidelines

- All new planning documents should be added to the appropriate subdirectory
- Use lowercase filenames with hyphens for spaces
- Include date in filename for time-sensitive documents (YYYY-MM-DD-title.md)
- Add metadata to the top of each document (title, created date, last updated date)

### Document Lifecycle

- Archive outdated documents to `planning/archive/` directory
- Update README index files when adding new documents
- Keep document section in main README.md updated
