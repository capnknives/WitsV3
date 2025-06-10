#!/usr/bin/env python3
"""
Script to migrate markdown documentation files to the new organization structure.
"""

import os
import shutil
import datetime

# Map of source files to destination files
FILE_MAPPING = {
    "planning-md.md": "planning/architecture/system-architecture.md",
    "task-md.md": "planning/tasks/task-management.md",
    "IMPLEMENTATION_SUMMARY.md": "planning/implementation/personality-ethics-network-implementation.md",
    "WITSV3_FIXES.md": "planning/technical-notes/system-fixes.md",
    "witsv3_improvement_roadmap.md": "planning/roadmap/neural-web-roadmap.md",
    "DEBUG_INTERPRETER_FIX.md": "planning/technical-notes/interpreter-fixes.md",
    "DEBUG_SYSTEM_IMPROVEMENTS.md": "planning/technical-notes/system-improvements.md",
    "DEBUG_SETUP_COMPLETE.md": "planning/technical-notes/setup-completion.md",
    "CLAUDE_EVOLUTION_PROMPT.md": "planning/implementation/claude-evolution-prompt.md",
    "AUTHENTICATION_STATUS.md": "planning/technical-notes/authentication-status.md",
    "copper-scroll-adaptive-llm.md": "planning/implementation/adaptive-llm-design.md"
}

# Metadata template to add to the top of each file
METADATA_TEMPLATE = """---
title: "{title}"
created: "{created_date}"
last_updated: "{last_updated_date}"
status: "active"
---

"""

def get_title_from_filename(filename):
    """Extract a human-readable title from the filename."""
    basename = os.path.basename(filename)
    base, _ = os.path.splitext(basename)

    # Replace hyphens with spaces and capitalize
    title = base.replace('-', ' ').replace('_', ' ')

    # Special case for all caps filenames
    if title.isupper():
        title = title.title()

    return title

def migrate_files():
    """Migrate files according to the mapping."""
    now = datetime.datetime.now().strftime("%Y-%m-%d")

    for source, destination in FILE_MAPPING.items():
        if not os.path.exists(source):
            print(f"Warning: Source file {source} does not exist. Skipping.")
            continue

        # Create destination directory if it doesn't exist
        dest_dir = os.path.dirname(destination)
        os.makedirs(dest_dir, exist_ok=True)

        # Read the source file
        with open(source, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract title from first line if it starts with # or from filename
        title = get_title_from_filename(destination)
        lines = content.splitlines()
        if lines and lines[0].startswith('# '):
            title = lines[0][2:].strip()

        # Add metadata
        metadata = METADATA_TEMPLATE.format(
            title=title,
            created_date=now,
            last_updated_date=now
        )

        # Create backup
        backup_path = f"{source}.bak"
        shutil.copy2(source, backup_path)
        print(f"Created backup: {backup_path}")

        # Write to destination
        with open(destination, 'w', encoding='utf-8') as f:
            f.write(metadata + content)

        print(f"Migrated: {source} -> {destination}")

if __name__ == "__main__":
    migrate_files()
    print("Documentation migration complete!")
