#!/usr/bin/env python3
"""
Script to add metadata to markdown files.
"""

import os
import datetime

# Files to add metadata to
FILES = [
    "planning/architecture/system-architecture.md",
    "planning/tasks/task-management.md",
    "planning/implementation/adaptive-llm-design.md",
]

# Metadata template
METADATA_TEMPLATE = """---
title: "{title}"
created: "{created_date}"
last_updated: "{last_updated_date}"
status: "active"
---

"""

def add_metadata():
    """Add metadata to files."""
    now = datetime.datetime.now().strftime("%Y-%m-%d")

    for file_path in FILES:
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist. Skipping.")
            continue

        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check if metadata already exists
        if content.startswith('---'):
            print(f"File {file_path} already has metadata. Skipping.")
            continue

        # Extract title from first line if it starts with # or from filename
        title = os.path.basename(file_path).replace('-', ' ').replace('_', ' ').replace('.md', '')
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
        backup_path = f"{file_path}.bak"
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Created backup: {backup_path}")

        # Write to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(metadata + content)

        print(f"Added metadata to: {file_path}")

if __name__ == "__main__":
    add_metadata()
    print("Metadata addition complete!")
