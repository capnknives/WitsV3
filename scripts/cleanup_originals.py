#!/usr/bin/env python3
"""
Script to clean up original markdown files in the root directory after migration.
"""

import os
import shutil

# Files to be cleaned up - map from original file to migrated file
FILE_MAPPING = {
    "planning-md.md": "docs/architecture/system-architecture.md",
    "task-md.md": "docs/tasks/task-management.md",
    "IMPLEMENTATION_SUMMARY.md": "docs/implementation/personality-ethics-network-implementation.md",
    "WITSV3_FIXES.md": "docs/technical-notes/system-fixes.md",
    "witsv3_improvement_roadmap.md": "docs/roadmap/neural-web-roadmap.md",
    "DEBUG_INTERPRETER_FIX.md": "docs/technical-notes/interpreter-fixes.md",
    "DEBUG_SYSTEM_IMPROVEMENTS.md": "docs/technical-notes/system-improvements.md",
    "DEBUG_SETUP_COMPLETE.md": "docs/technical-notes/setup-completion.md",
    "CLAUDE_EVOLUTION_PROMPT.md": "docs/implementation/claude-evolution-prompt.md",
    "AUTHENTICATION_STATUS.md": "docs/technical-notes/authentication-status.md",
    "copper-scroll-adaptive-llm.md": "docs/implementation/adaptive-llm-design.md"
}

# Files to ignore during cleanup
IGNORE_FILES = [
    "README.md",  # Keep main README
    ".cursorrules.md"  # System file
]

def cleanup_files():
    """Check which files have been migrated and delete original files."""
    for source, destination in FILE_MAPPING.items():
        if source in IGNORE_FILES:
            print(f"Ignoring: {source}")
            continue

        # Check if source file exists
        if not os.path.exists(source):
            print(f"Source file {source} does not exist. Skipping.")
            continue

        # Check if destination file exists
        if not os.path.exists(destination):
            print(f"WARNING: Destination file {destination} does not exist. Will not delete {source}.")
            continue

        # Create a backup directory if it doesn't exist
        backup_dir = "docs/archive/originals"
        os.makedirs(backup_dir, exist_ok=True)

        # Move the original file to backup
        backup_path = os.path.join(backup_dir, source)
        shutil.copy2(source, backup_path)
        print(f"Backed up: {source} -> {backup_path}")

        # Delete the original file
        os.remove(source)
        print(f"Deleted: {source}")

if __name__ == "__main__":
    print("Starting cleanup of original markdown files...")
    cleanup_files()
    print("Cleanup complete!")
