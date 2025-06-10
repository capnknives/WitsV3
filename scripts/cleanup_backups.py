#!/usr/bin/env python3
"""
Script to clean up temporary backup files created during the documentation migration.
"""

import os
import glob

# Directories to process
DIRECTORIES = [
    "planning/architecture",
    "planning/implementation",
    "planning/roadmap",
    "planning/tasks",
    "planning/technical-notes"
]

def cleanup_backups():
    """Clean up backup files."""
    backup_count = 0

    # Clean up backups in each directory
    for directory in DIRECTORIES:
        if not os.path.exists(directory):
            print(f"Directory {directory} does not exist. Skipping.")
            continue

        # Find all .bak files
        backup_files = glob.glob(f"{directory}/*.bak")
        backup_files.extend(glob.glob(f"{directory}/*.format.bak"))
        backup_files.extend(glob.glob(f"{directory}/*.fix.bak"))

        for backup_file in backup_files:
            if os.path.exists(backup_file):
                os.remove(backup_file)
                print(f"Deleted: {backup_file}")
                backup_count += 1
            else:
                print(f"Warning: Backup file {backup_file} not found.")

    print(f"Cleaned up {backup_count} backup files.")

if __name__ == "__main__":
    print("Starting backup cleanup...")
    cleanup_backups()
    print("Backup cleanup complete!")
