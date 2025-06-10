#!/usr/bin/env python3
"""
Script to help with document maintenance tasks.
"""

import os
import re
import datetime
import argparse
import glob

PLANNING_DIR = "planning"
METADATA_TEMPLATE = """---
title: "{title}"
created: "{created_date}"
last_updated: "{today}"
status: "{status}"
---
"""

def update_metadata(file_path, title=None, status="active"):
    """Update metadata in a markdown file."""
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return False

    today = datetime.datetime.now().strftime("%Y-%m-%d")

    # Read file content
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if metadata exists
    has_metadata = content.startswith('---')
    created_date = today  # Default for new metadata

    if has_metadata:
        # Find end of metadata
        metadata_end = content.find('---', 3)
        if metadata_end != -1:
            metadata = content[0:metadata_end+3]
            body = content[metadata_end+3:].strip()

            # Extract created date if it exists
            created_match = re.search(r'created: "([^"]+)"', metadata)
            if created_match:
                created_date = created_match.group(1)

            # Extract title if not provided
            if not title:
                title_match = re.search(r'title: "([^"]+)"', metadata)
                if title_match:
                    title = title_match.group(1)
        else:
            # Malformed metadata
            has_metadata = False
            body = content
    else:
        body = content

    # Extract title from first heading if not provided
    if not title:
        title_match = re.search(r'^# (.*?)$', body, re.MULTILINE)
        if title_match:
            title = title_match.group(1)
        else:
            # Use filename as title
            basename = os.path.basename(file_path)
            title = os.path.splitext(basename)[0].replace('-', ' ').replace('_', ' ').title()

    # Create new metadata
    new_metadata = METADATA_TEMPLATE.format(
        title=title,
        created_date=created_date,
        today=today,
        status=status
    )

    # Create backup
    backup_path = f"{file_path}.bak"
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)

    # Write updated content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_metadata + body)

    print(f"Updated metadata in {file_path}")
    return True

def archive_document(file_path):
    """Archive a document by moving it to the archive directory and updating its status."""
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return False

    # Create archive directory if it doesn't exist
    archive_dir = os.path.join(PLANNING_DIR, "archive")
    os.makedirs(archive_dir, exist_ok=True)

    # Update metadata to mark as archived
    update_metadata(file_path, status="archived")

    # Move to archive
    basename = os.path.basename(file_path)
    archive_path = os.path.join(archive_dir, basename)

    # Handle duplicate filenames
    if os.path.exists(archive_path):
        today = datetime.datetime.now().strftime("%Y%m%d")
        name, ext = os.path.splitext(basename)
        archive_path = os.path.join(archive_dir, f"{name}-{today}{ext}")

    os.rename(file_path, archive_path)
    print(f"Archived {file_path} to {archive_path}")
    return True

def create_document(file_path, title):
    """Create a new document with proper metadata."""
    if os.path.exists(file_path):
        print(f"Error: File {file_path} already exists.")
        return False

    # Ensure directory exists
    directory = os.path.dirname(file_path)
    os.makedirs(directory, exist_ok=True)

    today = datetime.datetime.now().strftime("%Y-%m-%d")

    # Create metadata
    metadata = METADATA_TEMPLATE.format(
        title=title,
        created_date=today,
        today=today,
        status="active"
    )

    # Create document with metadata and title
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(metadata + f"# {title}\n\n")

    print(f"Created document: {file_path}")
    return True

def list_documents():
    """List all markdown documents in the planning directory."""
    print("WitsV3 Documentation Index:")
    print("--------------------------\n")

    # First list documents in the planning root
    print("## Planning Root\n")
    for file_path in glob.glob(f"{PLANNING_DIR}/*.md"):
        if os.path.basename(file_path) != "README.md":
            # Extract title if possible
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            title = os.path.basename(file_path).replace('.md', '')

            # Try to extract from metadata
            title_match = re.search(r'title: "([^"]+)"', content)
            if title_match:
                title = title_match.group(1)
            else:
                # Try to extract from first heading
                heading_match = re.search(r'^# (.*?)$', content, re.MULTILINE)
                if heading_match:
                    title = heading_match.group(1)

            print(f"- [{title}]({file_path})")

    # Then list documents in subdirectories
    for category in ['architecture', 'implementation', 'roadmap', 'tasks', 'technical-notes']:
        dir_path = os.path.join(PLANNING_DIR, category)
        if not os.path.exists(dir_path):
            continue

        print(f"\n## {category.title()}\n")

        for file_path in glob.glob(f"{dir_path}/*.md"):
            if os.path.basename(file_path) != "README.md":
                # Extract title if possible
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                title = os.path.basename(file_path).replace('.md', '')

                # Try to extract from metadata
                title_match = re.search(r'title: "([^"]+)"', content)
                if title_match:
                    title = title_match.group(1)
                else:
                    # Try to extract from first heading
                    heading_match = re.search(r'^# (.*?)$', content, re.MULTILINE)
                    if heading_match:
                        title = heading_match.group(1)

                print(f"- [{title}]({file_path})")

    # Add archive and other directories
    other_categories = ['archive']
    for category in other_categories:
        dir_path = os.path.join(PLANNING_DIR, category)
        if not os.path.exists(dir_path):
            continue

        print(f"\n## {category.title()}\n")

        for file_path in glob.glob(f"{dir_path}/*.md"):
            if os.path.basename(file_path) != "README.md":
                # Extract title if possible
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                title = os.path.basename(file_path).replace('.md', '')

                # Try to extract from metadata
                title_match = re.search(r'title: "([^"]+)"', content)
                if title_match:
                    title = title_match.group(1)
                else:
                    # Try to extract from first heading
                    heading_match = re.search(r'^# (.*?)$', content, re.MULTILINE)
                    if heading_match:
                        title = heading_match.group(1)

                print(f"- [{title}]({file_path})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WitsV3 Documentation Maintenance Tool")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Update metadata command
    update_parser = subparsers.add_parser("update", help="Update metadata in a document")
    update_parser.add_argument("file", help="Path to the markdown file")
    update_parser.add_argument("--title", help="Document title (optional)")
    update_parser.add_argument("--status", default="active", choices=["active", "archived", "draft"], help="Document status")

    # Archive document command
    archive_parser = subparsers.add_parser("archive", help="Archive a document")
    archive_parser.add_argument("file", help="Path to the markdown file")

    # Create document command
    create_parser = subparsers.add_parser("create", help="Create a new document")
    create_parser.add_argument("file", help="Path to the new markdown file")
    create_parser.add_argument("title", help="Document title")

    # List documents command
    list_parser = subparsers.add_parser("list", help="List all documents")

    args = parser.parse_args()

    if args.command == "update":
        update_metadata(args.file, args.title, args.status)
    elif args.command == "archive":
        archive_document(args.file)
    elif args.command == "create":
        create_document(args.file, args.title)
    elif args.command == "list":
        list_documents()
    else:
        parser.print_help()
