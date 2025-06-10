#!/usr/bin/env python3
"""
Script to standardize the format of all migrated markdown files.
"""

import os
import re
import datetime

# Directories to process
DIRECTORIES = [
    "planning/architecture",
    "planning/implementation",
    "planning/roadmap",
    "planning/tasks",
    "planning/technical-notes"
]

def standardize_file(file_path):
    """Standardize the format of a markdown file."""
    print(f"Processing: {file_path}")

    # Read file content
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Create backup
    backup_path = f"{file_path}.format.bak"
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)

    # Check if file already has metadata
    has_metadata = content.startswith('---')

    # Extract and update metadata if it exists
    if has_metadata:
        # Find end of metadata
        metadata_end = content.find('---', 3)
        if metadata_end != -1:
            metadata = content[0:metadata_end+3]
            body = content[metadata_end+3:].strip()

            # Update last_updated date
            now = datetime.datetime.now().strftime("%Y-%m-%d")
            metadata = re.sub(r'last_updated: ".*?"', f'last_updated: "{now}"', metadata)
        else:
            # Malformed metadata, treat as no metadata
            metadata = ""
            body = content
            has_metadata = False
    else:
        metadata = ""
        body = content

    # Extract title from first heading if no metadata
    if not has_metadata:
        title_match = re.search(r'^# (.*?)$', body, re.MULTILINE)
        if title_match:
            title = title_match.group(1)
        else:
            # Use filename as title if no heading
            basename = os.path.basename(file_path)
            title = os.path.splitext(basename)[0].replace('-', ' ').replace('_', ' ').title()

        # Create metadata
        now = datetime.datetime.now().strftime("%Y-%m-%d")
        metadata = f"""---
title: "{title}"
created: "{now}"
last_updated: "{now}"
status: "active"
---

"""

    # Add table of contents if file is large enough
    lines = body.split('\n')
    if len(lines) > 20:  # Only add TOC for larger files
        # Check if TOC already exists
        toc_exists = False
        for i, line in enumerate(lines):
            if line.lower().startswith('## table of contents') or line.lower().startswith('## contents'):
                toc_exists = True
                break

        if not toc_exists:
            # Find all headings
            headings = []
            for line in lines:
                if line.startswith('## '):
                    heading = line[3:].strip()
                    anchor = heading.lower().replace(' ', '-').replace('&', '').replace(':', '')
                    headings.append((heading, anchor))

            # Create TOC if headings found
            if headings:
                toc = "## Table of Contents\n\n"
                for heading, anchor in headings:
                    toc += f"- [{heading}](#{anchor})\n"
                toc += "\n"

                # Insert TOC after first heading
                first_heading_index = -1
                for i, line in enumerate(lines):
                    if line.startswith('# '):
                        first_heading_index = i
                        break

                if first_heading_index != -1:
                    lines.insert(first_heading_index + 1, "\n" + toc)
                    body = '\n'.join(lines)

    # Write standardized content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(metadata + body)

    print(f"Standardized: {file_path}")

def process_directories():
    """Process all files in specified directories."""
    for directory in DIRECTORIES:
        if not os.path.exists(directory):
            print(f"Directory {directory} does not exist. Skipping.")
            continue

        for filename in os.listdir(directory):
            if filename.endswith('.md') and filename != 'README.md':
                file_path = os.path.join(directory, filename)
                standardize_file(file_path)

if __name__ == "__main__":
    print("Starting format standardization...")
    process_directories()
    print("Format standardization complete!")
