#!/usr/bin/env python3
"""
Script to update the main README.md with information about the new documentation structure.
"""

import os
import re

README_PATH = "README.md"

# Documentation section to insert
DOCS_SECTION = """
## Documentation

All project documentation is now organized in the `/planning` directory:

- **[Architecture](planning/architecture/)** - System design and components
- **[Implementation](planning/implementation/)** - Implementation details
- **[Roadmap](planning/roadmap/)** - Future plans and enhancements
- **[Tasks](planning/tasks/)** - Task tracking and management
- **[Technical Notes](planning/technical-notes/)** - Debug info and fixes

See the [Planning Documentation](planning/README.md) for more details.
"""

def update_readme():
    """Update the README.md with documentation section."""
    # Check if README.md exists
    if not os.path.exists(README_PATH):
        print(f"Error: {README_PATH} not found.")
        return False

    # Read current README
    with open(README_PATH, 'r', encoding='utf-8') as f:
        content = f.read()

    # Create backup
    with open(f"{README_PATH}.bak", 'w', encoding='utf-8') as f:
        f.write(content)

    # Check if Documentation section already exists
    docs_pattern = re.compile(r'^## Documentation\s*$', re.MULTILINE)
    match = docs_pattern.search(content)

    if match:
        # Replace existing Documentation section
        section_start = match.start()

        # Find the next section heading
        next_section = re.search(r'^##\s+', content[section_start+1:], re.MULTILINE)
        if next_section:
            section_end = section_start + 1 + next_section.start()
            new_content = content[:section_start] + DOCS_SECTION + content[section_end:]
        else:
            # If no next section, append to the end
            new_content = content[:section_start] + DOCS_SECTION
    else:
        # Add new Documentation section after Installation or Usage section
        install_pattern = re.compile(r'^## (?:Installation|Usage|Getting Started)\s*$', re.MULTILINE)
        install_match = install_pattern.search(content)

        if install_match:
            # Find the next section heading after Installation/Usage
            section_start = install_match.start()
            next_section = re.search(r'^##\s+', content[section_start+1:], re.MULTILINE)
            if next_section:
                insert_point = section_start + 1 + next_section.start()
                new_content = content[:insert_point] + DOCS_SECTION + content[insert_point:]
            else:
                # If no next section, append to the end
                new_content = content + "\n" + DOCS_SECTION
        else:
            # If no Installation/Usage section, append to the end
            new_content = content + "\n" + DOCS_SECTION

    # Write updated README
    with open(README_PATH, 'w', encoding='utf-8') as f:
        f.write(new_content)

    print(f"Updated {README_PATH} with documentation section.")
    return True

if __name__ == "__main__":
    update_readme()
