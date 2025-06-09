#!/usr/bin/env python3
"""
Fix Unicode Encoding Issues in WitsV3 Logs
This script modifies the logging configuration to properly handle Unicode characters (emojis)
"""

import os
import sys
import re
import codecs
from pathlib import Path

def fix_logging_in_file(file_path):
    """Fix logging configuration in a specific file to handle Unicode properly."""
    if not os.path.exists(file_path):
        print(f"❌ Error: {file_path} not found")
        return False

    # Create backup
    backup_path = f"{file_path}.backup.logging"
    if not os.path.exists(backup_path):
        with open(file_path, 'rb') as src, open(backup_path, 'wb') as dst:
            dst.write(src.read())
        print(f"✅ Created backup at {backup_path}")

    # Read the file content
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Look for logging configuration pattern
    basic_logging_pattern = r'logging\.basicConfig\s*\(([^)]*)\)'
    match = re.search(basic_logging_pattern, content, re.DOTALL)

    if match:
        log_config = match.group(1)

        # Check if encoding is already specified
        if 'encoding' not in log_config:
            # Add encoding='utf-8' parameter to the logging.basicConfig call
            modified_config = log_config.rstrip() + ',\n    encoding=\'utf-8\'\n'
            modified_content = content.replace(log_config, modified_config)

            # Write the modified content back to the file
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(modified_content)

            print(f"✅ Updated {file_path}: Added UTF-8 encoding to logging configuration")
            return True
        else:
            print(f"ℹ️ {file_path} already has encoding specified in logging configuration")
            return True
    else:
        print(f"❌ Error: Could not find logging.basicConfig in {file_path}")
        return False

def update_system_encoding():
    """Add code to set system encoding to utf-8 for Python in run scripts."""
    run_files = ['run.py', 'run_test.py', 'test_witsv3.py', 'llm_diagnostic.py', 'llm_diagnostic_basic.py']

    for file_path in run_files:
        if not os.path.exists(file_path):
            print(f"ℹ️ File {file_path} not found, skipping")
            continue

        # Create backup
        backup_path = f"{file_path}.backup.encoding"
        if not os.path.exists(backup_path):
            with open(file_path, 'rb') as src, open(backup_path, 'wb') as dst:
                dst.write(src.read())
            print(f"✅ Created backup at {backup_path}")

        # Read the file content
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # Look for import statements
        import_section_end = re.search(r'import [^\n]+', content)
        if import_section_end:
            # Add UTF-8 encoding fix after imports
            encoding_fix = """
# Fix Unicode encoding issues
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

import sys
import codecs
# Set UTF-8 encoding for stdout/stderr
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)
"""
            pos = import_section_end.end()
            modified_content = content[:pos] + encoding_fix + content[pos:]

            # Write the modified content back to the file
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(modified_content)

            print(f"✅ Updated {file_path}: Added UTF-8 encoding configuration")
        else:
            print(f"❌ Error: Could not find import section in {file_path}")

def main():
    print("Fixing Unicode encoding issues in WitsV3 logs...")

    # Fix logging configuration in test files
    files_to_fix = ['test_witsv3.py', 'llm_diagnostic.py', 'llm_diagnostic_basic.py']
    for file_path in files_to_fix:
        fix_logging_in_file(file_path)

    # Update system encoding in run scripts
    update_system_encoding()

    print("✅ Unicode encoding fixes applied")

if __name__ == "__main__":
    main()
