#!/usr/bin/env python3
"""
Fix for WitsV3 Memory Search Issue
This script fixes the memory search issue where AdaptiveLLMInterface.get_embedding() is being called with 'model' parameter
"""

import os
import sys
import shutil
import re
from pathlib import Path

def create_backup(file_path):
    """Create a backup of the original file."""
    backup_path = f"{file_path}.backup"

    # Check if backup already exists
    if not os.path.exists(backup_path):
        shutil.copy2(file_path, backup_path)
        print(f"✅ Created backup at {backup_path}")
    else:
        print(f"ℹ️ Backup already exists at {backup_path}")

    return backup_path

def fix_memory_search():
    """Fix the memory search issue by updating the AdaptiveLLMInterface get_embedding method."""
    adaptive_llm_interface_path = "core/adaptive_llm_interface.py"

    if not os.path.exists(adaptive_llm_interface_path):
        print(f"❌ Error: {adaptive_llm_interface_path} not found")
        return False

    # Create backup
    create_backup(adaptive_llm_interface_path)

    # Read the file
    with open(adaptive_llm_interface_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Find the get_embedding method
    get_embedding_pattern = r'async def get_embedding\(self,([^)]*)\)'
    match = re.search(get_embedding_pattern, content)

    if match:
        # Extract the current parameters
        current_params = match.group(1)

        # Check if 'model' parameter already exists
        if 'model' in current_params:
            print("ℹ️ 'model' parameter already exists in get_embedding method")
            return True

        # Add the model parameter with a default value of None
        if current_params.strip():
            # There are other parameters, add model as an optional parameter
            new_params = current_params + ", model=None"
        else:
            # No other parameters, add model as the first parameter
            new_params = " model=None"

        # Replace the old parameter list with the new one
        updated_content = content.replace(
            f"async def get_embedding(self,{current_params})",
            f"async def get_embedding(self,{new_params})"
        )

        # Add a comment to explain the model parameter
        comment = "        # 'model' parameter is ignored but accepted for compatibility with MemoryManager\n"
        # Find the first line after the method declaration
        method_body_start = updated_content.find('\n', match.start()) + 1
        updated_content = updated_content[:method_body_start] + comment + updated_content[method_body_start:]

        # Write the updated content back to the file
        with open(adaptive_llm_interface_path, 'w', encoding='utf-8') as file:
            file.write(updated_content)

        print(f"✅ Updated {adaptive_llm_interface_path}: Added 'model' parameter to get_embedding method")
        return True
    else:
        print(f"❌ Error: get_embedding method not found in {adaptive_llm_interface_path}")
        return False

def main():
    print("Fixing WitsV3 Memory Search Issue...")
    success = fix_memory_search()
    if success:
        print("✅ Memory search fix applied successfully")
    else:
        print("❌ Failed to apply memory search fix")

if __name__ == "__main__":
    main()
