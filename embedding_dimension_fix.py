#!/usr/bin/env python3
"""
Fix for WitsV3 Embedding Dimension Mismatch
This script fixes the memory search issue where embeddings of different dimensions are being compared
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

def fix_embedding_dimensions():
    """Fix the memory search issue by updating the search_segments method to handle dimension mismatches."""
    memory_manager_path = "core/memory_manager.py"

    if not os.path.exists(memory_manager_path):
        print(f"❌ Error: {memory_manager_path} not found")
        return False

    # Create backup
    create_backup(memory_manager_path)

    # Read the file
    with open(memory_manager_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Find the similarity calculation part in search_segments method
    # Look for: similarity = np.dot(query_np, segment_np) / (np.linalg.norm(query_np) * np.linalg.norm(segment_np))
    similarity_pattern = r'segment_np = np\.array\(segment\.embedding\)\n                # Cosine similarity\n                similarity = np\.dot\(query_np, segment_np\)'

    if similarity_pattern in content:
        # Add dimension check before similarity calculation
        dimension_check = """                segment_np = np.array(segment.embedding)

                # Check for dimension mismatch
                if query_np.shape[0] != segment_np.shape[0]:
                    self.logger.warning(f"Embedding dimension mismatch: query {query_np.shape[0]} != segment {segment_np.shape[0]}. Skipping this segment.")
                    continue

                # Cosine similarity"""

        # Replace the old code with the updated one
        updated_content = content.replace(
            "                segment_np = np.array(segment.embedding)\n                # Cosine similarity",
            dimension_check
        )

        # Write the updated content back to the file
        with open(memory_manager_path, 'w', encoding='utf-8') as file:
            file.write(updated_content)

        print(f"✅ Updated {memory_manager_path}: Added dimension check to search_segments method")
        return True
    else:
        print(f"❌ Error: similarity calculation not found in {memory_manager_path}")
        return False

def main():
    print("Fixing WitsV3 Embedding Dimension Mismatch...")
    success = fix_embedding_dimensions()
    if success:
        print("✅ Embedding dimension fix applied successfully")
    else:
        print("❌ Failed to apply embedding dimension fix")

if __name__ == "__main__":
    main()
