#!/usr/bin/env python3
"""
Create Dummy Model Files for WitsV3
This script creates placeholder files for missing model files like creative_expert.safetensors
"""

import os
import sys
import numpy as np
from pathlib import Path

def create_dummy_file(file_path, size_kb=10):
    """Create a dummy binary file of specified size."""
    # Create parent directories if they don't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Create a small random tensor file
    dummy_data = np.random.bytes(size_kb * 1024)

    with open(file_path, 'wb') as f:
        f.write(dummy_data)

    print(f"✅ Created dummy file at {file_path} ({size_kb} KB)")
    return True

def create_model_files():
    """Create dummy model files that are referenced but missing."""
    # Define paths for model files
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    files_to_create = [
        model_dir / "creative_expert.safetensors",
        model_dir / "reasoning_expert.safetensors",
        model_dir / "planning_expert.safetensors"
    ]

    for file_path in files_to_create:
        if not file_path.exists():
            create_dummy_file(file_path)
        else:
            print(f"ℹ️ File {file_path} already exists, skipping")

def main():
    print("Creating dummy model files for WitsV3...")
    create_model_files()
    print("✅ All dummy model files created")

if __name__ == "__main__":
    main()
