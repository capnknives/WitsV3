#!/usr/bin/env python3
"""
Setup Local Data Files for WitsV3
Initializes memory and neural web files from templates if they don't exist
"""

import os
import shutil
from pathlib import Path

def setup_local_data():
    """Initialize local data files from templates"""
    print("🔧 Setting up WitsV3 local data files...")

    data_dir = Path("data")

    # Files to initialize from templates
    files_to_setup = [
        ("wits_memory.json.template", "wits_memory.json"),
        ("neural_web.json.template", "neural_web.json")
    ]

    for template_name, target_name in files_to_setup:
        template_path = data_dir / template_name
        target_path = data_dir / target_name

        if template_path.exists() and not target_path.exists():
            print(f"📄 Creating {target_name} from template...")
            shutil.copy(template_path, target_path)
        elif target_path.exists():
            print(f"✅ {target_name} already exists")
        else:
            print(f"⚠️  Template {template_name} not found")

    # Create directories if they don't exist
    directories_to_create = [
        "logs",
        "cache"
    ]

    for dir_name in directories_to_create:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            print(f"📁 Creating directory: {dir_name}")
            dir_path.mkdir(exist_ok=True)
        else:
            print(f"✅ Directory {dir_name} already exists")

    print("\n🎉 Local data setup complete!")
    print("\n📝 Note: These files contain personal data and are excluded from git tracking.")
    print("To reset your memory, delete wits_memory.json and neural_web.json - they will be recreated.")

if __name__ == "__main__":
    setup_local_data()
