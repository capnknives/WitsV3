#!/usr/bin/env python3
"""
Setup Local Data Files for WitsV3
Initializes memory and neural web files from templates if they don't exist
"""

import os
import shutil
import sys
from pathlib import Path

# Ensure emoji output works on Windows consoles (cp1252)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from core.runtime_paths import data_dir, ensure_runtime_layout


def setup_local_data():
    """Initialize local data files from templates under var/data/."""
    print("🔧 Setting up WitsV3 local data files...")

    ensure_runtime_layout()
    data_path = data_dir()

    # Files to initialize from templates
    files_to_setup = [
        ("wits_memory.json.template", "wits_memory.json"),
        ("neural_web.json.template", "neural_web.json"),
    ]

    for template_name, target_name in files_to_setup:
        template_path = data_path / template_name
        target_path = data_path / target_name

        if template_path.exists() and not target_path.exists():
            print(f"📄 Creating {target_name} from template...")
            shutil.copy(template_path, target_path)
        elif target_path.exists():
            print(f"✅ {target_name} already exists")
        else:
            print(f"⚠️  Template {template_name} not found")

    # Migrate legacy repo-root data/knowledge_log.json → var/data/
    legacy_klog = Path("data/knowledge_log.json")
    target_klog = data_path / "knowledge_log.json"
    if legacy_klog.exists() and not target_klog.exists():
        print("📄 Migrating data/knowledge_log.json → var/data/knowledge_log.json...")
        target_klog.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(legacy_klog, target_klog)

    print("\n🎉 Local data setup complete!")
    print("\n📝 Runtime data lives under var/ (data, documents, exports, logs, workspace, cache).")
    print("Personal files are gitignored. To reset memory, delete var/data/wits_memory.json "
          "and var/data/neural_web.json — they will be recreated from templates.")


if __name__ == "__main__":
    setup_local_data()
