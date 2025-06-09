#!/usr/bin/env python3
"""
WitsV3 Fix Applier
This script applies fixes from the WitsV3 directory to the appropriate locations
"""

import os
import sys
import shutil
import time
import json
from pathlib import Path

def create_backup(file_path):
    """Create a backup of the file if it exists."""
    if not os.path.exists(file_path):
        return None

    backup_path = f"{file_path}.backup.{int(time.time())}"
    shutil.copy2(file_path, backup_path)
    print(f"✅ Created backup of {file_path} at {backup_path}")
    return backup_path

def apply_fixes():
    """Apply fixes from WitsV3 directory to appropriate locations."""
    print("Applying WitsV3 fixes...")

    # Verify source directory exists
    source_dir = Path("WitsV3")
    if not source_dir.exists() or not source_dir.is_dir():
        print(f"❌ Error: Source directory {source_dir} not found")
        return False

    # Files to copy
    files_to_copy = {
        "test_runner_script.py": "test_witsv3.py",
        "enhanced_run_py.py": "run_enhanced.py",
    }

    # Directories to ensure exist
    dirs_to_ensure = [
        "logs"
    ]

    # Create directories if they don't exist
    for dir_path in dirs_to_ensure:
        Path(dir_path).mkdir(exist_ok=True)
        print(f"✅ Ensured directory {dir_path} exists")

    # Copy files
    success = True
    for source_file, target_file in files_to_copy.items():
        source_path = source_dir / source_file
        target_path = Path(target_file)

        if not source_path.exists():
            print(f"⚠️ Warning: Source file {source_path} not found, skipping")
            continue

        try:
            # Create backup of target file if it exists
            if target_path.exists():
                create_backup(target_path)

            # Copy file
            shutil.copy2(source_path, target_path)
            print(f"✅ Copied {source_path} to {target_path}")

            # Make file executable
            target_path.chmod(target_path.stat().st_mode | 0o111)  # Add executable bit
            print(f"✅ Made {target_path} executable")

        except Exception as e:
            print(f"❌ Error copying {source_path} to {target_path}: {e}")
            success = False

    # Run fix_run_py.py if it exists
    if os.path.exists("fix_run_py.py"):
        try:
            print("\nRunning fix_run_py.py to update run.py...")
            # Make it executable
            os.chmod("fix_run_py.py", os.stat("fix_run_py.py").st_mode | 0o111)
            # Run the script
            exit_code = os.system("python fix_run_py.py")
            if exit_code != 0:
                print(f"⚠️ Warning: fix_run_py.py exited with code {exit_code}")
                success = False
        except Exception as e:
            print(f"❌ Error running fix_run_py.py: {e}")
            success = False

    return success

def update_launch_config():
    """Update VS Code launch configuration if it exists."""
    vscode_dir = Path(".vscode")
    launch_file = vscode_dir / "launch.json"

    if not vscode_dir.exists():
        vscode_dir.mkdir()
        print(f"✅ Created .vscode directory")

    source_file = Path("WitsV3") / "updated_launch_config.json"
    if not source_file.exists():
        print(f"⚠️ Warning: Source launch config {source_file} not found, skipping")
        return False

    try:
        # Create backup of existing launch.json if it exists
        if launch_file.exists():
            create_backup(launch_file)

        # Copy the updated launch config
        shutil.copy2(source_file, launch_file)
        print(f"✅ Updated VS Code launch configuration at {launch_file}")
        return True
    except Exception as e:
        print(f"❌ Error updating launch configuration: {e}")
        return False

def update_cursor_config():
    """Update Cursor environment configuration."""
    cursor_dir = Path(".cursor")
    if not cursor_dir.exists():
        cursor_dir.mkdir()
        print(f"✅ Created .cursor directory")

    # Config files to apply
    config_files = {
        "cursor_environment_fix.json": ".cursor/environment.json",
        "Dockerfile": ".cursor/Dockerfile"
    }

    success = True
    for source_file, target_path in config_files.items():
        source_path = Path("WitsV3") / source_file
        target_path = Path(target_path)

        if not source_path.exists():
            print(f"⚠️ Warning: Source file {source_path} not found, skipping")
            continue

        try:
            # Create parent directory if it doesn't exist
            target_path.parent.mkdir(parents=True, exist_ok=True)

            # Create backup of existing file if it exists
            if target_path.exists():
                create_backup(target_path)

            # Copy the file
            shutil.copy2(source_path, target_path)
            print(f"✅ Updated Cursor config at {target_path}")
        except Exception as e:
            print(f"❌ Error updating Cursor config {target_path}: {e}")
            success = False

    return success

def main():
    """Main entry point."""
    print("WitsV3 Fix Applier")
    print("=" * 50)

    fixes_success = apply_fixes()
    launch_success = update_launch_config()
    cursor_success = update_cursor_config()

    if fixes_success and launch_success and cursor_success:
        print("\n✅ All fixes applied successfully!")
        print("\nYou can now run:")
        print("1. python test_witsv3.py           - Run non-interactive tests")
        print("2. python run.py --test            - Run test mode")
        print("3. python run_enhanced.py          - Run with enhanced features")
        return 0
    elif fixes_success:
        print("\n⚠️ Fixes applied but some configuration updates failed")
        return 1
    else:
        print("\n❌ Failed to apply fixes")
        return 1

if __name__ == "__main__":
    sys.exit(main())
