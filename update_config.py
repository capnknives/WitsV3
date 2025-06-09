#!/usr/bin/env python3
"""
Update WitsV3 Configuration
This script updates the config.yaml file to use ollama as the default provider
"""

import os
import sys
import re
from pathlib import Path
import yaml

def update_config():
    """Update the config.yaml file to use ollama as the default provider."""
    config_path = Path("config.yaml")

    if not config_path.exists():
        print(f"❌ Error: {config_path} not found")
        return False

    # Create backup
    backup_path = Path(f"{config_path}.backup")
    if not backup_path.exists():
        import shutil
        shutil.copy2(config_path, backup_path)
        print(f"✅ Created backup at {backup_path}")

    try:
        # Load YAML file
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)

        # Check if the config has llm_interface section
        if 'llm_interface' in config:
            # Update default_provider from 'adaptive' to 'ollama'
            if config['llm_interface'].get('default_provider') == 'adaptive':
                config['llm_interface']['default_provider'] = 'ollama'
                print("✅ Changed default_provider from 'adaptive' to 'ollama'")
            else:
                print(f"ℹ️ default_provider is already set to '{config['llm_interface']['default_provider']}'")
        else:
            print("❌ Error: llm_interface section not found in config.yaml")
            return False

        # Write updated config back to file
        with open(config_path, 'w', encoding='utf-8') as file:
            yaml.dump(config, file, default_flow_style=False)

        print(f"✅ Updated {config_path} successfully")
        return True

    except Exception as e:
        print(f"❌ Error updating config: {str(e)}")
        return False

def main():
    print("Updating WitsV3 configuration...")
    update_config()
    print("Configuration update complete")

if __name__ == "__main__":
    main()
