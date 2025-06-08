#!/usr/bin/env python
"""
Script to set up MCP servers from the command line.
This is a user-friendly wrapper around the clone_mcp_servers.py script.
"""

import os
import sys
import subprocess
import asyncio
import argparse
from pathlib import Path

def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"{title.center(60)}")
    print("=" * 60 + "\n")

def print_step(step_num, total_steps, description):
    """Print a formatted step."""
    print(f"[{step_num}/{total_steps}] {description}...")

async def main():
    parser = argparse.ArgumentParser(description="Set up MCP servers for WitsV3")
    parser.add_argument("--config", default="data/mcp_tools.json", help="Path to MCP configuration file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show verbose output")
    args = parser.parse_args()

    print_header("WitsV3 MCP Server Setup")

    # Ensure we're in the project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)

    print_step(1, 4, "Checking for existing MCP servers")
    mcp_servers_dir = Path("mcp_servers")
    if mcp_servers_dir.exists() and any(mcp_servers_dir.iterdir()):
        print(f"Found existing MCP servers in {mcp_servers_dir}")
    else:
        print(f"No existing MCP servers found, will clone fresh")
        mcp_servers_dir.mkdir(exist_ok=True)

    print_step(2, 4, "Running clone_mcp_servers.py script")
    clone_script = script_dir / "clone_mcp_servers.py"
    if not clone_script.exists():
        print(f"Error: Could not find {clone_script}")
        return 1

    try:
        cmd = [sys.executable, str(clone_script), "--config", args.config]
        if args.verbose:
            process = subprocess.Popen(cmd)
        else:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

        return_code = process.wait()

        if return_code != 0:
            if not args.verbose:
                stdout, stderr = process.communicate()
                if stderr:
                    print(f"Error: {stderr.decode()}")
            print("Failed to clone MCP servers")
            return 1
    except Exception as e:
        print(f"Error running clone script: {e}")
        return 1

    print_step(3, 4, "Verifying MCP server installation")
    success = True
    config_path = Path(args.config)

    if not config_path.exists():
        print(f"Error: Configuration file {config_path} not found")
        return 1

    try:
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)

        for server in config.get("servers", []):
            name = server.get("name", "")
            if "working_directory" in server and Path(server["working_directory"]).exists():
                print(f"✓ {name} is properly configured")
            else:
                print(f"✗ {name} is not properly configured")
                success = False
    except Exception as e:
        print(f"Error verifying installation: {e}")
        success = False

    print_step(4, 4, "Setup complete")

    if success:
        print_header("SUCCESS")
        print("All MCP servers have been successfully set up.")
        print("You can now run WitsV3 with: python run.py")
    else:
        print_header("PARTIAL SUCCESS")
        print("Some MCP servers may not have been properly set up.")
        print("You can try running this script again with --verbose for more details.")
        print("Or check the logs for more information.")

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
