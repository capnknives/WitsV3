#!/usr/bin/env python
"""
Script to clone and set up MCP servers from GitHub repositories.
This helps with the automatic setup of Model Context Protocol servers used by WitsV3.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("WitsV3.MCPSetup")

# Directory where MCP servers will be cloned
MCP_SERVERS_DIR = Path("mcp_servers")

async def clone_github_repo(repo_url: str, target_dir: Path) -> bool:
    """
    Clone a GitHub repository to the specified directory.

    Args:
        repo_url: The GitHub repository URL to clone
        target_dir: The directory to clone the repository to

    Returns:
        True if cloning was successful, False otherwise
    """
    logger.info(f"Cloning {repo_url} to {target_dir}")

    # Create the target directory if it doesn't exist
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Run git clone command
        process = await asyncio.create_subprocess_exec(
            "git", "clone", repo_url, str(target_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            error_msg = stderr.decode().strip() if stderr else "Unknown error"
            logger.error(f"Failed to clone {repo_url}: {error_msg}")
            return False

        logger.info(f"Successfully cloned {repo_url}")
        return True
    except Exception as e:
        logger.error(f"Error cloning {repo_url}: {e}")
        return False

async def install_dependencies(repo_dir: Path) -> bool:
    """
    Install dependencies for the cloned repository.

    Args:
        repo_dir: The directory containing the cloned repository

    Returns:
        True if dependency installation was successful, False otherwise
    """
    logger.info(f"Installing dependencies in {repo_dir}")

    try:
        # Check for package.json (Node.js)
        if (repo_dir / "package.json").exists():
            logger.info(f"Found package.json, installing npm dependencies")
            process = await asyncio.create_subprocess_exec(
                "npm", "install",
                cwd=str(repo_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode().strip() if stderr else "Unknown error"
                logger.error(f"Failed to install npm dependencies: {error_msg}")
                return False

            logger.info("Successfully installed npm dependencies")

        # Check for requirements.txt (Python)
        elif (repo_dir / "requirements.txt").exists():
            logger.info(f"Found requirements.txt, installing Python dependencies")
            process = await asyncio.create_subprocess_exec(
                "pip", "install", "-r", "requirements.txt",
                cwd=str(repo_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode().strip() if stderr else "Unknown error"
                logger.error(f"Failed to install Python dependencies: {error_msg}")
                return False

            logger.info("Successfully installed Python dependencies")
        else:
            logger.info("No package.json or requirements.txt found, skipping dependency installation")

        return True
    except Exception as e:
        logger.error(f"Error installing dependencies: {e}")
        return False

async def update_mcp_config(config_path: Path, cloned_repos: Dict[str, Path]) -> bool:
    """
    Update the MCP configuration with the cloned repositories.

    Args:
        config_path: Path to the MCP configuration file
        cloned_repos: Dictionary mapping repository names to their cloned paths

    Returns:
        True if the configuration was updated successfully, False otherwise
    """
    logger.info(f"Updating MCP configuration at {config_path}")

    try:
        if not config_path.exists():
            logger.error(f"Configuration file {config_path} does not exist")
            return False

        # Load existing configuration
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Update server configurations
        for server in config.get("servers", []):
            name = server.get("name", "")
            repo_name = name.split("/")[-1].split(".git")[0]

            if repo_name in cloned_repos:
                repo_path = cloned_repos[repo_name]

                # Update working_directory for this server
                server["working_directory"] = str(repo_path)

                # Try to detect the appropriate command to run
                if (repo_path / "build" / "index.js").exists():
                    server["command"] = "node .\\build\\index.js"
                elif (repo_path / "index.js").exists():
                    server["command"] = "node .\\index.js"
                elif (repo_path / "index.ts").exists():
                    server["command"] = "npx ts-node .\\index.ts"
                elif (repo_path / "src" / "index.js").exists():
                    server["command"] = "node .\\src\\index.js"
                elif (repo_path / "main.py").exists():
                    server["command"] = "python main.py"

                logger.info(f"Updated configuration for {name}")

        # Save updated configuration
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Successfully updated MCP configuration")
        return True
    except Exception as e:
        logger.error(f"Error updating MCP configuration: {e}")
        return False

async def main():
    parser = argparse.ArgumentParser(description="Clone and set up MCP servers")
    parser.add_argument("--config", default="data/mcp_tools.json", help="Path to MCP configuration file")
    args = parser.parse_args()

    config_path = Path(args.config)

    # Ensure the MCP servers directory exists
    MCP_SERVERS_DIR.mkdir(parents=True, exist_ok=True)

    # Load configuration
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}")
        return 1

    # Extract GitHub repositories from configuration
    github_repos = []
    for server in config.get("servers", []):
        if server.get("type") == "github" and "clone_url" in server:
            github_repos.append((server["name"], server["clone_url"]))

    if not github_repos:
        logger.error("No GitHub repositories found in configuration")
        return 1

    logger.info(f"Found {len(github_repos)} GitHub repositories to clone")

    # Clone repositories
    cloned_repos = {}
    for name, url in github_repos:
        repo_name = url.split("/")[-1].split(".git")[0]
        target_dir = MCP_SERVERS_DIR / repo_name

        # Skip if already cloned
        if target_dir.exists() and any(target_dir.iterdir()):
            logger.info(f"Repository {repo_name} already cloned, skipping")
            cloned_repos[repo_name] = target_dir
            continue

        # Clone repository
        success = await clone_github_repo(url, target_dir)
        if success:
            # Install dependencies
            await install_dependencies(target_dir)
            cloned_repos[repo_name] = target_dir

    # Update MCP configuration
    await update_mcp_config(config_path, cloned_repos)

    # Special handling for specific repositories
    if "servers" in cloned_repos:
        # Handle the servers repo with subdirectories
        servers_dir = cloned_repos["servers"]

        # Install dependencies in src/sequentialthinking
        seq_thinking_dir = servers_dir / "src" / "sequentialthinking"
        if seq_thinking_dir.exists():
            await install_dependencies(seq_thinking_dir)

        # Install dependencies in src/filesystem
        filesystem_dir = servers_dir / "src" / "filesystem"
        if filesystem_dir.exists():
            await install_dependencies(filesystem_dir)

    logger.info("MCP server setup complete")
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
