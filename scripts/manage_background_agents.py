#!/usr/bin/env python3
"""
Script to manage WitsV3 background agents in Docker
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("WitsV3.BackgroundManager")

class BackgroundAgentManager:
    """Manages background agents in Docker containers"""
    
    def __init__(self, config_path: str = "config.yaml"):
        # Get the project root directory (where this script is located)
        self.project_root = Path(__file__).parent.parent.absolute()
        
        # Set absolute paths for configuration files
        self.config_path = str(self.project_root / config_path)
        self.docker_compose_path = str(self.project_root / "docker-compose.background.yml")
        
        # Verify files exist
        if not os.path.exists(self.docker_compose_path):
            raise FileNotFoundError(f"Docker Compose file not found at: {self.docker_compose_path}")
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found at: {self.config_path}")
            
        logger.info(f"Using Docker Compose file: {self.docker_compose_path}")
        logger.info(f"Using config file: {self.config_path}")
        
    async def start_agents(self, num_instances: Optional[int] = None) -> None:
        """Start background agents"""
        try:
            # Build the Docker image
            logger.info("Building background agent Docker image...")
            subprocess.run(
                ["docker-compose", "-f", self.docker_compose_path, "build"],
                check=True,
                cwd=self.project_root  # Set working directory to project root
            )
            
            # Start the containers
            logger.info("Starting background agents...")
            if num_instances:
                subprocess.run(
                    ["docker-compose", "-f", self.docker_compose_path, "up", "-d", "--scale", f"background-agent={num_instances}"],
                    check=True,
                    cwd=self.project_root  # Set working directory to project root
                )
            else:
                subprocess.run(
                    ["docker-compose", "-f", self.docker_compose_path, "up", "-d"],
                    check=True,
                    cwd=self.project_root  # Set working directory to project root
                )
            
            logger.info("Background agents started successfully")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error starting background agents: {e}")
            raise
    
    async def stop_agents(self) -> None:
        """Stop all background agents"""
        try:
            logger.info("Stopping background agents...")
            subprocess.run(
                ["docker-compose", "-f", self.docker_compose_path, "down"],
                check=True,
                cwd=self.project_root  # Set working directory to project root
            )
            logger.info("Background agents stopped successfully")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error stopping background agents: {e}")
            raise
    
    async def restart_agents(self) -> None:
        """Restart all background agents"""
        await self.stop_agents()
        await self.start_agents()
    
    async def get_agent_status(self) -> List[Dict[str, Any]]:
        """Get status of all background agents"""
        try:
            result = subprocess.run(
                ["docker-compose", "-f", self.docker_compose_path, "ps", "--format", "json"],
                capture_output=True,
                text=True,
                check=True,
                cwd=self.project_root  # Set working directory to project root
            )
            
            # Parse JSON output
            status_list = []
            for line in result.stdout.strip().split("\n"):
                if line:
                    status_list.append(json.loads(line))
            return status_list
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error getting agent status: {e}")
            raise

async def main():
    parser = argparse.ArgumentParser(description="Manage WitsV3 background agents")
    parser.add_argument("action", choices=["start", "stop", "restart", "status"])
    parser.add_argument("--instances", type=int, help="Number of agent instances to start")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    
    args = parser.parse_args()
    
    try:
        manager = BackgroundAgentManager(args.config)
        
        if args.action == "start":
            await manager.start_agents(args.instances)
        elif args.action == "stop":
            await manager.stop_agents()
        elif args.action == "restart":
            await manager.restart_agents()
        elif args.action == "status":
            status = await manager.get_agent_status()
            print("\nBackground Agent Status:")
            for container in status:
                print(f"- {container}")
    
    except FileNotFoundError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 