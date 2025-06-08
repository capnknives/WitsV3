"""
Network Access Control Tool for WitsV3
Allows authorized users to enable/disable network access for Python execution
"""

import logging
import yaml
from typing import Any, Dict, Optional
from core.base_tool import BaseTool
from core.config import load_config
from core.personality_manager import get_personality_manager
from core.auth_manager import verify_auth

logger = logging.getLogger(__name__)

class NetworkControlTool(BaseTool):
    """Tool for controlling network access permissions."""

    def __init__(self):
        """Initialize the network control tool."""
        super().__init__(
            name="network_control",
            description="Control network access permissions for Python execution tool (authorized users only)"
        )
        self.config = load_config()
        self.personality_manager = get_personality_manager()

    async def execute(self, action: str, user_id: str = "default", duration_minutes: int = 60, auth_token: str = "") -> Dict[str, Any]:
        """
        Execute network control action.

        Args:
            action: Action to perform ('enable_network', 'disable_network', 'status')
            user_id: User requesting the action
            duration_minutes: Duration for temporary access (if applicable)
            auth_token: Authentication token for secure operations

        Returns:
            Dictionary containing operation result
        """
        try:
            # Check authorization (both user ID and authentication token)
            is_authorized, auth_message = verify_auth(user_id, auth_token, "network_control")
            if not is_authorized:
                return {
                    "success": False,
                    "message": f"Unauthorized: {auth_message}",
                    "current_status": self._get_current_status()
                }

            if action == "enable_network":
                return await self._enable_network(user_id, duration_minutes)
            elif action == "disable_network":
                return await self._disable_network(user_id)
            elif action == "status":
                return self._get_status()
            else:
                return {
                    "success": False,
                    "message": f"Unknown action: {action}. Valid actions: enable_network, disable_network, status"
                }

        except Exception as e:
            logger.error(f"Error in network control: {e}")
            return {
                "success": False,
                "message": f"Error: {str(e)}"
            }

    async def _enable_network(self, user_id: str, duration_minutes: int) -> Dict[str, Any]:
        """Enable network access for Python execution"""
        try:
            # Update configuration in memory
            self.config.security.python_execution_network_access = True

            # Also update the YAML file for persistence
            try:
                with open("config.yaml", 'r') as f:
                    config_data = yaml.safe_load(f)

                if 'security' not in config_data:
                    config_data['security'] = {}
                config_data['security']['python_execution_network_access'] = True

                with open("config.yaml", 'w') as f:
                    yaml.safe_dump(config_data, f, default_flow_style=False)

                logger.info("Updated config.yaml with network access enabled")
            except Exception as e:
                logger.warning(f"Could not update config.yaml: {e}")

            # Log the action
            logger.warning(f"Network access ENABLED for Python execution by {user_id} for {duration_minutes} minutes")

            # If duration is specified, could set up automatic disable (simplified for now)
            if duration_minutes > 0:
                logger.info(f"Network access will remain enabled (manual disable required)")

            return {
                "success": True,
                "message": f"Network access enabled for Python execution",
                "enabled_by": user_id,
                "duration_minutes": duration_minutes,
                "current_status": self._get_current_status()
            }

        except Exception as e:
            logger.error(f"Failed to enable network access: {e}")
            return {
                "success": False,
                "message": f"Failed to enable network access: {str(e)}"
            }

    async def _disable_network(self, user_id: str) -> Dict[str, Any]:
        """Disable network access for Python execution"""
        try:
            # Update configuration in memory
            self.config.security.python_execution_network_access = False

            # Also update the YAML file for persistence
            try:
                with open("config.yaml", 'r') as f:
                    config_data = yaml.safe_load(f)

                if 'security' not in config_data:
                    config_data['security'] = {}
                config_data['security']['python_execution_network_access'] = False

                with open("config.yaml", 'w') as f:
                    yaml.safe_dump(config_data, f, default_flow_style=False)

                logger.info("Updated config.yaml with network access disabled")
            except Exception as e:
                logger.warning(f"Could not update config.yaml: {e}")

            # Log the action
            logger.info(f"Network access DISABLED for Python execution by {user_id}")

            return {
                "success": True,
                "message": "Network access disabled for Python execution",
                "disabled_by": user_id,
                "current_status": self._get_current_status()
            }

        except Exception as e:
            logger.error(f"Failed to disable network access: {e}")
            return {
                "success": False,
                "message": f"Failed to disable network access: {str(e)}"
            }

    def _get_status(self) -> Dict[str, Any]:
        """Get current network access status"""
        return {
            "success": True,
            "message": "Network access status retrieved",
            "current_status": self._get_current_status()
        }

    def _get_current_status(self) -> Dict[str, Any]:
        """Get detailed current status"""
        return {
            "network_access_enabled": self.config.security.python_execution_network_access,
            "subprocess_access_enabled": self.config.security.python_execution_subprocess_access,
            "authorized_user": self.config.security.authorized_network_override_user,
            "ethics_system_enabled": self.config.security.ethics_system_enabled
        }

    def get_schema(self) -> Dict[str, Any]:
        """Get the tool's schema for LLM consumption."""
        return {
            "name": "network_control",
            "description": "Control network access permissions for Python execution (authorized users only)",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "Action to perform",
                        "enum": ["enable_network", "disable_network", "status"]
                    },
                    "user_id": {
                        "type": "string",
                        "description": "User ID requesting the action",
                        "default": "default"
                    },
                    "duration_minutes": {
                        "type": "integer",
                        "description": "Duration for temporary access (minutes)",
                        "default": 60,
                        "minimum": 1,
                        "maximum": 1440
                    },
                    "auth_token": {
                        "type": "string",
                        "description": "Authentication token for secure operations",
                        "default": ""
                    }
                },
                "required": ["action"]
            }
        }
