"""
Authentication Manager for WitsV3
Provides secure token-based authentication for sensitive operations
"""

import hashlib
import secrets
import logging
import os
from typing import Optional, Tuple
from core.config import load_config

logger = logging.getLogger(__name__)

class AuthManager:
    """Manages authentication for WitsV3 sensitive operations"""

    def __init__(self):
        """Initialize the authentication manager"""
        self.config = load_config()

    def generate_token(self, length: int = 32) -> str:
        """
        Generate a cryptographically secure random token

        Args:
            length: Length of the token in bytes

        Returns:
            Hex-encoded secure random token
        """
        return secrets.token_hex(length)

    def hash_token(self, token: str) -> str:
        """
        Create SHA256 hash of a token

        Args:
            token: The token to hash

        Returns:
            SHA256 hash as hex string
        """
        return hashlib.sha256(token.encode('utf-8')).hexdigest()

    def verify_token(self, provided_token: str, user_id: Optional[str] = None) -> bool:
        """
        Verify a provided token against the stored hash

        Args:
            provided_token: The token provided by the user
            user_id: Optional user ID for additional validation

        Returns:
            True if token is valid, False otherwise
        """
        if not provided_token:
            logger.warning("No authentication token provided")
            return False

        # Get stored token hash from config
        stored_hash = self.config.security.auth_token_hash

        if not stored_hash:
            logger.error("No authentication token hash configured in system")
            return False

        # Hash the provided token and compare
        provided_hash = self.hash_token(provided_token)

        if provided_hash == stored_hash:
            logger.info(f"Authentication successful for user: {user_id or 'unknown'}")
            return True
        else:
            logger.warning(f"Authentication failed for user: {user_id or 'unknown'}")
            return False

    def verify_user_and_token(self, user_id: str, auth_token: str, operation_type: str = "general") -> Tuple[bool, str]:
        """
        Verify both user ID and authentication token

        Args:
            user_id: The user ID claiming access
            auth_token: The authentication token provided
            operation_type: Type of operation being attempted

        Returns:
            Tuple of (is_authorized, reason_message)
        """
        # Check if authentication is required for this operation
        if operation_type == "network_control":
            if not self.config.security.require_auth_for_network_control:
                return True, "Authentication not required for network control"
            authorized_user = self.config.security.authorized_network_override_user
        elif operation_type == "ethics_override":
            if not self.config.security.require_auth_for_ethics_override:
                return True, "Authentication not required for ethics override"
            authorized_user = self.config.security.ethics_override_authorized_user
        else:
            return False, "Unknown operation type"

        # Check user ID
        if user_id != authorized_user:
            return False, f"Unauthorized user: only '{authorized_user}' can perform {operation_type}"

        # Check authentication token
        if not self.verify_token(auth_token, user_id):
            return False, "Invalid authentication token"

        return True, f"Authentication successful for {operation_type}"

    def is_token_configured(self) -> bool:
        """
        Check if an authentication token is configured

        Returns:
            True if token is configured, False otherwise
        """
        return bool(self.config.security.auth_token_hash)

    def setup_initial_token(self, token: str) -> bool:
        """
        Set up the initial authentication token (for first-time setup)

        Args:
            token: The token to configure

        Returns:
            True if setup successful, False otherwise
        """
        try:
            import yaml

            # Hash the token
            token_hash = self.hash_token(token)

            # Update config file
            with open("config.yaml", 'r') as f:
                config_data = yaml.safe_load(f)

            if 'security' not in config_data:
                config_data['security'] = {}

            config_data['security']['auth_token_hash'] = token_hash

            with open("config.yaml", 'w') as f:
                yaml.safe_dump(config_data, f, default_flow_style=False)

            logger.info("Authentication token configured successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to setup authentication token: {e}")
            return False

# Global authentication manager instance
_auth_manager: Optional[AuthManager] = None

def get_auth_manager() -> AuthManager:
    """Get global authentication manager instance"""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthManager()
    return _auth_manager

def generate_new_token() -> str:
    """Generate a new secure authentication token"""
    return get_auth_manager().generate_token()

def verify_auth(user_id: str, auth_token: str, operation_type: str = "general") -> Tuple[bool, str]:
    """
    Convenience function for authentication verification

    Args:
        user_id: User ID claiming access
        auth_token: Authentication token provided
        operation_type: Type of operation being attempted

    Returns:
        Tuple of (is_authorized, reason_message)
    """
    return get_auth_manager().verify_user_and_token(user_id, auth_token, operation_type)
