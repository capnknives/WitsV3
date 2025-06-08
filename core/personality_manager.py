"""
Personality and Ethics Management System for WitsV3
Handles personality profiles, ethics overlay, and behavioral adaptation
"""

import yaml
import logging
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pydantic import BaseModel

logger = logging.getLogger(__name__)

@dataclass
class PersonalityProfile:
    """Represents a loaded personality profile"""
    name: str
    version: str
    profile_id: str
    author: str
    core_directives: List[str]
    communication: Dict[str, Any]
    cognitive_model: Dict[str, Any]
    execution_logic: Dict[str, Any]
    memory_management: Dict[str, Any]
    persona_layers: Dict[str, Any]
    trust_protocols: Dict[str, Any]
    backend_interface: Dict[str, Any]
    audit: Dict[str, Any]
    ethics_overlay: Dict[str, Any]

@dataclass
class EthicsFramework:
    """Represents a loaded ethics framework"""
    name: str
    version: str
    author: str
    core_principles: Dict[str, Any]
    decision_framework: Dict[str, Any]
    operational_guidelines: Dict[str, Any]
    safety_protocols: Dict[str, Any]
    testing_framework: Dict[str, Any]
    compliance: Dict[str, Any]
    improvement_mechanisms: Dict[str, Any]
    implementation: Dict[str, Any]

class PersonalityManager:
    """
    Manages personality profiles and ethics frameworks for WitsV3
    """

    def __init__(self, config=None):
        """Initialize the personality manager"""
        from core.config import load_config
        self.config = config or load_config()
        self.personality_profile = None
        self.ethics_framework = None
        self.current_persona = "Engineer-Strategist hybrid"
        self.ethics_overrides = {}
        self.override_expiry = None

        # Load default personality and ethics
        self._load_personality()
        self._load_ethics()

        logger.info(f"PersonalityManager initialized")

    def _load_personality(self) -> bool:
        """Load personality profile from configuration"""
        if not self.config.personality.enabled:
            logger.info("Personality system disabled in configuration")
            return False

        try:
            profile_path = self.config.personality.profile_path

            if not os.path.exists(profile_path):
                logger.error(f"Personality profile not found: {profile_path}")
                return False

            with open(profile_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            if 'wits_personality' not in data:
                logger.error("Invalid personality profile format")
                return False

            self.personality_profile = data['wits_personality']
            logger.info(f"Loaded personality profile: {self.personality_profile.get('name', 'Unknown')}")
            return True

        except Exception as e:
            logger.error(f"Error loading personality profile: {e}")
            return False

    def _load_ethics(self) -> bool:
        """Load ethics framework"""
        if not self.config.security.ethics_system_enabled:
            logger.info("Ethics system disabled in configuration")
            return False

        try:
            ethics_path = "config/ethics_overlay.yaml"

            if not os.path.exists(ethics_path):
                logger.error(f"Ethics framework not found: {ethics_path}")
                return False

            with open(ethics_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            if 'ethics_overlay' not in data:
                logger.error("Invalid ethics framework format")
                return False

            self.ethics_framework = data['ethics_overlay']
            logger.info(f"Loaded ethics framework: {self.ethics_framework.get('name', 'Unknown')}")
            return True

        except Exception as e:
            logger.error(f"Error loading ethics framework: {e}")
            return False

    def get_system_prompt(self, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate system prompt based on current personality and ethics"""
        if not self.personality_profile:
            return "You are WitsV3, an AI assistant."

        prompt_parts = []

        # Identity and role
        name = self.personality_profile.get('name', 'WitsV3')
        profile_id = self.personality_profile.get('profile_id', 'default')
        prompt_parts.append(f"You are {name} ({profile_id})")

        # Core directives
        core_directives = self.personality_profile.get('core_directives', [])
        if core_directives:
            prompt_parts.append("\nCore Directives:")
            for directive in core_directives:
                prompt_parts.append(f"- {directive}")

        # Communication style
        comm = self.personality_profile.get('communication', {})
        if comm:
            prompt_parts.append(f"\nCommunication Style:")
            prompt_parts.append(f"- Tone: {comm.get('tone', 'professional')}")
            prompt_parts.append(f"- Language Level: {comm.get('language_level', 'clear')}")
            prompt_parts.append(f"- Structure Preference: {comm.get('structure_preference', 'organized')}")

        # Ethics considerations
        if self.ethics_framework and self._is_ethics_active():
            prompt_parts.append("\nEthics Framework Active:")
            principles = self.ethics_framework.get('core_principles', {})
            for principle_name, principle_data in principles.items():
                if isinstance(principle_data, dict):
                    priority = principle_data.get('priority', 99)
                    desc = principle_data.get('description', '')
                    prompt_parts.append(f"- {principle_name.replace('_', ' ').title()} (Priority {priority}): {desc}")

        return "\n".join(prompt_parts)

    def evaluate_ethics(self, action: str, context: Optional[Dict[str, Any]] = None) -> Tuple[bool, str, List[str]]:
        """Evaluate an action against the ethics framework"""
        if not self.ethics_framework or not self._is_ethics_active():
            return True, "Ethics framework not active", []

        # Simple ethics check
        action_lower = action.lower()

                # Check for harmful keywords
        harmful_keywords = ['harm', 'hurt', 'damage', 'attack', 'destroy', 'kill']
        if any(keyword in action_lower for keyword in harmful_keywords):
            return False, "Action potentially harmful", ["Consider safer alternatives"]

        return True, "Action passes ethics evaluation", []

    def enable_ethics_override(self, user_id: str, override_type: str, duration_minutes: int = 60, auth_token: str = "") -> bool:
        """Enable ethics override for testing (authorized users only)"""
        from core.auth_manager import verify_auth

        # Check authorization with both user ID and token
        is_authorized, auth_message = verify_auth(user_id, auth_token, "ethics_override")
        if not is_authorized:
            logger.warning(f"Unauthorized ethics override attempt by user: {user_id} - {auth_message}")
            return False

        from datetime import timedelta

        self.ethics_overrides[override_type] = {
            "enabled": True,
            "authorized_by": user_id,
            "timestamp": datetime.now(),
            "duration_minutes": duration_minutes
        }

        self.override_expiry = datetime.now() + timedelta(minutes=duration_minutes)

        logger.warning(f"Ethics override '{override_type}' enabled by {user_id} for {duration_minutes} minutes")
        return True

    def _check_override_authorization(self, user_id: str) -> bool:
        """Check if user is authorized to override ethics"""
        if not self.ethics_framework:
            return False

        authorized_user = self.ethics_framework.get('testing_framework', {}).get('authorized_override_user', '')
        return user_id == authorized_user

    def _is_ethics_active(self) -> bool:
        """Check if ethics framework is currently active"""
        # Check for expired overrides
        if self.override_expiry and datetime.now() > self.override_expiry:
            self.ethics_overrides.clear()
            self.override_expiry = None
            logger.info("Ethics overrides expired and have been cleared")

        # Ethics is active unless completely overridden
        return not self.ethics_overrides.get('complete_disable', {}).get('enabled', False)

# Global instance for easy access
_personality_manager = None

def get_personality_manager():
    """Get global personality manager instance"""
    global _personality_manager
    if _personality_manager is None:
        _personality_manager = PersonalityManager()
    return _personality_manager

def reload_personality_manager():
    """Reload the global personality manager"""
    global _personality_manager
    _personality_manager = PersonalityManager()
    return _personality_manager
