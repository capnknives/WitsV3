"""Pydantic request/response models for the WitsV3 web UI."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class SettingsUpdate(BaseModel):
    """Runtime-adjustable settings from the /settings page. All optional."""

    history_window: Optional[int] = None
    default_temperature: Optional[float] = None
    max_iterations: Optional[int] = None
    default_model: Optional[str] = None
    orchestrator_model: Optional[str] = None
    routing_enabled: Optional[bool] = None
    routing_trivial_model: Optional[str] = None
    routing_code_model: Optional[str] = None
    routing_complex_model: Optional[str] = None
    routing_trivial_max_chars: Optional[int] = None
    escalation_enabled: Optional[bool] = None
    escalation_model: Optional[str] = None
    escalation_max_tokens: Optional[int] = None


class MCPServerAdd(BaseModel):
    name: str
    command: str
    working_directory: Optional[str] = None
    args: Optional[list[str]] = None


class MCPRegistryInstall(BaseModel):
    """Install a server discovered via MCP registry search."""

    name: str
    command: list[str]
    working_directory: Optional[str] = None
    env: Optional[Dict[str, str]] = None
    connect: bool = False


class MCPToolInvoke(BaseModel):
    """Invoke a connected MCP tool from the web UI playground."""

    arguments: Dict[str, Any] = {}


class EscalationDecision(BaseModel):
    session_id: Optional[str] = None


class PersonalityAnswers(BaseModel):
    """Questionnaire answers from the /personality page."""

    identity_label: Optional[str] = None
    default_role: Optional[str] = None
    tone: Optional[str] = None
    language_level: Optional[str] = None
    verbosity: Optional[str] = None
    structure_preference: Optional[str] = None
    humor: Optional[str] = None
    default_persona: Optional[str] = None
    core_directives: Optional[list[str]] = None
