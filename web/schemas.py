"""Pydantic request/response models for the WitsV3 web UI."""

from typing import Any

from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ExportRequest(BaseModel):
    session_id: str | None = None
    file_path: str | None = None


class SettingsUpdate(BaseModel):
    """Runtime-adjustable settings from the /settings page. All optional."""

    history_window: int | None = None
    default_temperature: float | None = None
    max_iterations: int | None = None
    default_model: str | None = None
    orchestrator_model: str | None = None
    routing_enabled: bool | None = None
    routing_trivial_model: str | None = None
    routing_code_model: str | None = None
    routing_complex_model: str | None = None
    routing_trivial_max_chars: int | None = None
    escalation_enabled: bool | None = None
    escalation_model: str | None = None
    escalation_max_tokens: int | None = None


class MemoryPruneRequest(BaseModel):
    """Request to delete memory segments by filter (web UI safeguard)."""

    filter_dict: dict[str, Any]
    confirm: str


class MCPServerAdd(BaseModel):
    name: str
    command: str
    working_directory: str | None = None
    args: list[str] | None = None


class MCPRegistryInstall(BaseModel):
    """Install a server discovered via MCP registry search."""

    name: str
    command: list[str]
    working_directory: str | None = None
    env: dict[str, str] | None = None
    connect: bool = False


class MCPToolInvoke(BaseModel):
    """Invoke a connected MCP tool from the web UI playground."""

    arguments: dict[str, Any] = {}


class EscalationDecision(BaseModel):
    session_id: str | None = None


class PersonalityAnswers(BaseModel):
    """Questionnaire answers from the /personality page."""

    identity_label: str | None = None
    default_role: str | None = None
    tone: str | None = None
    language_level: str | None = None
    verbosity: str | None = None
    structure_preference: str | None = None
    humor: str | None = None
    default_persona: str | None = None
    core_directives: list[str] | None = None
