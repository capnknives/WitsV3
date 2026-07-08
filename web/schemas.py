"""Pydantic request/response models for the WitsV3 web UI."""

from typing import Any

from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class GuestRegisterRequest(BaseModel):
    invite_code: str
    display_name: str
    device_id: str
    age_band: str | None = None  # ignored — owner assigns tier; new guests get default_guest_age_band


class GuestSetAgeBandRequest(BaseModel):
    age_band: str
    guest_id: str | None = None
    display_name: str | None = None


class GuestRevokeRequest(BaseModel):
    guest_id: str


class GuestMergeRequest(BaseModel):
    target_guest_id: str
    source_guest_id: str


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


class OwnerControlRequest(BaseModel):
    """Chat/API owner process controls (shutdown or restart)."""

    confirm: str
    delay_seconds: float | None = 1.0


class DocumentDeleteRequest(BaseModel):
    """Remove a single ingested document file and its memory chunks."""

    name: str


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
