"""
Configuration management for WitsV3 system
"""

import os

import yaml
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env from the working directory, if present
except ImportError:
    pass

class OllamaSettings(BaseModel):
    url: str = Field(default="http://localhost:11434")
    default_model: str = Field(default="qwen3:8b")
    control_center_model: str = Field(default="qwen3:8b")
    orchestrator_model: str = Field(default="qwen3:8b")
    embedding_model: str = Field(default="nomic-embed-text")
    request_timeout: int = Field(default=120)
    retry_attempts: int = Field(default=3, description="Number of retry attempts for failed requests")
    retry_delay: float = Field(default=1.0, description="Delay between retry attempts in seconds")
    exponential_backoff: bool = Field(default=True, description="Use exponential backoff for retries")

    # Model fallback configuration
    fallback_models: List[str] = Field(default=["qwen3:8b", "llama3.2:3b", "qwen2.5-coder:7b"], description="Fallback models in order of preference")
    enable_model_fallback: bool = Field(default=True, description="Enable automatic fallback to alternative models")
    model_failure_threshold: int = Field(default=3, description="Number of consecutive failures before switching models")
    model_timeout: int = Field(default=300, description="Timeout for individual model operations in seconds")

    # Model health monitoring
    health_check_interval: int = Field(default=60, description="Interval for model health checks in seconds")
    enable_health_monitoring: bool = Field(default=True, description="Enable continuous model health monitoring")
    quarantine_duration: int = Field(default=300, description="Time to quarantine failed models in seconds")

class LLMInterfaceSettings(BaseModel):
    default_provider: str = Field(default="ollama")
    timeout_seconds: int = Field(default=120)
    streaming_enabled: bool = Field(default=True)

class AgentSettings(BaseModel):
    default_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_iterations: int = Field(default=15, gt=0)
    history_window: int = Field(default=20, ge=2, le=100, description="How many recent conversation messages are included in the prompt")

    class Config:
        validate_assignment = True

class NeuralWebSettings(BaseModel):
    activation_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    decay_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    auto_connect: bool = Field(default=True)
    reasoning_patterns: List[str] = Field(default=["modus_ponens", "analogy", "chain", "contradiction"])
    max_concept_connections: int = Field(default=50, gt=0)
    connection_strength_threshold: float = Field(default=0.2, ge=0.0, le=1.0)
    # Cross-domain learning settings
    enable_cross_domain_learning: bool = Field(default=True, description="Enable cross-domain learning capabilities")
    cross_domain_similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum similarity threshold for cross-domain connections")
    max_cross_domain_connections: int = Field(default=10, gt=0, description="Maximum number of connections across domains per concept")
    domain_classification_confidence: float = Field(default=0.6, ge=0.0, le=1.0, description="Confidence threshold for domain classification")

class MemoryManagerSettings(BaseModel):
    backend: str = Field(default="basic") # basic, faiss_cpu, faiss_gpu, neural
    memory_file_path: str = Field(default="data/wits_memory.json")
    faiss_index_path: str = Field(default="data/wits_faiss_index.bin")
    neural_web_path: str = Field(default="data/neural_web.json")
    vector_dim: int = Field(default=768) # nomic-embed-text embedding dimension
    max_results_per_search: int = Field(default=5)
    pruning_interval_seconds: int = Field(default=3600)
    max_memory_segments: int = Field(default=10000)

    # Enhanced pruning settings
    enable_auto_pruning: bool = Field(default=True, description="Enable automatic memory pruning")
    max_memory_size_mb: int = Field(default=50, gt=0, description="Maximum memory size in MB before pruning")
    pruning_threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Size threshold (as fraction) to trigger pruning")
    pruning_strategy: str = Field(default="hybrid", description="Pruning strategy: oldest_first, least_relevant, or hybrid")

    neural_web_settings: NeuralWebSettings = Field(default_factory=NeuralWebSettings)

class SecuritySettings(BaseModel):
    python_execution_network_access: bool = Field(default=False, description="Allow network access in Python execution tool")
    python_execution_subprocess_access: bool = Field(default=False, description="Allow subprocess execution in Python execution tool")
    authorized_network_override_user: str = Field(default="richard_elliot", description="Only this user can override network restrictions")
    ethics_system_enabled: bool = Field(default=True, description="Enable ethics overlay system")
    ethics_override_authorized_user: str = Field(default="richard_elliot", description="Only this user can disable ethics during testing")
    auth_token_hash: str = Field(default="", description="SHA256 hash of authorization token for secure operations")
    require_auth_for_network_control: bool = Field(default=True, description="Require authentication token for network control")
    require_auth_for_ethics_override: bool = Field(default=True, description="Require authentication token for ethics override")

class ToolSystemSettings(BaseModel):
    enable_mcp_tools: bool = Field(default=True)
    mcp_connect_on_startup: bool = Field(default=False, description="Connect to external MCP servers during startup (slows boot when servers are unavailable)")
    mcp_tool_definitions_path: str = Field(default="data/mcp_tools.json")
    mcp_registry_url: str = Field(default="https://registry.modelcontextprotocol.io", description="Official MCP registry used by the discover/search feature")
    # langchain_bridge_enabled: bool = Field(default=False) # Placeholder

class CLISettings(BaseModel):
    show_thoughts: bool = Field(default=True)
    show_tool_calls: bool = Field(default=True)

class WebUISettings(BaseModel):
    enabled: bool = Field(default=True, description="Enable the web UI server")
    host: str = Field(default="0.0.0.0", description="Bind address (0.0.0.0 = reachable on the LAN, e.g. from a phone)")
    port: int = Field(default=8000, gt=0, lt=65536, description="Web UI port")
    require_auth: bool = Field(default=True, description="Require the WITSV3_WEB_TOKEN bearer token for API access")

class DocumentRAGSettings(BaseModel):
    enabled: bool = Field(default=True, description="Enable the document RAG system")
    documents_path: str = Field(default="documents", description="Folder watched for documents to ingest")
    chunk_size: int = Field(default=1200, gt=0, description="Target chunk size in characters")
    chunk_overlap: int = Field(default=150, ge=0, description="Characters of overlap between consecutive chunks")
    auto_ingest_on_startup: bool = Field(default=True, description="Scan and ingest documents during system startup")
    file_extensions: List[str] = Field(
        default=[".txt", ".md", ".markdown", ".py", ".json", ".csv", ".html", ".log", ".pdf"],
        description="File extensions considered for ingestion (.pdf requires pypdf)"
    )

class SupabaseSettings(BaseModel):
    url: str = Field(default="")
    key: str = Field(default="")
    enable_realtime: bool = Field(default=True)

class EscalationSettings(BaseModel):
    """Ask-Claude escalation: WITS can queue a question for the Claude API,
    but every request must be approved by the user in the web UI first.
    The API key comes from the ANTHROPIC_API_KEY environment variable (.env)."""
    enabled: bool = Field(default=True, description="Allow WITS to queue escalation requests to Claude")
    model: str = Field(default="claude-opus-4-8", description="Claude model used for approved escalations")
    max_tokens: int = Field(default=2048, ge=256, le=16000, description="Hard cap on Claude's response length (caps cost)")

class ModelRoutingSettings(BaseModel):
    """Smart model routing: pick the Ollama model per request based on what the
    user actually asked, instead of sending everything to the default model.
    Trivial chat goes to a small fast model, code work goes to the coder model,
    everything else stays on the default. The router only ever sees the raw
    user message or goal, never full prompt templates."""
    enabled: bool = Field(default=True, description="Enable per-request model routing")
    trivial_model: str = Field(default="llama3.2:3b", description="Model for short casual chat")
    code_model: str = Field(default="qwen2.5-coder:7b", description="Model for code-related requests")
    complex_model: str = Field(default="qwen3:8b", description="Model for everything else")
    trivial_max_chars: int = Field(default=140, gt=0, description="Messages longer than this are never routed to the trivial model")

class WebSearchSettings(BaseModel):
    """Web search tool configuration.

    The tool tries providers in a fallback chain so it returns *real* web
    results even with no API key configured. Quality improves a lot if you
    add a key (Tavily or Brave) in the gitignored .env:
      - TAVILY_API_KEY        (https://tavily.com — 1000 free searches/mo, LLM-optimized)
      - BRAVE_SEARCH_API_KEY  (https://brave.com/search/api — 2000 free/mo)
    Keys are NEVER stored in config.yaml; they are injected from the
    environment at load time (see _apply_env_overrides)."""
    provider: str = Field(
        default="auto",
        description="Provider to use: auto (try all, best-available first), tavily, brave, or duckduckgo",
    )
    max_results: int = Field(default=5, gt=0, le=25, description="Default number of results to return")
    tavily_search_depth: str = Field(
        default="advanced",
        description="Tavily depth: 'advanced' (more accurate answers, 2 credits) or 'basic' (1 credit). "
                    "'basic' can return wrong answers for date/fact queries — 'advanced' is worth it.",
    )
    timeout_seconds: float = Field(default=12.0, gt=0, description="Per-request HTTP timeout")
    region: str = Field(default="wt-wt", description="DuckDuckGo region code (wt-wt = no region)")
    safesearch: str = Field(default="moderate", description="Safe search level: off, moderate, strict")
    # Secrets — populated from the environment, not config.yaml.
    tavily_api_key: str = Field(default="", description="Set via TAVILY_API_KEY env var")
    brave_api_key: str = Field(default="", description="Set via BRAVE_SEARCH_API_KEY env var")


class PersonalitySettings(BaseModel):
    enabled: bool = Field(default=True, description="Enable personality system")
    profile_path: str = Field(default="config/wits_personality.yaml", description="Path to personality profile")
    profile_id: str = Field(default="richard_elliot_wits", description="Active personality profile ID")
    allow_runtime_switching: bool = Field(default=False, description="Allow personality switching at runtime")

class WitsV3Config(BaseModel):
    project_name: str = Field(default="WitsV3")
    version: str = Field(default="3.0.0")
    logging_level: str = Field(default="INFO")
    debug_mode: bool = Field(default=False)
    auto_restart_on_file_change: bool = Field(default=True, description="Automatically restart the system when Python files are changed")

    llm_interface: LLMInterfaceSettings = LLMInterfaceSettings()
    ollama_settings: OllamaSettings = OllamaSettings()
    agents: AgentSettings = AgentSettings()
    memory_manager: MemoryManagerSettings = Field(default_factory=MemoryManagerSettings)
    tool_system: ToolSystemSettings = ToolSystemSettings()
    cli: CLISettings = CLISettings()
    document_rag: DocumentRAGSettings = Field(default_factory=DocumentRAGSettings)
    web_ui: WebUISettings = Field(default_factory=WebUISettings)
    supabase: SupabaseSettings = Field(default_factory=SupabaseSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    personality: PersonalitySettings = Field(default_factory=PersonalitySettings)
    escalation: EscalationSettings = Field(default_factory=EscalationSettings)
    model_routing: ModelRoutingSettings = Field(default_factory=ModelRoutingSettings)
    web_search: WebSearchSettings = Field(default_factory=WebSearchSettings)

    class Config:
        validate_assignment = True

    @classmethod
    def from_yaml(cls, config_path: str = "config.yaml") -> "WitsV3Config":
        """Loads the YAML configuration file and returns a WitsV3Config object."""
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            if not config_data:
                print(f"Warning: Configuration file '{config_path}' is empty. Using default values.")
                return cls()
            return cls(**config_data)
        except FileNotFoundError:
            print(f"Error: Configuration file '{config_path}' not found. Using default values.")
            return cls()
        except yaml.YAMLError as e:
            print(f"Error parsing YAML configuration file '{config_path}': {e}. Using default values.")
            return cls()
        except Exception as e:
            print(f"An unexpected error occurred while loading config '{config_path}': {e}. Using default values.")
            return cls()

def _apply_env_overrides(config: "WitsV3Config") -> "WitsV3Config":
    """Apply secret values from environment variables (or .env).

    Secrets are intentionally NOT stored in config.yaml (which is committed
    to git). Set them in a local .env file or the environment instead:
      - WITSV3_SUPABASE_URL:      Supabase project URL
      - WITSV3_SUPABASE_KEY:      Supabase API key
      - WITSV3_AUTH_TOKEN_HASH:   SHA256 hash of the admin auth token
      - TAVILY_API_KEY:           optional web-search provider (better results)
      - BRAVE_SEARCH_API_KEY:     optional web-search provider (better results)
    """
    supabase_url = os.getenv("WITSV3_SUPABASE_URL")
    if supabase_url:
        config.supabase.url = supabase_url

    supabase_key = os.getenv("WITSV3_SUPABASE_KEY")
    if supabase_key:
        config.supabase.key = supabase_key

    auth_token_hash = os.getenv("WITSV3_AUTH_TOKEN_HASH")
    if auth_token_hash:
        config.security.auth_token_hash = auth_token_hash

    # Web search API keys (optional — the tool falls back to keyless DuckDuckGo)
    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key:
        config.web_search.tavily_api_key = tavily_key

    brave_key = os.getenv("BRAVE_SEARCH_API_KEY")
    if brave_key:
        config.web_search.brave_api_key = brave_key

    return config


# Settings changed at runtime (web UI settings page) are stored in a separate
# gitignored overrides file, deep-merged over config.yaml at load. This keeps
# the hand-written, commented config.yaml untouched by programmatic saves.
LOCAL_OVERRIDES_PATH = "config.local.yaml"


def _deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _apply_local_overrides(config: "WitsV3Config", config_path: str) -> "WitsV3Config":
    overrides_path = os.path.join(os.path.dirname(config_path) or ".", LOCAL_OVERRIDES_PATH)
    if not os.path.exists(overrides_path):
        return config
    try:
        with open(overrides_path, 'r', encoding='utf-8') as f:
            overrides = yaml.safe_load(f) or {}
        if isinstance(overrides, dict) and overrides:
            merged = _deep_merge(config.model_dump() if hasattr(config, "model_dump") else config.dict(), overrides)
            return WitsV3Config(**merged)
    except Exception as e:
        print(f"Warning: ignoring bad local config overrides ({overrides_path}): {e}")
    return config


def save_local_overrides(new_overrides: Dict[str, Any], config_path: str = "config.yaml") -> str:
    """Merge new_overrides into config.local.yaml (creating it if needed)."""
    overrides_path = os.path.join(os.path.dirname(config_path) or ".", LOCAL_OVERRIDES_PATH)
    existing: Dict[str, Any] = {}
    if os.path.exists(overrides_path):
        try:
            with open(overrides_path, 'r', encoding='utf-8') as f:
                existing = yaml.safe_load(f) or {}
        except Exception:
            existing = {}
    merged = _deep_merge(existing, new_overrides)
    header = (
        "# Runtime settings saved from the WITS web UI (/settings).\n"
        "# Deep-merged over config.yaml at load - delete this file to revert.\n"
    )
    with open(overrides_path, 'w', encoding='utf-8') as f:
        f.write(header + yaml.safe_dump(merged, sort_keys=False, allow_unicode=True))
    return overrides_path


# Convenience function for loading config
def load_config(config_path: str = "config.yaml") -> WitsV3Config:
    """Load configuration from YAML file, with local + env-var overrides."""
    config = WitsV3Config.from_yaml(config_path)
    config = _apply_local_overrides(config, config_path)
    return _apply_env_overrides(config)

# For backwards compatibility
AppConfig = WitsV3Config
LLMConfig = LLMInterfaceSettings
MemoryConfig = MemoryManagerSettings
AgentConfig = AgentSettings
ToolConfig = ToolSystemSettings

# Test function
def test_config():
    """Test configuration loading and validation"""
    import os

    # Test default config
    default_config = WitsV3Config()
    print(f"Default config: {default_config.project_name}")

    # Test loading from file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    config_file_path = os.path.join(project_root, "config.yaml")

    if os.path.exists(config_file_path):
        config = WitsV3Config.from_yaml(config_file_path)
        print(f"Loaded config: {config.project_name}")
        print(f"LLM Provider: {config.llm_interface.default_provider}")
        print(f"Ollama URL: {config.ollama_settings.url}")
        print(f"Default Model: {config.ollama_settings.default_model}")
    else:
        print(f"Config file not found at {config_file_path}")

    # Test validation
    try:
        config.agents.default_temperature = -0.5
    except Exception as e:
        print(f"Validation works: {e}")

if __name__ == "__main__":
    test_config()
