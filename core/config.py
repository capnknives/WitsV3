"""
Configuration management for WitsV3 system
"""

import yaml
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class OllamaSettings(BaseModel):
    url: str = Field(default="http://localhost:11434")
    default_model: str = Field(default="llama3")
    control_center_model: str = Field(default="llama3")
    orchestrator_model: str = Field(default="llama3")
    embedding_model: str = Field(default="llama3")
    request_timeout: int = Field(default=120)
    retry_attempts: int = Field(default=3, description="Number of retry attempts for failed requests")
    retry_delay: float = Field(default=1.0, description="Delay between retry attempts in seconds")
    exponential_backoff: bool = Field(default=True, description="Use exponential backoff for retries")

    # Model fallback configuration
    fallback_models: List[str] = Field(default=["llama3", "llama3:8b", "codellama:7b"], description="Fallback models in order of preference")
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
    vector_dim: int = Field(default=4096) # llama3 typically 4096
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
    mcp_tool_definitions_path: str = Field(default="data/mcp_tools.json")
    # langchain_bridge_enabled: bool = Field(default=False) # Placeholder

class CLISettings(BaseModel):
    show_thoughts: bool = Field(default=True)
    show_tool_calls: bool = Field(default=True)

class SupabaseSettings(BaseModel):
    url: str = Field(default="")
    key: str = Field(default="")
    enable_realtime: bool = Field(default=True)

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
    supabase: SupabaseSettings = Field(default_factory=SupabaseSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    personality: PersonalitySettings = Field(default_factory=PersonalitySettings)

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

# Convenience function for loading config
def load_config(config_path: str = "config.yaml") -> WitsV3Config:
    """Load configuration from YAML file."""
    return WitsV3Config.from_yaml(config_path)

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
