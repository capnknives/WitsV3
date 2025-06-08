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

class MemoryManagerSettings(BaseModel):
    backend: str = Field(default="basic") # basic, faiss_cpu, faiss_gpu, neural
    memory_file_path: str = Field(default="data/wits_memory.json")
    faiss_index_path: str = Field(default="data/wits_faiss_index.bin")
    neural_web_path: str = Field(default="data/neural_web.json")
    vector_dim: int = Field(default=4096) # llama3 typically 4096
    max_results_per_search: int = Field(default=5)
    pruning_interval_seconds: int = Field(default=3600)
    max_memory_segments: int = Field(default=10000)
    neural_web_settings: NeuralWebSettings = Field(default_factory=NeuralWebSettings)

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
