import pytest
import yaml
import os
from unittest.mock import patch, mock_open

from core.config import WitsV3Config, OllamaSettings, LLMInterfaceSettings, AgentSettings, MemoryManagerSettings, ToolSystemSettings, CLISettings, NeuralWebSettings

# Helper function to get project root
def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

def test_load_default_config():
    """Test that WitsV3Config loads with default values when no file is provided or found."""
    with patch('builtins.open', mock_open(read_data="")) as mock_file:
        mock_file.side_effect = FileNotFoundError  # Simulate file not found
        config = WitsV3Config.from_yaml("non_existent_config.yaml")

    assert config.project_name == "WitsV3"
    assert config.version == "3.0.0"
    assert config.logging_level == "INFO"
    assert config.debug_mode is False
    assert config.auto_restart_on_file_change is True

    assert isinstance(config.llm_interface, LLMInterfaceSettings)
    assert config.llm_interface.default_provider == "ollama"
    assert config.llm_interface.timeout_seconds == 120
    assert config.llm_interface.streaming_enabled is True

    assert isinstance(config.ollama_settings, OllamaSettings)
    assert config.ollama_settings.url == "http://localhost:11434"
    assert config.ollama_settings.default_model == "llama3"
    assert config.ollama_settings.control_center_model == "llama3"
    assert config.ollama_settings.orchestrator_model == "llama3"
    assert config.ollama_settings.embedding_model == "llama3" # Added assertion for embedding_model
    assert config.ollama_settings.request_timeout == 120

    assert isinstance(config.agents, AgentSettings)
    assert config.agents.default_temperature == 0.7
    assert config.agents.max_iterations == 15

    assert isinstance(config.memory_manager, MemoryManagerSettings)
    assert config.memory_manager.backend == "basic"
    assert config.memory_manager.memory_file_path == "data/wits_memory.json"
    assert config.memory_manager.faiss_index_path == "data/wits_faiss_index.bin"
    assert config.memory_manager.neural_web_path == "data/neural_web.json"
    assert config.memory_manager.vector_dim == 4096
    assert config.memory_manager.max_results_per_search == 5
    assert config.memory_manager.pruning_interval_seconds == 3600
    assert config.memory_manager.max_memory_segments == 10000
    assert isinstance(config.memory_manager.neural_web_settings, NeuralWebSettings) # Added assertion for neural_web_settings

    assert isinstance(config.tool_system, ToolSystemSettings)
    assert config.tool_system.enable_mcp_tools is True
    assert config.tool_system.mcp_tool_definitions_path == "data/mcp_tools.json"

    assert isinstance(config.cli, CLISettings)
    assert config.cli.show_thoughts is True
    assert config.cli.show_tool_calls is True

def test_load_from_existing_config_yaml():
    """Test loading configuration from the actual config.yaml file if it exists."""
    project_root = get_project_root()
    config_path = os.path.join(project_root, "config.yaml")

    if not os.path.exists(config_path):
        pytest.skip(f"config.yaml not found at {config_path}, skipping test.")

    config = WitsV3Config.from_yaml(config_path)

    # Check a few key values that might differ from defaults
    # We don't assert all values, as they can be user-defined
    assert isinstance(config, WitsV3Config)
    assert config.project_name is not None # Should have a value

    # Check that nested models are loaded
    assert isinstance(config.ollama_settings, OllamaSettings)
    assert config.ollama_settings.url is not None

    assert isinstance(config.memory_manager, MemoryManagerSettings)
    assert config.memory_manager.backend is not None
    assert isinstance(config.memory_manager.neural_web_settings, NeuralWebSettings)


def test_load_from_empty_yaml_file():
    """Test that loading from an empty YAML file results in default values."""
    with patch('builtins.open', mock_open(read_data="")) as mocked_file, \
         patch('yaml.safe_load', return_value=None) as mocked_safe_load:
        config = WitsV3Config.from_yaml("empty_config.yaml")

    mocked_file.assert_called_once_with("empty_config.yaml", 'r')
    mocked_safe_load.assert_called_once()

    # Assertions are same as test_load_default_config
    assert config.project_name == "WitsV3"
    assert config.ollama_settings.default_model == "llama3"
    assert config.agents.default_temperature == 0.7


def test_load_from_invalid_yaml_file():
    """Test that loading from an invalid YAML file results in default values."""
    invalid_yaml_content = ": invalid_yaml" # This is not valid YAML
    with patch('builtins.open', mock_open(read_data=invalid_yaml_content)) as mocked_file, \
         patch('yaml.safe_load', side_effect=yaml.YAMLError("Mocked YAML Error")) as mocked_safe_load:
        config = WitsV3Config.from_yaml("invalid_config.yaml")

    mocked_file.assert_called_once_with("invalid_config.yaml", 'r')
    mocked_safe_load.assert_called_once()

    # Assertions are same as test_load_default_config
    assert config.project_name == "WitsV3"
    assert config.ollama_settings.default_model == "llama3"
    assert config.agents.default_temperature == 0.7


def test_agent_settings_validation():
    """Test Pydantic validation for AgentSettings."""
    # Valid temperature
    agent_settings = AgentSettings(default_temperature=1.0, max_iterations=10)
    assert agent_settings.default_temperature == 1.0
    assert agent_settings.max_iterations == 10

    # Invalid temperature (too low)
    with pytest.raises(ValueError): # Pydantic raises ValueError for validation errors
        AgentSettings(default_temperature=-0.5, max_iterations=10)

    # Invalid temperature (too high)
    with pytest.raises(ValueError):
        AgentSettings(default_temperature=2.5, max_iterations=10)

    # Invalid max_iterations (zero)
    with pytest.raises(ValueError):
        AgentSettings(default_temperature=0.7, max_iterations=0)
    
    # Invalid max_iterations (negative)
    with pytest.raises(ValueError):
        AgentSettings(default_temperature=0.7, max_iterations=-5)


def test_ollama_settings_defaults():
    """Test OllamaSettings default values."""
    settings = OllamaSettings()
    assert settings.url == "http://localhost:11434"
    assert settings.default_model == "llama3"
    assert settings.control_center_model == "llama3"
    assert settings.orchestrator_model == "llama3"
    assert settings.embedding_model == "llama3"
    assert settings.request_timeout == 120


def test_neural_web_settings_validation():
    """Test Pydantic validation for NeuralWebSettings."""
    # Valid settings
    settings = NeuralWebSettings(activation_threshold=0.5, decay_rate=0.05, max_concept_connections=30, connection_strength_threshold=0.3)
    assert settings.activation_threshold == 0.5
    assert settings.decay_rate == 0.05
    assert settings.max_concept_connections == 30
    assert settings.connection_strength_threshold == 0.3

    # Invalid activation_threshold
    with pytest.raises(ValueError):
        NeuralWebSettings(activation_threshold=1.5)
    with pytest.raises(ValueError):
        NeuralWebSettings(activation_threshold=-0.1)
    
    # Invalid decay_rate
    with pytest.raises(ValueError):
        NeuralWebSettings(decay_rate=1.1)
    with pytest.raises(ValueError):
        NeuralWebSettings(decay_rate=-0.2)

    # Invalid max_concept_connections
    with pytest.raises(ValueError):
        NeuralWebSettings(max_concept_connections=0)
    with pytest.raises(ValueError):
        NeuralWebSettings(max_concept_connections=-10)
        
    # Invalid connection_strength_threshold
    with pytest.raises(ValueError):
        NeuralWebSettings(connection_strength_threshold=1.2)
    with pytest.raises(ValueError):
        NeuralWebSettings(connection_strength_threshold=-0.3)

def test_witsv3config_assignment_validation():
    """Test that assignments to WitsV3Config fields are validated."""
    config = WitsV3Config()
    with pytest.raises(ValueError): # Pydantic raises ValueError with validate_assignment = True
        config.agents.default_temperature = -0.5

    with pytest.raises(ValueError):
        config.agents.max_iterations = 0
        
    # Test a valid assignment
    config.agents.default_temperature = 1.5
    assert config.agents.default_temperature == 1.5


# Example of how to run tests if this file is executed directly (optional)
if __name__ == "__main__":
    pytest.main([__file__]) 