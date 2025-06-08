import pytest
import httpx
import json
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

from core.config import WitsV3Config, OllamaSettings, LLMInterfaceSettings, AgentSettings
from core.llm_interface import OllamaInterface, BaseLLMInterface, get_llm_interface

@pytest.fixture
def mock_config() -> WitsV3Config:
    """Provides a mock WitsV3Config for testing."""
    config = WitsV3Config()
    config.ollama_settings = OllamaSettings(
        url="http://test-ollama:11434",
        default_model="test-model",
        embedding_model="test-embedding-model",
        request_timeout=5
    )
    config.agents = AgentSettings(default_temperature=0.5)
    config.llm_interface = LLMInterfaceSettings(default_provider="ollama")
    return config

@pytest.fixture
def mock_async_client() -> MagicMock:
    """Mocks httpx.AsyncClient."""
    mock_client = MagicMock(spec=httpx.AsyncClient)
    # AsyncMock for methods that are awaited
    mock_client.post = AsyncMock()
    mock_client.stream = AsyncMock() 
    # __aenter__ and __aexit__ for async context manager
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    return mock_client

@pytest.mark.asyncio
async def test_ollama_interface_initialization(mock_config: WitsV3Config):
    """Test OllamaInterface initializes correctly."""
    with patch('httpx.AsyncClient') as MockedAsyncClient:
        instance = MockedAsyncClient.return_value
        interface = OllamaInterface(mock_config)
        assert interface.ollama_settings == mock_config.ollama_settings
        assert interface.http_client == instance
        MockedAsyncClient.assert_called_once_with(timeout=mock_config.ollama_settings.request_timeout)

@pytest.mark.asyncio
async def test_prepare_payload_default_params(mock_config: WitsV3Config):
    """Test _prepare_payload with default parameters."""
    interface = OllamaInterface(mock_config)
    prompt = "Test prompt"
    payload = await interface._prepare_payload(prompt=prompt, stream=False)
    
    assert payload["model"] == "test-model"
    assert payload["prompt"] == prompt
    assert payload["stream"] is False
    assert payload["options"]["temperature"] == 0.5
    assert "num_predict" not in payload["options"]
    assert "stop" not in payload["options"]

@pytest.mark.asyncio
async def test_prepare_payload_override_params(mock_config: WitsV3Config):
    """Test _prepare_payload with overridden parameters."""
    interface = OllamaInterface(mock_config)
    prompt = "Test prompt"
    model = "override-model"
    temperature = 0.9
    max_tokens = 100
    stop_sequences = ["stop1", "stop2"]
    
    payload = await interface._prepare_payload(
        prompt=prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        stop_sequences=stop_sequences,
        stream=True
    )
    
    assert payload["model"] == model
    assert payload["prompt"] == prompt
    assert payload["stream"] is True
    assert payload["options"]["temperature"] == temperature
    assert payload["options"]["num_predict"] == max_tokens
    assert payload["options"]["stop"] == stop_sequences

@pytest.mark.asyncio
async def test_generate_text_success(mock_config: WitsV3Config, mock_async_client: MagicMock):
    """Test successful text generation."""
    mock_response_data = {"response": "Generated text"}
    mock_httpx_response = MagicMock(spec=httpx.Response)
    mock_httpx_response.json = MagicMock(return_value=mock_response_data)
    mock_httpx_response.raise_for_status = MagicMock() # Does not raise
    mock_async_client.post.return_value = mock_httpx_response

    with patch('httpx.AsyncClient', return_value=mock_async_client):
        interface = OllamaInterface(mock_config)
        response_text = await interface.generate_text("Hello")

    expected_payload = {
        "model": "test-model",
        "prompt": "Hello",
        "stream": False,
        "options": {"temperature": 0.5}
    }
    mock_async_client.post.assert_called_once_with(
        f"{mock_config.ollama_settings.url}/api/generate",
        json=expected_payload
    )
    mock_httpx_response.raise_for_status.assert_called_once()
    assert response_text == "Generated text"

@pytest.mark.asyncio
async def test_generate_text_http_status_error(mock_config: WitsV3Config, mock_async_client: MagicMock):
    """Test handling of HTTPStatusError during text generation."""
    mock_httpx_response = MagicMock(spec=httpx.Response)
    mock_httpx_response.status_code = 500
    mock_httpx_response.text = "Internal Server Error"
    error = httpx.HTTPStatusError("Server Error", request=MagicMock(), response=mock_httpx_response)
    mock_httpx_response.raise_for_status = MagicMock(side_effect=error)
    mock_async_client.post.return_value = mock_httpx_response

    with patch('httpx.AsyncClient', return_value=mock_async_client):
        interface = OllamaInterface(mock_config)
        with pytest.raises(httpx.HTTPStatusError):
            await interface.generate_text("Hello")

    expected_payload = {
        "model": "test-model",
        "prompt": "Hello",
        "stream": False,
        "options": {"temperature": 0.5}
    }
    mock_async_client.post.assert_called_once_with(
        f"{mock_config.ollama_settings.url}/api/generate",
        json=expected_payload
    )
    mock_httpx_response.raise_for_status.assert_called_once()

@pytest.mark.asyncio
async def test_generate_text_request_error(mock_config: WitsV3Config, mock_async_client: MagicMock):
    """Test handling of RequestError during text generation."""
    error = httpx.RequestError("Connection failed", request=MagicMock())
    mock_async_client.post.side_effect = error

    with patch('httpx.AsyncClient', return_value=mock_async_client):
        interface = OllamaInterface(mock_config)
        with pytest.raises(httpx.RequestError):
            await interface.generate_text("Hello")

    expected_payload = {
        "model": "test-model",
        "prompt": "Hello",
        "stream": False,
        "options": {"temperature": 0.5}
    }
    mock_async_client.post.assert_called_once_with(
        f"{mock_config.ollama_settings.url}/api/generate",
        json=expected_payload
    )

@pytest.mark.asyncio
async def test_stream_text_success(mock_config: WitsV3Config, mock_async_client: MagicMock):
    """Test successful text streaming."""
    stream_chunks_data = [
        {"response": "Stream part 1", "done": False},
        {"response": " part 2", "done": False},
        {"response": "", "done": True} # Represents the end of stream with final metadata
    ]
    
    async def mock_aiter_lines():
        for chunk_data in stream_chunks_data:
            yield json.dumps(chunk_data)

    mock_httpx_response = MagicMock(spec=httpx.Response)
    mock_httpx_response.raise_for_status = MagicMock()
    mock_httpx_response.aiter_lines = mock_aiter_lines # Assign the async generator
    
    # The stream method of the client itself needs to be an async context manager returning the response
    mock_stream_context_manager = AsyncMock()
    mock_stream_context_manager.__aenter__.return_value = mock_httpx_response
    mock_stream_context_manager.__aexit__.return_value = None
    mock_async_client.stream.return_value = mock_stream_context_manager

    with patch('httpx.AsyncClient', return_value=mock_async_client):
        interface = OllamaInterface(mock_config)
        result_chunks = []
        async for chunk in interface.stream_text("Stream hello"):
            result_chunks.append(chunk)

    expected_payload = {
        "model": "test-model",
        "prompt": "Stream hello",
        "stream": True,
        "options": {"temperature": 0.5}
    }
    mock_async_client.stream.assert_called_once_with(
        "POST",
        f"{mock_config.ollama_settings.url}/api/generate",
        json=expected_payload
    )
    mock_httpx_response.raise_for_status.assert_called_once()
    assert result_chunks == ["Stream part 1", " part 2"]

@pytest.mark.asyncio
async def test_stream_text_http_status_error(mock_config: WitsV3Config, mock_async_client: MagicMock):
    """Test handling of HTTPStatusError during text streaming."""
    mock_httpx_response = MagicMock(spec=httpx.Response)
    mock_httpx_response.status_code = 400
    mock_httpx_response.text = "Bad Request"
    error = httpx.HTTPStatusError("Bad Request", request=MagicMock(), response=mock_httpx_response)
    mock_httpx_response.raise_for_status = MagicMock(side_effect=error)

    mock_stream_context_manager = AsyncMock()
    mock_stream_context_manager.__aenter__.return_value = mock_httpx_response
    mock_stream_context_manager.__aexit__.return_value = None
    mock_async_client.stream.return_value = mock_stream_context_manager

    with patch('httpx.AsyncClient', return_value=mock_async_client):
        interface = OllamaInterface(mock_config)
        with pytest.raises(httpx.HTTPStatusError):
            async for _ in interface.stream_text("Stream error"):
                pass # pragma: no cover
    
    expected_payload = {
        "model": "test-model",
        "prompt": "Stream error",
        "stream": True,
        "options": {"temperature": 0.5}
    }
    mock_async_client.stream.assert_called_once_with(
        "POST",
        f"{mock_config.ollama_settings.url}/api/generate",
        json=expected_payload
    )
    mock_httpx_response.raise_for_status.assert_called_once()

@pytest.mark.asyncio
async def test_stream_text_request_error(mock_config: WitsV3Config, mock_async_client: MagicMock):
    """Test handling of RequestError during text streaming."""
    error = httpx.RequestError("Network issue", request=MagicMock())
    mock_async_client.stream.side_effect = error

    with patch('httpx.AsyncClient', return_value=mock_async_client):
        interface = OllamaInterface(mock_config)
        with pytest.raises(httpx.RequestError):
            async for _ in interface.stream_text("Stream net error"):
                pass # pragma: no cover

    expected_payload = {
        "model": "test-model",
        "prompt": "Stream net error",
        "stream": True,
        "options": {"temperature": 0.5}
    }
    mock_async_client.stream.assert_called_once_with(
        "POST",
        f"{mock_config.ollama_settings.url}/api/generate",
        json=expected_payload
    )

@pytest.mark.asyncio
async def test_stream_text_json_decode_error(mock_config: WitsV3Config, mock_async_client: MagicMock, capsys):
    """Test handling of JSONDecodeError during stream parsing."""
    stream_chunks_data = [
        json.dumps({"response": "Valid chunk", "done": False}),
        "This is not valid JSON",
        json.dumps({"response": "Another valid chunk", "done": False}),
        json.dumps({"response": "", "done": True})
    ]
    
    async def mock_aiter_lines_with_invalid_json():
        for line in stream_chunks_data:
            yield line

    mock_httpx_response = MagicMock(spec=httpx.Response)
    mock_httpx_response.raise_for_status = MagicMock()
    mock_httpx_response.aiter_lines = mock_aiter_lines_with_invalid_json
    
    mock_stream_context_manager = AsyncMock()
    mock_stream_context_manager.__aenter__.return_value = mock_httpx_response
    mock_stream_context_manager.__aexit__.return_value = None
    mock_async_client.stream.return_value = mock_stream_context_manager

    with patch('httpx.AsyncClient', return_value=mock_async_client):
        interface = OllamaInterface(mock_config)
        result_chunks = []
        async for chunk in interface.stream_text("Stream json error"):
            result_chunks.append(chunk)

    assert result_chunks == ["Valid chunk", "Another valid chunk"]
    captured = capsys.readouterr()
    assert "Warning: Failed to decode JSON stream line: This is not valid JSON" in captured.out

@pytest.mark.asyncio
async def test_get_embedding_success(mock_config: WitsV3Config, mock_async_client: MagicMock):
    """Test successful embedding generation."""
    mock_embedding_data = {"embedding": [0.1, 0.2, 0.3]}
    mock_httpx_response = MagicMock(spec=httpx.Response)
    mock_httpx_response.json = MagicMock(return_value=mock_embedding_data)
    mock_httpx_response.raise_for_status = MagicMock()
    mock_async_client.post.return_value = mock_httpx_response

    with patch('httpx.AsyncClient', return_value=mock_async_client):
        interface = OllamaInterface(mock_config)
        embedding = await interface.get_embedding("Embed this text")

    expected_payload = {
        "model": "test-embedding-model",
        "prompt": "Embed this text"
    }
    mock_async_client.post.assert_called_once_with(
        f"{mock_config.ollama_settings.url}/api/embeddings",
        json=expected_payload
    )
    mock_httpx_response.raise_for_status.assert_called_once()
    assert embedding == [0.1, 0.2, 0.3]

@pytest.mark.asyncio
async def test_get_embedding_override_model(mock_config: WitsV3Config, mock_async_client: MagicMock):
    """Test successful embedding generation with overridden model."""
    mock_embedding_data = {"embedding": [0.4, 0.5, 0.6]}
    mock_httpx_response = MagicMock(spec=httpx.Response)
    mock_httpx_response.json = MagicMock(return_value=mock_embedding_data)
    mock_httpx_response.raise_for_status = MagicMock()
    mock_async_client.post.return_value = mock_httpx_response

    with patch('httpx.AsyncClient', return_value=mock_async_client):
        interface = OllamaInterface(mock_config)
        embedding = await interface.get_embedding("Embed this", model="override-embed-model")

    expected_payload = {
        "model": "override-embed-model", # Overridden model
        "prompt": "Embed this"
    }
    mock_async_client.post.assert_called_once_with(
        f"{mock_config.ollama_settings.url}/api/embeddings",
        json=expected_payload
    )
    assert embedding == [0.4, 0.5, 0.6]

@pytest.mark.asyncio
async def test_get_embedding_http_status_error(mock_config: WitsV3Config, mock_async_client: MagicMock):
    """Test handling of HTTPStatusError during embedding generation."""
    mock_httpx_response = MagicMock(spec=httpx.Response)
    mock_httpx_response.status_code = 503
    mock_httpx_response.text = "Service Unavailable"
    error = httpx.HTTPStatusError("Service Unavailable", request=MagicMock(), response=mock_httpx_response)
    mock_httpx_response.raise_for_status = MagicMock(side_effect=error)
    mock_async_client.post.return_value = mock_httpx_response

    with patch('httpx.AsyncClient', return_value=mock_async_client):
        interface = OllamaInterface(mock_config)
        with pytest.raises(httpx.HTTPStatusError):
            await interface.get_embedding("Embed this")
    
    mock_async_client.post.assert_called_once()
    mock_httpx_response.raise_for_status.assert_called_once()

@pytest.mark.asyncio
async def test_get_embedding_request_error(mock_config: WitsV3Config, mock_async_client: MagicMock):
    """Test handling of RequestError during embedding generation."""
    error = httpx.RequestError("Connection refused", request=MagicMock())
    mock_async_client.post.side_effect = error

    with patch('httpx.AsyncClient', return_value=mock_async_client):
        interface = OllamaInterface(mock_config)
        with pytest.raises(httpx.RequestError):
            await interface.get_embedding("Embed this")
    mock_async_client.post.assert_called_once()

# Tests for get_llm_interface factory function
def test_get_llm_interface_ollama(mock_config: WitsV3Config):
    """Test get_llm_interface returns OllamaInterface for 'ollama' provider."""
    mock_config.llm_interface.default_provider = "ollama"
    with patch('core.llm_interface.OllamaInterface') as MockOllamaInterface:
        interface = get_llm_interface(mock_config)
        MockOllamaInterface.assert_called_once_with(mock_config)
        assert isinstance(interface, MockOllamaInterface)


def test_get_llm_interface_adaptive(mock_config: WitsV3Config):
    """Test get_llm_interface returns AdaptiveLLMInterface for 'adaptive' provider."""
    mock_config.llm_interface.default_provider = "adaptive"
    
    # Mock the AdaptiveLLMInterface and the OllamaInterface it might create internally
    with patch('core.llm_interface.AdaptiveLLMInterface') as MockAdaptiveLLMInterface, \
         patch('core.llm_interface.OllamaInterface') as MockOllamaInterfaceInternal:
        
        # Ensure the internal OllamaInterface is created and passed to AdaptiveLLMInterface
        mock_ollama_internal_instance = MockOllamaInterfaceInternal.return_value
        mock_adaptive_instance = MockAdaptiveLLMInterface.return_value

        interface = get_llm_interface(mock_config)
        
        MockOllamaInterfaceInternal.assert_called_once_with(mock_config)
        MockAdaptiveLLMInterface.assert_called_once_with(mock_config, mock_ollama_internal_instance)
        assert interface == mock_adaptive_instance

def test_get_llm_interface_unsupported(mock_config: WitsV3Config):
    """Test get_llm_interface raises ValueError for an unsupported provider."""
    mock_config.llm_interface.default_provider = "unsupported_provider"
    with pytest.raises(ValueError) as excinfo:
        get_llm_interface(mock_config)
    assert "Unsupported LLM provider: unsupported_provider" in str(excinfo.value)

# Minimal BaseLLMInterface implementation for testing purposes
class ConcreteTestLLMInterface(BaseLLMInterface):
    async def generate_text(self, prompt: str, **kwargs) -> str: return "test"
    async def stream_text(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]: yield "test"
    async def get_embedding(self, text: str, **kwargs) -> list[float]: return [0.1]

@pytest.mark.asyncio
async def test_base_llm_interface_abstract_methods(mock_config):
    """Ensure BaseLLMInterface methods are abstract and can be implemented."""
    # This test primarily checks that the abstract methods are defined and can be called
    # on a concrete implementation without raising NotImplementedError immediately.
    concrete_interface = ConcreteTestLLMInterface(mock_config)
    assert await concrete_interface.generate_text("prompt") == "test"
    async for chunk in concrete_interface.stream_text("prompt"):
        assert chunk == "test"
    assert await concrete_interface.get_embedding("text") == [0.1] 