"""Tests for the LLM interface module."""

import json
import httpx
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from typing import AsyncGenerator

from core.llm_interface import OllamaInterface, BaseLLMInterface, get_llm_interface
from core.config import WitsV3Config


@pytest.fixture
def mock_config() -> WitsV3Config:
    """Create a mock configuration."""
    config = MagicMock(spec=WitsV3Config)

    # Create nested mock objects manually
    config.ollama_settings = MagicMock()
    config.ollama_settings.url = "http://localhost:11434"
    config.ollama_settings.default_model = "test-model"
    config.ollama_settings.embedding_model = "test-embedding-model"
    config.ollama_settings.request_timeout = 60

    config.agents = MagicMock()
    config.agents.default_temperature = 0.5

    config.llm_interface = MagicMock()
    config.llm_interface.default_provider = "ollama"

    return config


@pytest.fixture
def mock_async_client() -> MagicMock:
    """Create a mock AsyncClient."""
    client = MagicMock(spec=httpx.AsyncClient)
    # Ensure all async methods are AsyncMocks
    client.post = AsyncMock()
    client.stream = AsyncMock()
    return client


@pytest.mark.asyncio
async def test_ollama_interface_initialization(mock_config: WitsV3Config):
    """Test OllamaInterface initialization."""
    with patch('httpx.AsyncClient'):
        interface = OllamaInterface(mock_config)
        assert interface.config == mock_config
        assert interface.ollama_settings == mock_config.ollama_settings


@pytest.mark.asyncio
async def test_prepare_payload_default_params(mock_config: WitsV3Config):
    """Test payload preparation with default parameters."""
    with patch('httpx.AsyncClient'):
        interface = OllamaInterface(mock_config)
        payload = await interface._prepare_payload("Test prompt")

    expected = {
        "model": "test-model",
        "prompt": "Test prompt",
        "stream": False,
        "options": {"temperature": 0.5}
    }
    assert payload == expected


@pytest.mark.asyncio
async def test_prepare_payload_override_params(mock_config: WitsV3Config):
    """Test payload preparation with overridden parameters."""
    with patch('httpx.AsyncClient'):
        interface = OllamaInterface(mock_config)
        payload = await interface._prepare_payload(
            "Test prompt",
            model="custom-model",
            temperature=0.8,
            max_tokens=100,
            stop_sequences=["stop1", "stop2"],
            stream=True
        )

    expected = {
        "model": "custom-model",
        "prompt": "Test prompt",
        "stream": True,
        "options": {
            "temperature": 0.8,
            "num_predict": 100,
            "stop": ["stop1", "stop2"]
        }
    }
    assert payload == expected


@pytest.mark.asyncio
async def test_generate_text_success(mock_config: WitsV3Config, mock_async_client: MagicMock):
    """Test successful text generation."""
    mock_response_data = {"response": "Generated text response"}
    mock_httpx_response = MagicMock(spec=httpx.Response)
    mock_httpx_response.json = MagicMock(return_value=mock_response_data)
    mock_httpx_response.raise_for_status = MagicMock()
    mock_async_client.post.return_value = mock_httpx_response

    with patch('httpx.AsyncClient', return_value=mock_async_client):
        interface = OllamaInterface(mock_config)
        response = await interface.generate_text("Hello")

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
    assert response == "Generated text response"


@pytest.mark.asyncio
async def test_generate_text_http_status_error(mock_config: WitsV3Config, mock_async_client: MagicMock):
    """Test handling of HTTPStatusError during text generation."""
    mock_httpx_response = MagicMock(spec=httpx.Response)
    mock_httpx_response.status_code = 500
    mock_httpx_response.text = "Internal Server Error"
    error = httpx.HTTPStatusError("Internal Server Error", request=MagicMock(), response=mock_httpx_response)
    mock_httpx_response.raise_for_status = MagicMock(side_effect=error)
    mock_async_client.post.return_value = mock_httpx_response

    with patch('httpx.AsyncClient', return_value=mock_async_client):
        interface = OllamaInterface(mock_config)
        with pytest.raises(httpx.HTTPStatusError):
            await interface.generate_text("Hello")

    mock_async_client.post.assert_called_once()
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

    # Properly mock the async context manager
    mock_stream_context_manager = AsyncMock()
    mock_stream_context_manager.__aenter__ = AsyncMock(return_value=mock_httpx_response)
    mock_stream_context_manager.__aexit__ = AsyncMock(return_value=None)
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
    mock_stream_context_manager.__aenter__ = AsyncMock(return_value=mock_httpx_response)
    mock_stream_context_manager.__aexit__ = AsyncMock(return_value=None)
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
    mock_stream_context_manager.__aenter__ = AsyncMock(return_value=mock_httpx_response)
    mock_stream_context_manager.__aexit__ = AsyncMock(return_value=None)
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
    with patch('httpx.AsyncClient'):
        interface = get_llm_interface(mock_config)
        assert isinstance(interface, OllamaInterface)  # Check actual instance type


def test_get_llm_interface_adaptive(mock_config: WitsV3Config):
    """Test get_llm_interface returns AdaptiveLLMInterface for 'adaptive' provider."""
    mock_config.llm_interface.default_provider = "adaptive"

    # Mock the AdaptiveLLMInterface import at the correct location
    with patch('core.adaptive_llm_interface.AdaptiveLLMInterface') as MockAdaptiveLLMInterface, \
         patch('httpx.AsyncClient'):

        mock_adaptive_instance = MockAdaptiveLLMInterface.return_value
        interface = get_llm_interface(mock_config)

        # Verify AdaptiveLLMInterface was called with config and an OllamaInterface
        MockAdaptiveLLMInterface.assert_called_once()
        call_args = MockAdaptiveLLMInterface.call_args
        assert call_args[0][0] == mock_config  # First arg is config
        assert isinstance(call_args[0][1], OllamaInterface)  # Second arg is OllamaInterface
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
