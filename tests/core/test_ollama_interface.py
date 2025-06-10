"""
Test module for the OllamaInterface in WitsV3.
Tests the OllamaInterface for generating text, streaming text, and getting embeddings.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import asyncio
import json
import logging
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from typing import Dict, Any, List

import httpx

from core.llm_interface import OllamaInterface, BaseLLMInterface
from core.config import WitsV3Config, OllamaSettings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def mock_config():
    """Fixture for a mock WitsV3Config."""
    config = WitsV3Config()
    # Customize for testing
    config.ollama_settings.url = "http://test-ollama:11434"
    config.ollama_settings.default_model = "test-model"
    config.ollama_settings.embedding_model = "test-embedding-model"
    config.ollama_settings.retry_attempts = 3
    config.ollama_settings.retry_delay = 0.01  # Fast for testing
    return config


@pytest.fixture
def ollama_interface(mock_config):
    """Fixture for an OllamaInterface."""
    return OllamaInterface(mock_config)


@pytest.mark.asyncio
async def test_generate_text_success(ollama_interface):
    """Test successful text generation."""
    # Mock successful response
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"response": "Test response"}

    with patch.object(ollama_interface.http_client, 'post', new=AsyncMock(return_value=mock_response)):
        result = await ollama_interface.generate_text("Test prompt")

        assert result == "Test response"
        ollama_interface.http_client.post.assert_called_once()


@pytest.mark.asyncio
async def test_generate_text_retry_success(ollama_interface):
    """Test text generation with retry that succeeds."""
    # Mock HTTP status error response
    http_error = httpx.HTTPStatusError(
        "Service unavailable",
        request=MagicMock(),
        response=MagicMock(status_code=503, text="Service unavailable")
    )

    # Create a mock for the post method
    mock_post = AsyncMock()
    # First call raises the error, second call returns success
    mock_post.side_effect = [
        http_error,  # This will be raised directly, not returned
        MagicMock(
            raise_for_status=MagicMock(),
            json=MagicMock(return_value={"response": "Test response after retry"})
        )
    ]

    # Apply the mock
    with patch.object(ollama_interface.http_client, 'post', side_effect=[
        # First call raises error
        AsyncMock(side_effect=http_error),
        # Second call succeeds
        AsyncMock(return_value=MagicMock(
            raise_for_status=MagicMock(),
            json=MagicMock(return_value={"response": "Test response after retry"})
        ))
    ]):
        result = await ollama_interface.generate_text("Test prompt")

        assert result == "Test response after retry"
        assert ollama_interface.http_client.post.call_count == 2


@pytest.mark.asyncio
async def test_generate_text_max_retries(ollama_interface):
    """Test text generation with max retries exceeded."""
    # Mock HTTP status error for service unavailable
    http_error = httpx.HTTPStatusError(
        "Service unavailable",
        request=MagicMock(),
        response=MagicMock(status_code=503, text="Service unavailable")
    )

    # Patch with side_effect that always raises the error
    with patch.object(ollama_interface.http_client, 'post', side_effect=http_error):
        # Should retry 3 times then raise ValueError
        with pytest.raises(ValueError):
            await ollama_interface.generate_text("Test prompt")

        # Check it was called the expected number of times (1 initial + 2 retries)
        assert ollama_interface.http_client.post.call_count == 3


@pytest.mark.asyncio
async def test_generate_text_404_error(ollama_interface):
    """Test text generation with 404 error (model not found)."""
    # Mock HTTP status error for model not found
    http_error = httpx.HTTPStatusError(
        "Model not found",
        request=MagicMock(),
        response=MagicMock(status_code=404, text="Model not found")
    )

    # Patch with side_effect that raises the error
    with patch.object(ollama_interface.http_client, 'post', side_effect=http_error):
        # Should raise ValueError immediately (no retry for 404)
        with pytest.raises(ValueError) as excinfo:
            await ollama_interface.generate_text("Test prompt")

        # Check the error message mentions the model
        assert "Model not found" in str(excinfo.value)

        # Check it was called only once (no retries)
        assert ollama_interface.http_client.post.call_count == 1


@pytest.mark.asyncio
async def test_stream_text_success(ollama_interface):
    """Test successful text streaming."""
    # Mock context manager for streaming
    mock_stream = AsyncMock()
    mock_stream.__aenter__.return_value = mock_stream
    mock_stream.raise_for_status = MagicMock()

    # Mock aiter_lines to return JSON chunks
    lines = [
        json.dumps({"response": "Test", "done": False}),
        json.dumps({"response": " response", "done": False}),
        json.dumps({"response": ".", "done": True})
    ]
    mock_stream.aiter_lines = AsyncMock()
    mock_stream.aiter_lines.return_value.__aiter__.return_value = AsyncMock()
    mock_stream.aiter_lines.return_value.__aiter__.return_value.__anext__ = AsyncMock(side_effect=[
        lines[0], lines[1], lines[2], StopAsyncIteration
    ])

    # Apply the mock
    with patch.object(ollama_interface.http_client, 'stream', return_value=mock_stream):
        chunks = []
        async for chunk in ollama_interface.stream_text("Test prompt"):
            chunks.append(chunk)

        assert chunks == ["Test", " response"]


@pytest.mark.asyncio
async def test_stream_text_retry_success(ollama_interface):
    """Test text streaming with retry that succeeds."""
    # Mock HTTP status error for service unavailable
    http_error = httpx.HTTPStatusError(
        "Service unavailable",
        request=MagicMock(),
        response=MagicMock(status_code=503, text="Service unavailable")
    )

    # Create a failing mock stream
    mock_stream_fail = AsyncMock()
    mock_stream_fail.__aenter__.side_effect = http_error

    # Create a successful mock stream
    mock_stream_success = AsyncMock()
    mock_stream_success.__aenter__.return_value = mock_stream_success
    mock_stream_success.raise_for_status = MagicMock()

    # Mock aiter_lines to return JSON chunks
    lines = [
        json.dumps({"response": "Test", "done": False}),
        json.dumps({"response": " response", "done": False}),
        json.dumps({"response": ".", "done": True})
    ]
    mock_stream_success.aiter_lines = AsyncMock()
    mock_stream_success.aiter_lines.return_value.__aiter__.return_value = AsyncMock()
    mock_stream_success.aiter_lines.return_value.__aiter__.return_value.__anext__ = AsyncMock(side_effect=[
        lines[0], lines[1], lines[2], StopAsyncIteration
    ])

    # Apply the mocks
    with patch.object(ollama_interface.http_client, 'stream', side_effect=[mock_stream_fail, mock_stream_success]):
        chunks = []
        async for chunk in ollama_interface.stream_text("Test prompt"):
            chunks.append(chunk)

        assert chunks == ["Test", " response"]
        assert ollama_interface.http_client.stream.call_count == 2


@pytest.mark.asyncio
async def test_stream_text_max_retries(ollama_interface):
    """Test text streaming with max retries exceeded."""
    # Mock HTTP status error for service unavailable
    http_error = httpx.HTTPStatusError(
        "Service unavailable",
        request=MagicMock(),
        response=MagicMock(status_code=503, text="Service unavailable")
    )

    # Create a failing mock stream
    mock_stream_fail = AsyncMock()
    mock_stream_fail.__aenter__.side_effect = http_error

    # Apply the mock to always fail
    with patch.object(ollama_interface.http_client, 'stream', return_value=mock_stream_fail):
        chunks = []
        async for chunk in ollama_interface.stream_text("Test prompt"):
            chunks.append(chunk)

        # Should yield an error message after max retries
        assert any("Error:" in chunk for chunk in chunks)

        # The test passes if we get this far without errors


@pytest.mark.asyncio
async def test_get_embedding_success(ollama_interface):
    """Test successful embedding generation."""
    # Mock successful response
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}

    with patch.object(ollama_interface.http_client, 'post', new=AsyncMock(return_value=mock_response)):
        result = await ollama_interface.get_embedding("Test text")

        assert result == [0.1, 0.2, 0.3]
        ollama_interface.http_client.post.assert_called_once()


@pytest.mark.asyncio
async def test_get_embedding_retry_success(ollama_interface):
    """Test embedding generation with retry that succeeds."""
    # Mock HTTP status error for service unavailable
    http_error = httpx.HTTPStatusError(
        "Service unavailable",
        request=MagicMock(),
        response=MagicMock(status_code=503, text="Service unavailable")
    )

    # Apply the mocks
    with patch.object(ollama_interface.http_client, 'post', side_effect=[
        # First call raises error
        AsyncMock(side_effect=http_error),
        # Second call succeeds
        AsyncMock(return_value=MagicMock(
            raise_for_status=MagicMock(),
            json=MagicMock(return_value={"embedding": [0.1, 0.2, 0.3]})
        ))
    ]):
        result = await ollama_interface.get_embedding("Test text")

        assert result == [0.1, 0.2, 0.3]
        assert ollama_interface.http_client.post.call_count == 2


@pytest.mark.asyncio
async def test_is_service_available_success(ollama_interface):
    """Test service availability check - success."""
    # Mock successful response
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()

    with patch.object(ollama_interface.http_client, 'get', new=AsyncMock(return_value=mock_response)):
        result = await ollama_interface.is_service_available()

        assert result is True
        ollama_interface.http_client.get.assert_called_once()


@pytest.mark.asyncio
async def test_is_service_available_failure(ollama_interface):
    """Test service availability check - failure."""
    # Mock HTTP status error
    http_error = httpx.HTTPStatusError(
        "Service unavailable",
        request=MagicMock(),
        response=MagicMock(status_code=503, text="Service unavailable")
    )

    # Apply the mock
    with patch.object(ollama_interface.http_client, 'get', side_effect=http_error):
        result = await ollama_interface.is_service_available()

        assert result is False
        ollama_interface.http_client.get.assert_called_once()


@pytest.mark.asyncio
async def test_shutdown(ollama_interface):
    """Test shutdown method."""
    # Mock the aclose method
    ollama_interface.http_client.aclose = AsyncMock()

    await ollama_interface.shutdown()

    ollama_interface.http_client.aclose.assert_called_once()


if __name__ == "__main__":
    asyncio.run(test_is_service_available_success(OllamaInterface(WitsV3Config())))
