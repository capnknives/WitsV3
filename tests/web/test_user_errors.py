"""Tests for web UI user-facing error formatting."""

from web.user_errors import format_chat_error, is_ollama_unavailable


def test_is_ollama_unavailable_detects_connection_errors():
    assert is_ollama_unavailable("Failed to connect to Ollama at http://localhost:11434")
    assert is_ollama_unavailable("Connection refused")
    assert not is_ollama_unavailable("Model not found")


def test_format_chat_error_ollama():
    fmt = format_chat_error(
        ValueError(
            "Failed to connect to Ollama at http://localhost:11434. Please ensure Ollama is running."
        ),
        ollama_url="http://localhost:11434",
    )
    assert fmt["code"] == "ollama_unavailable"
    assert "Can't reach Ollama" in fmt["message"]
    assert "ollama serve" in fmt["hint"]
    assert "11434" in fmt["hint"]


def test_format_chat_error_generic():
    fmt = format_chat_error("Something unexpected broke")
    assert fmt["code"] == "generic"
    assert "unexpected" in fmt["message"]
