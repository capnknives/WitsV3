"""Tests for core.user_errors (CLI + web)."""

from core.user_errors import format_chat_error, format_cli_error, is_ollama_unavailable


def test_is_ollama_unavailable_detects_connection_refused():
    assert is_ollama_unavailable("Failed to connect to Ollama at localhost:11434")


def test_format_chat_error_ollama():
    fmt = format_chat_error("connection refused on port 11434")
    assert fmt["code"] == "ollama_unavailable"
    assert "Ollama" in fmt["message"]
    assert fmt["hint"]


def test_format_cli_error_includes_hint():
    text = format_cli_error("all connection attempts failed")
    assert "Ollama" in text
    assert "ollama serve" in text.lower() or "tray" in text.lower()
