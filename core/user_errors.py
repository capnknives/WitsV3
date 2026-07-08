"""
User-facing error messages for WitsV3 (CLI and web).

Translates low-level exceptions (especially Ollama connection failures after
retries) into actionable copy.
"""

import re
from typing import Any

_OLLAMA_UNAVAILABLE = (
    re.compile(r"failed to connect to ollama", re.I),
    re.compile(r"ollama service may be unavailable", re.I),
    re.compile(r"connection refused", re.I),
    re.compile(r"connect call failed.*11434", re.I),
    re.compile(r"all connection attempts failed", re.I),
    re.compile(r"name or service not known.*11434", re.I),
)


def is_ollama_unavailable(text: str) -> bool:
    """Return True if *text* looks like an Ollama connectivity failure."""
    if not text:
        return False
    return any(p.search(text) for p in _OLLAMA_UNAVAILABLE)


def format_chat_error(
    exc_or_text: Any, ollama_url: str = "http://localhost:11434"
) -> dict[str, str]:
    """
    Build a structured user-facing error from an exception or message string.

    Returns:
        Dict with keys: code ("ollama_unavailable" | "generic"), message, hint.
    """
    raw = str(exc_or_text).strip() if exc_or_text is not None else "Unknown error"

    if is_ollama_unavailable(raw):
        return {
            "code": "ollama_unavailable",
            "message": "Can't reach Ollama — WITS needs it to think.",
            "hint": (
                "Start the Ollama app (look for the tray icon), or run `ollama serve`. "
                f"Expected at {ollama_url.rstrip('/')}. "
                "On Windows: %LOCALAPPDATA%\\Programs\\Ollama\\ollama app.exe"
            ),
        }

    for prefix in ("An error occurred during orchestration: ", "Error processing request: "):
        if raw.startswith(prefix):
            raw = raw[len(prefix) :]

    return {
        "code": "generic",
        "message": raw or "Something went wrong. Please try again.",
        "hint": "",
    }


def format_cli_error(exc_or_text: Any, ollama_url: str = "http://localhost:11434") -> str:
    """Single string for CLI display (message + optional hint)."""
    fmt = format_chat_error(exc_or_text, ollama_url=ollama_url)
    if fmt.get("hint"):
        return f"{fmt['message']}\n\n{fmt['hint']}"
    return fmt["message"]
