"""Re-export user error helpers from core (web + CLI share one implementation)."""

from core.user_errors import format_chat_error, is_ollama_unavailable

__all__ = ["format_chat_error", "is_ollama_unavailable"]
