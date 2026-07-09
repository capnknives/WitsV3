"""PII redaction for memory storage and guest-visible responses."""

from __future__ import annotations

import re
from typing import Any

_DEFAULT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("phone", re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("credit_card", re.compile(r"\b(?:\d[ -]*?){13,16}\b")),
)

_REDACTED = "[REDACTED]"


def _compiled_patterns(config: Any | None) -> tuple[tuple[str, re.Pattern[str]], ...]:
    security = getattr(config, "security", None) if config else None
    pii = getattr(security, "pii_redaction", None) if security else None
    extra = getattr(pii, "patterns", None) if pii else None
    if not extra:
        return _DEFAULT_PATTERNS
    compiled: list[tuple[str, re.Pattern[str]]] = list(_DEFAULT_PATTERNS)
    for item in extra:
        try:
            compiled.append((f"custom_{len(compiled)}", re.compile(str(item), re.I)))
        except re.error:
            continue
    return tuple(compiled)


def is_pii_redaction_enabled(config: Any | None, *, role: str = "owner") -> bool:
    security = getattr(config, "security", None) if config else None
    pii = getattr(security, "pii_redaction", None) if security else None
    if pii is None or not getattr(pii, "enabled", False):
        return False
    role_key = (role or "owner").lower()
    if role_key == "owner" and not getattr(pii, "redact_owner", False):
        return False
    return True


def redact_pii(text: str, config: Any | None = None) -> str:
    """Replace common PII patterns with [REDACTED]."""
    if not text:
        return text
    out = text
    for _name, pattern in _compiled_patterns(config):
        out = pattern.sub(_REDACTED, out)
    return out


def maybe_redact_for_storage(text: str, config: Any | None, *, role: str = "owner") -> str:
    security = getattr(config, "security", None) if config else None
    pii = getattr(security, "pii_redaction", None) if security else None
    if pii is None or not getattr(pii, "enabled", False):
        return text
    if not getattr(pii, "redact_before_store", True):
        return text
    if role == "owner" and not getattr(pii, "redact_owner", False):
        return text
    return redact_pii(text, config)


def maybe_redact_for_guest(text: str, config: Any | None, *, role: str) -> str:
    if role != "guest" and not role.startswith("family"):
        return text
    if not is_pii_redaction_enabled(config, role=role):
        return text
    return redact_pii(text, config)
