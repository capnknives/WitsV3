"""Inbound text cleaning for the workspace (contract section 8, steps a and b)."""

import unicodedata

from .store import WorkspaceError

# Reason: Cc, Cf, Co, Cs and Cn cover zero-width, bidirectional and tag characters, but U+2028 and
# U+2029 are Zl/Zp, and the variation selectors are Mn, so they are listed explicitly.
_DROP_CATEGORIES = frozenset({"Cc", "Cf", "Co", "Cs", "Cn"})
_KEEP = frozenset({"\n", "\t"})
_EXTRA_DROP_RANGES = (
    (0x2028, 0x2029),
    (0xFE00, 0xFE0F),  # variation selectors
    (0xE0100, 0xE01EF),  # variation selectors supplement
)


def _dropped(ch: str) -> bool:
    if ch in _KEEP:
        return False
    cp = ord(ch)
    if any(lo <= cp <= hi for lo, hi in _EXTRA_DROP_RANGES):
        return True
    return unicodedata.category(ch) in _DROP_CATEGORIES


def clean_text(text: str, max_len: int, required: bool = True) -> str:
    """Remove invisible and control characters, then enforce the length limit.

    Args:
        text: Raw text from an agent or the owner.
        max_len: Maximum length in characters after cleaning.
        required: When true, text that is empty after cleaning is rejected.

    Returns:
        The cleaned text.

    Raises:
        WorkspaceError: ``invalid_argument`` if the text is not a string, is empty while
            required, or is longer than ``max_len``. Over-limit text is never truncated.
    """
    if not isinstance(text, str):
        raise WorkspaceError("invalid_argument", "text must be a string")
    cleaned = "".join(ch for ch in text if not _dropped(ch))
    if required and not cleaned.strip():
        raise WorkspaceError("invalid_argument", "text must not be empty")
    if len(cleaned) > max_len:
        raise WorkspaceError(
            "invalid_argument", f"text is {len(cleaned)} characters; the limit is {max_len}"
        )
    return cleaned
