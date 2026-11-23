"""Caller role normalisation (RT-10): only the exact string "owner" has owner rights.

Every place a role enters a turn calls :func:`normalize_role` once, so code downstream
only ever sees ``"owner"`` or ``"guest"`` and its ``== "guest"`` checks stay correct.
"""

from __future__ import annotations

from typing import Any, Literal

OWNER = "owner"
GUEST = "guest"

Role = Literal["owner", "guest"]


def normalize_role(role: Any) -> Role:
    """Collapse any caller role to ``"owner"`` or ``"guest"``.

    Args:
        role: The role as the caller supplied it (may be None, empty or unknown).

    Returns:
        ``"owner"`` only when ``role`` is exactly the str ``"owner"``; otherwise ``"guest"``.
    """
    # Reason: fail closed. No stripping or lower-casing: "Owner", "owner " and "OWNER" are
    # not the value the auth layer issues, so they must not be promoted to owner.
    if isinstance(role, str) and role == OWNER:
        return OWNER
    return GUEST


def is_owner(role: Any) -> bool:
    """True only when ``role`` is exactly the str ``"owner"``.

    Args:
        role: The role as the caller supplied it.

    Returns:
        Whether the caller has owner rights.
    """
    return normalize_role(role) == OWNER
