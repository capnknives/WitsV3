"""Persist web/CLI chat sessions under var/sessions/."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from core.runtime_paths import sessions_dir
from core.schemas import ConversationHistory

logger = logging.getLogger("WitsV3.SessionStore")

_SAFE_ID_RE = re.compile(r"^[\w:\-]+$")


def _safe_session_filename(session_id: str) -> str | None:
    """Return a filesystem-safe filename stem for a session id."""
    if not session_id or not _SAFE_ID_RE.match(session_id):
        return None
    return session_id.replace(":", "_")


def session_file_path(session_id: str, root: str = "var") -> Path | None:
    stem = _safe_session_filename(session_id)
    if not stem:
        return None
    return sessions_dir(root) / f"{stem}.json"


def persist_session(
    conversation: ConversationHistory,
    root: str = "var",
) -> Path | None:
    """Write one session to disk (atomic replace)."""
    path = session_file_path(conversation.session_id, root)
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    payload = conversation.model_dump(mode="json")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)
    return path


def load_sessions(root: str = "var") -> dict[str, ConversationHistory]:
    """Load all persisted sessions from var/sessions/."""
    store = sessions_dir(root)
    if not store.is_dir():
        return {}
    sessions: dict[str, ConversationHistory] = {}
    for path in store.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            conv = ConversationHistory.model_validate(data)
            sessions[conv.session_id] = conv
        except Exception as e:
            logger.warning("Skipping corrupt session file %s: %s", path, e)
    return sessions


def load_persisted_sessions_into(
    target: dict[str, ConversationHistory],
    root: str = "var",
) -> int:
    """Merge disk sessions into an in-memory session_histories dict."""
    loaded = load_sessions(root)
    for sid, conv in loaded.items():
        if sid not in target:
            target[sid] = conv
    return len(loaded)
