"""Path checks for the export, document upload and document delete routes (WP F2b).

Contract: docs/architecture/folder-access.md sections 2-4. The owner's exports land in
the outbox through ``path_access.check``; guests keep the fixed runtime exports folder
but still meet the never-list. Document routes keep ``documents_path`` and add the
never-list for mode "index", so secrets files are never written or removed there.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from core import path_access, runtime_paths

NEVER_DETAIL = "refused: protected file name or folder (folder-access section 4)"


def export_name(file_path: str | None) -> str | None:
    """The single-part export file name, a server-chosen one, or None when invalid."""
    if not file_path:
        return f"chat_export_{uuid.uuid4().hex[:8]}.txt"
    rel = Path(file_path.replace("\\", "/"))
    if rel.is_absolute() or ".." in rel.parts or len(rel.parts) != 1:
        return None
    return rel.name


async def export_target(name: str, *, guest: bool, config: Any) -> Path:
    """Where an export named *name* is written; raises PermissionError when refused.

    The returned path is the resolved one that was checked, so callers write exactly
    there and a symlink planted in the folder cannot redirect the write.
    """
    if guest:
        folder = runtime_paths.exports_dir()
        folder.mkdir(parents=True, exist_ok=True)
        root = folder.resolve()
        target = (root / name).resolve()
        if not target.is_relative_to(root) or target == root:
            raise PermissionError("refused: export must stay inside the exports folder")
        if path_access.is_never(root / name, "write") or path_access.is_never(target, "write"):
            raise PermissionError(NEVER_DETAIL)
        return target
    folder = runtime_paths.outbox_dir(getattr(getattr(config, "folders", None), "base", None))
    if path_access.is_never(folder / name, "write"):
        raise PermissionError(NEVER_DETAIL)
    target = await path_access.check(folder / name, "write", role="owner", config=config)
    folder.mkdir(parents=True, exist_ok=True)
    return target


def document_target(docs_dir: Path, name: str) -> Path:
    """Resolve *name* inside *docs_dir*.

    Raises ValueError when it falls outside (or is the folder itself) and
    PermissionError when the never-list refuses the requested name or the resolved path.
    """
    root = docs_dir.resolve()
    target = (root / name).resolve()
    if not target.is_relative_to(root) or target == root:
        raise ValueError("invalid document name")
    # Reason: Windows drops trailing dots and spaces and treats "name:stream" as name, so
    # those spellings are refused on every platform before the name rules see them.
    if (
        path_access.alias_reason(name, windows=True)
        or path_access.is_never(root / name, "index")
        or path_access.is_never(target, "index")
    ):
        raise PermissionError(NEVER_DETAIL)
    return target
