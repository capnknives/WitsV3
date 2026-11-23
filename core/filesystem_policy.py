"""Filesystem read allowlist — owner vs guest/role read roots."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from core.roles import is_owner
from core.runtime_paths import documents_dir, inbox_dir, outbox_dir, project_root, projects_dir
from core.safe_code_editor import PROJECT_ROOT


def _normalize_roots(paths: list[str] | tuple[str, ...] | None) -> list[Path]:
    roots: list[Path] = []
    for raw in paths or []:
        text = str(raw).strip()
        if not text:
            continue
        try:
            roots.append(Path(text).expanduser().resolve())
        except OSError:
            continue
    return roots


def _env_read_roots() -> list[Path]:
    raw = os.getenv("WITSV3_READ_ROOTS", "").strip()
    if not raw:
        return []
    return _normalize_roots(raw.split(os.pathsep))


def project_read_root() -> Path:
    return PROJECT_ROOT.resolve()


BASE_FOLDER_FNS = (inbox_dir, documents_dir, outbox_dir, projects_dir)
logger = logging.getLogger("WitsV3.FilesystemPolicy")


def _base_setting(config: Any | None) -> str | None:
    value = getattr(getattr(config, "folders", None), "base", None) if config else None
    return value if isinstance(value, str) else None


def configured_read_roots(config: Any | None = None) -> list[Path]:
    """Read roots: project, the four base folders, security roots, env variable."""
    base = _base_setting(config)
    roots: list[Path] = [project_read_root()]
    # Reason: the base folders are the owner's workspace, so they are always readable.
    roots.extend(_normalize_roots([str(f(base)) for f in BASE_FOLDER_FNS]))
    security = getattr(config, "security", None) if config else None
    extra = getattr(security, "filesystem_read_roots", None) if security else None
    roots.extend(_normalize_roots(extra))
    for env_root in _env_read_roots():
        if env_root not in roots:
            roots.append(env_root)
    # De-dupe while preserving order
    seen: set[Path] = set()
    unique: list[Path] = []
    for root in roots:
        if root not in seen:
            seen.add(root)
            unique.append(root)
    return unique


def read_roots_for_role(
    role: str,
    config: Any | None = None,
    *,
    guest_policy: dict[str, Any] | None = None,
) -> list[Path]:
    """Resolve read roots for owner, guest, or RBAC role name."""
    if is_owner(role):
        return configured_read_roots(config)
    # Reason: RT-10, fail closed. None, "" and owner look-alikes ("Owner", "owner ") get
    # the legacy guest roots, never the owner's.
    role_key = role.strip().lower() if isinstance(role, str) else ""
    if role_key in ("owner", ""):
        role_key = "guest"

    policy = guest_policy
    if policy is None:
        from core.guest_policy_loader import load_guest_policy

        policy = load_guest_policy()

    roles = policy.get("roles") if isinstance(policy, dict) else None
    if isinstance(roles, dict) and role_key in roles:
        role_cfg = roles[role_key]
        roots_cfg = role_cfg.get("read_roots") if isinstance(role_cfg, dict) else None
        if roots_cfg:
            resolved: list[Path] = []
            for entry in roots_cfg:
                token = str(entry).strip().lower()
                if token in ("project", "project_root"):
                    resolved.append(project_read_root())
                elif token in ("downloads", "user_downloads"):
                    for root in _roots_without_documents(config):
                        if root != project_read_root():
                            resolved.append(root)
                else:
                    resolved.extend(_normalize_roots([str(entry)]))
            if resolved:
                return resolved

    # Reason: legacy guest, and any role the policy does not define, read the project
    # only; an unknown role must not fall through to the owner's wider roots.
    return [project_read_root()]


def _roots_without_documents(config: Any | None) -> list[Path]:
    # Reason: guests keep pre-F5 limits (contract section 6); uploads are the owner's alone.
    documents = _normalize_roots([str(documents_folder(config))])
    return [r for r in configured_read_roots(config) if r not in documents]


def resolve_allowed_read_path(
    file_path: str,
    *,
    role: str = "owner",
    config: Any | None = None,
) -> Path:
    """Resolve *file_path* for reading, or raise PermissionError.

    Thin wrapper over :func:`core.path_access.enforce` kept for older callers. It is
    synchronous, so it sees no grants; tools use the async ``path_access.check``.
    """
    from core.path_access import enforce

    return enforce(file_path, "read", role=role, config=config)


def documents_folder(config: Any | None = None) -> Path:
    """Where uploaded documents live: ``<folders.base>/documents``, never documents_path."""
    return documents_dir(_base_setting(config))


def document_ingest_roots(config: Any | None = None) -> list[Path]:
    """Inbox, documents, then security.document_ingest_roots (folder-access section 5)."""
    security = getattr(config, "security", None) if config else None
    extra = getattr(security, "document_ingest_roots", None) if security else None
    # Reason: [] means "no extra roots"; inbox and documents are indexable regardless.
    fixed = _normalize_roots([str(inbox_dir(_base_setting(config))), str(documents_folder(config))])
    return fixed + [r for r in _normalize_roots(extra) if r not in fixed]


def old_documents_folder(config: Any | None = None) -> Path | None:
    """The old ``document_rag.documents_path`` when it is set and is not the documents folder."""
    rag = getattr(config, "document_rag", None) if config else None
    old = getattr(rag, "documents_path", None) if rag else None
    if not isinstance(old, str) or not old.strip():
        return None
    try:
        folder = Path(old).expanduser()
        folder = (folder if folder.is_absolute() else project_root() / folder).resolve()
        return None if folder == documents_folder(config).resolve() else folder
    except (OSError, RuntimeError, ValueError):
        return None


def warn_old_documents_path(config: Any | None = None) -> int:
    """Log one warning when the old ``document_rag.documents_path`` still holds files.

    Nothing there is moved, deleted or indexed; returns the file count (0 when no warning).
    """
    folder = old_documents_folder(config)
    try:
        if folder is None or not folder.is_dir():
            return 0
        count = sum(1 for p in folder.rglob("*") if p.is_file() and p.name != ".gitkeep")
    except OSError:
        return 0
    if count:
        logger.warning(
            "document_rag.documents_path %s holds %d file(s). Uploads now go to %s. Those "
            "files stay searchable from what is already indexed, but are not re-indexed if "
            "they change and do not appear on the Documents page; move them into %s.",
            folder,
            count,
            documents_folder(config),
            documents_folder(config),
        )
    return count
