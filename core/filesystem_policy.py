"""Filesystem read allowlist — owner vs guest/role read roots."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from core.runtime_paths import inbox_dir, outbox_dir, projects_dir
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


def _base_setting(config: Any | None) -> str | None:
    value = getattr(getattr(config, "folders", None), "base", None) if config else None
    return value if isinstance(value, str) else None


def configured_read_roots(config: Any | None = None) -> list[Path]:
    """Read roots: project, base inbox/outbox/projects, security roots, env variable."""
    base = _base_setting(config)
    roots: list[Path] = [project_read_root()]
    # Reason: the base folders are the owner's workspace, so they are always readable.
    roots.extend(_normalize_roots([str(f(base)) for f in (inbox_dir, outbox_dir, projects_dir)]))
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
    role_key = (role or "owner").strip().lower()
    if role_key in ("owner", ""):
        return configured_read_roots(config)

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
                    for root in configured_read_roots(config):
                        if root != project_read_root():
                            resolved.append(root)
                else:
                    resolved.extend(_normalize_roots([str(entry)]))
            if resolved:
                return resolved

    # Legacy guest: project only
    if role_key == "guest":
        return [project_read_root()]

    return configured_read_roots(config)


def _path_within_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def resolve_allowed_read_path(
    file_path: str,
    *,
    role: str = "owner",
    config: Any | None = None,
) -> Path:
    """Resolve *file_path* if it lies under an allowed read root for *role*.

    Raises PermissionError when the path escapes all allowed roots.
    """
    candidate = Path(file_path)
    resolved = (
        candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()
    )
    allowed_roots = read_roots_for_role(role, config)
    if any(_path_within_root(resolved, root) for root in allowed_roots):
        return resolved
    roots_display = ", ".join(str(r) for r in allowed_roots)
    raise PermissionError(
        f"Refusing to read outside allowed directories for role '{role}': {resolved}. "
        f"Allowed roots: {roots_display}"
    )


def document_ingest_roots(config: Any | None = None) -> list[Path]:
    """Inbox plus security.document_ingest_roots (documents_path is added by the tool)."""
    security = getattr(config, "security", None) if config else None
    extra = getattr(security, "document_ingest_roots", None) if security else None
    # Reason: [] means "no extra roots"; the inbox is indexable regardless.
    inbox = _normalize_roots([str(inbox_dir(_base_setting(config)))])
    return inbox + [r for r in _normalize_roots(extra) if r not in inbox]
