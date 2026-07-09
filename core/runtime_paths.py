"""Central runtime path layout for WitsV3 mutable data.

All personal/runtime files live under ``var/`` by default (configurable via
``runtime_paths.root``). Legacy top-level folders (``data/``, ``logs/``, …)
are migrated into ``var/`` on first startup when the new location is absent.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger("WitsV3.RuntimePaths")

# Package location (stable); runtime paths resolve from the process cwd so tests
# and scripts that chdir into an isolated directory stay sandboxed.
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PACKAGE_ROOT  # re-export for callers that need the install path


def project_root() -> Path:
    """Directory used for var/ layout — normally the repo root (cwd at startup)."""
    override = os.getenv("WITSV3_PROJECT_ROOT")
    if override:
        return Path(override)
    return Path.cwd()

RUNTIME_ROOT_NAME = "var"

SUBDIRS = ("data", "documents", "exports", "logs", "workspace", "cache", "sessions")


def runtime_root(root: str = RUNTIME_ROOT_NAME) -> Path:
    return project_root() / root


def runtime_subdir(name: str, root: str = RUNTIME_ROOT_NAME) -> Path:
    return runtime_root(root) / name


def data_dir(root: str = RUNTIME_ROOT_NAME) -> Path:
    return runtime_subdir("data", root)


def documents_dir(root: str = RUNTIME_ROOT_NAME) -> Path:
    return runtime_subdir("documents", root)


def exports_dir(root: str = RUNTIME_ROOT_NAME) -> Path:
    return runtime_subdir("exports", root)


def logs_dir(root: str = RUNTIME_ROOT_NAME) -> Path:
    return runtime_subdir("logs", root)


def workspace_dir(root: str = RUNTIME_ROOT_NAME) -> Path:
    return runtime_subdir("workspace", root)


def cache_dir(root: str = RUNTIME_ROOT_NAME) -> Path:
    return runtime_subdir("cache", root)


def sessions_dir(root: str = RUNTIME_ROOT_NAME) -> Path:
    return runtime_subdir("sessions", root)


def main_log_path(root: str = RUNTIME_ROOT_NAME) -> Path:
    return logs_dir(root) / "witsv3.log"


def guest_profiles_path(root: str = RUNTIME_ROOT_NAME) -> Path:
    return data_dir(root) / "guest_profiles.json"


def guest_audit_dir(root: str = RUNTIME_ROOT_NAME) -> Path:
    return data_dir(root) / "guest_audit"


def guest_user_profiles_dir(root: str = RUNTIME_ROOT_NAME) -> Path:
    return data_dir(root) / "guest_user_profiles"


def upgrade_runtime_path(path: str, root: str = RUNTIME_ROOT_NAME) -> str:
    """Map legacy top-level runtime paths to the var/ layout."""
    norm = path.replace("\\", "/")
    if norm.startswith(f"{root}/"):
        return norm
    for sub in SUBDIRS:
        if norm == sub or norm.startswith(f"{sub}/"):
            return f"{root}/{norm}"
    return norm


def workspace_subpath(name: str, root: str = RUNTIME_ROOT_NAME) -> str:
    return f"{root}/workspace/{name}"


def exports_subpath(filename: str, root: str = RUNTIME_ROOT_NAME) -> str:
    return f"{root}/exports/{filename}"


def resolve_project_path(relative: str | Path, root: str = RUNTIME_ROOT_NAME) -> Path:
    """Resolve a project-relative path, preferring the upgraded var/ location."""
    rel = Path(relative)
    if rel.is_absolute():
        return rel
    upgraded = upgrade_runtime_path(str(relative).replace("\\", "/"), root)
    new_path = project_root() / upgraded
    if new_path.exists():
        return new_path
    legacy_path = project_root() / rel
    if legacy_path.exists():
        return legacy_path
    return new_path


def migrate_legacy_runtime_dirs(root: str = RUNTIME_ROOT_NAME) -> list[str]:
    """Move legacy top-level runtime dirs into var/ when the target is absent."""
    moved: list[str] = []
    base = project_root()
    rt = base / root
    rt.mkdir(parents=True, exist_ok=True)
    for sub in SUBDIRS:
        legacy = base / sub
        target = rt / sub
        if legacy.exists() and legacy.is_dir() and not target.exists():
            shutil.move(str(legacy), str(target))
            moved.append(f"{sub}/ -> {root}/{sub}/")
            logger.info("Migrated runtime directory %s -> %s", legacy, target)
        elif not target.exists():
            target.mkdir(parents=True, exist_ok=True)
    return moved


def ensure_runtime_layout(root: str = RUNTIME_ROOT_NAME) -> list[str]:
    """Create var/ subdirs and migrate legacy top-level folders if needed."""
    return migrate_legacy_runtime_dirs(root)
