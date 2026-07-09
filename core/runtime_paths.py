"""Central runtime path layout for WitsV3 mutable data.

All personal/runtime files live under ``var/`` by default (configurable via
``runtime_paths.root``). Legacy top-level folders (``data/``, ``logs/``, …)
are migrated into ``var/`` on startup. When ``var/<subdir>/`` already exists
(e.g. git-tracked ``var/data/`` templates), legacy contents are merged in and
the empty legacy folder is removed.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger("WitsV3.RuntimePaths")

# Files tracked in git under var/data/ — keep the repo copy on conflict unless
# legacy is clearly the live runtime file (handled via _should_prefer_legacy_file).
TRACKED_DATA_FILENAMES = frozenset(
    {
        "README.md",
        "mcp_config.json",
        "mcp_tools.json",
    }
)

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

# Canonical runtime subdirs under var/. ``user_files`` replaced ``documents`` in Phase 4.
SUBDIRS = ("data", "user_files", "exports", "logs", "workspace", "cache", "sessions")

# Legacy top-level / config path names mapped to canonical SUBDIR names.
LEGACY_SUBDIR_ALIASES: dict[str, str] = {"documents": "user_files"}


def runtime_root(root: str = RUNTIME_ROOT_NAME) -> Path:
    return project_root() / root


def runtime_subdir(name: str, root: str = RUNTIME_ROOT_NAME) -> Path:
    return runtime_root(root) / name


def data_dir(root: str = RUNTIME_ROOT_NAME) -> Path:
    return runtime_subdir("data", root)


def user_files_dir(root: str = RUNTIME_ROOT_NAME) -> Path:
    """Personal RAG/upload corpus (PDFs, notes). Not repo ``docs/``."""
    return runtime_subdir("user_files", root)


def documents_dir(root: str = RUNTIME_ROOT_NAME) -> Path:
    """Deprecated alias for :func:`user_files_dir`."""
    return user_files_dir(root)


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


def _canonical_subdir(name: str) -> str:
    return LEGACY_SUBDIR_ALIASES.get(name, name)


def upgrade_runtime_path(path: str, root: str = RUNTIME_ROOT_NAME) -> str:
    """Map legacy top-level runtime paths to the var/ layout."""
    norm = path.replace("\\", "/")
    # var/documents → var/user_files (Phase 4 rename)
    if norm == f"{root}/documents" or norm.startswith(f"{root}/documents/"):
        suffix = norm[len(f"{root}/documents") :]
        return f"{root}/user_files{suffix}"
    if norm.startswith(f"{root}/"):
        return norm
    for legacy, canonical in LEGACY_SUBDIR_ALIASES.items():
        if norm == legacy or norm.startswith(f"{legacy}/"):
            suffix = norm[len(legacy) :]
            return f"{root}/{canonical}{suffix}"
    first = norm.split("/", 1)[0]
    canonical = _canonical_subdir(first)
    if canonical != first:
        suffix = norm[len(first) :]
        return f"{root}/{canonical}{suffix}"
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


def _is_tracked_data_file(name: str) -> bool:
    return name in TRACKED_DATA_FILENAMES or name.endswith(".template")


def _should_prefer_legacy_file(legacy: Path, target: Path) -> bool:
    """Prefer the legacy file when it looks like the pre-migration live data."""
    if _is_tracked_data_file(legacy.name):
        return False
    try:
        legacy_stat = legacy.stat()
        target_stat = target.stat()
    except OSError:
        return True
    if legacy_stat.st_size > target_stat.st_size:
        return True
    return legacy_stat.st_mtime > target_stat.st_mtime


def _merge_legacy_tree(legacy: Path, target: Path) -> list[str]:
    """Merge legacy runtime tree into an existing var/ target (recursive)."""
    actions: list[str] = []
    target.mkdir(parents=True, exist_ok=True)

    for item in list(legacy.iterdir()):
        dest = target / item.name
        if item.is_dir():
            actions.extend(_merge_legacy_tree(item, dest))
            try:
                item.rmdir()
            except OSError:
                pass
            continue

        if not dest.exists():
            shutil.move(str(item), str(dest))
            actions.append(f"merged {item.name}")
            continue

        if _is_tracked_data_file(item.name):
            item.unlink(missing_ok=True)
            continue

        if _should_prefer_legacy_file(item, dest):
            dest.unlink(missing_ok=True)
            shutil.move(str(item), str(dest))
            actions.append(f"replaced {item.name} from legacy")
        else:
            item.unlink(missing_ok=True)

    return actions


def _remove_empty_dir(path: Path) -> bool:
    try:
        path.rmdir()
        return True
    except OSError:
        return False


def migrate_legacy_runtime_dirs(root: str = RUNTIME_ROOT_NAME) -> list[str]:
    """Move or merge legacy top-level runtime dirs into var/."""
    moved: list[str] = []
    base = project_root()
    rt = base / root
    rt.mkdir(parents=True, exist_ok=True)

    # Phase 4: top-level documents/ → var/user_files/ (not var/documents/)
    legacy_documents = base / "documents"
    target_user_files = rt / "user_files"
    if legacy_documents.exists() and legacy_documents.is_dir():
        if not target_user_files.exists():
            shutil.move(str(legacy_documents), str(target_user_files))
            moved.append(f"documents/ -> {root}/user_files/")
            logger.info("Migrated %s -> %s", legacy_documents, target_user_files)
        else:
            merged = _merge_legacy_tree(legacy_documents, target_user_files)
            if merged:
                moved.append(
                    f"documents/ merged into {root}/user_files/ ({len(merged)} items)"
                )
                logger.info(
                    "Merged legacy %s into %s (%d items)",
                    legacy_documents,
                    target_user_files,
                    len(merged),
                )
            if _remove_empty_dir(legacy_documents):
                moved.append("removed empty legacy documents/")
                logger.info("Removed empty legacy directory %s", legacy_documents)

    for sub in SUBDIRS:
        legacy = base / sub
        target = rt / sub
        if legacy.exists() and legacy.is_dir() and not target.exists():
            shutil.move(str(legacy), str(target))
            moved.append(f"{sub}/ -> {root}/{sub}/")
            logger.info("Migrated runtime directory %s -> %s", legacy, target)
        elif legacy.exists() and legacy.is_dir() and target.exists():
            merged = _merge_legacy_tree(legacy, target)
            if merged:
                moved.append(f"{sub}/ merged into {root}/{sub}/ ({len(merged)} items)")
                logger.info(
                    "Merged legacy %s into %s (%d items)", legacy, target, len(merged)
                )
            if _remove_empty_dir(legacy):
                moved.append(f"removed empty legacy {sub}/")
                logger.info("Removed empty legacy directory %s", legacy)
        elif not target.exists():
            target.mkdir(parents=True, exist_ok=True)

    # Phase 4: var/documents/ → var/user_files/
    old_var_documents = rt / "documents"
    if old_var_documents.exists() and old_var_documents.is_dir():
        target_user_files.mkdir(parents=True, exist_ok=True)
        merged = _merge_legacy_tree(old_var_documents, target_user_files)
        if merged:
            moved.append(
                f"{root}/documents/ merged into {root}/user_files/ ({len(merged)} items)"
            )
            logger.info(
                "Merged %s into %s (%d items)",
                old_var_documents,
                target_user_files,
                len(merged),
            )
        if _remove_empty_dir(old_var_documents):
            moved.append(f"removed empty {root}/documents/")
            logger.info("Removed empty directory %s", old_var_documents)

    return moved


def ensure_runtime_layout(root: str = RUNTIME_ROOT_NAME) -> list[str]:
    """Create var/ subdirs and migrate legacy top-level folders if needed."""
    return migrate_legacy_runtime_dirs(root)
