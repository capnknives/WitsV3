"""The one place that decides whether WITS may read, write or index a path.

Contract: docs/architecture/folder-access.md, sections 3, 4 and 6. ``decide()`` is pure
so the rules can be tested with Windows and POSIX path forms on any machine;
``check()`` resolves symlinks and ``..`` on the real filesystem first, loads live
grants, and raises ``PermissionError`` on refusal.
"""

from __future__ import annotations

import logging
import ntpath
import os
import posixpath
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import Any, Literal

from core import filesystem_policy as fp
from core.runtime_paths import inbox_dir, outbox_dir, projects_dir

Mode = Literal["read", "write", "index"]
GrantMode = Literal["read", "read_write", "index"]

audit_logger = logging.getLogger("WitsV3.PathAccess")

_GRANT_ALLOWS: dict[str, frozenset[str]] = {
    "read": frozenset({"read"}),
    "read_write": frozenset({"read", "write"}),
    "index": frozenset({"read", "index"}),
}
_NEVER_DIRS = frozenset({".ssh", ".gnupg", ".aws", ".azure", ".kube"})
_BROWSER_DIRS = frozenset({"user data", ".mozilla"})
_BROWSER_VENDORS = frozenset(
    {
        "mozilla",
        "firefox",
        "thunderbird",
        "google",
        "chrome",
        "chromium",
        "microsoft",
        "edge",
        "bravesoftware",
        "brave-browser",
        "opera software",
        "vivaldi",
    }
)
_NEVER_FILES = (
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*.pfx",
    "*.p12",
    "*.kdbx",
    "id_rsa*",
    "id_ed25519*",
    "client_secret*.json",
    "*credentials*.json",
    "token*.json",
    "*.keystore",
    "secrets.*",
)
_WINDOWS_TREES = frozenset({"windows", "program files", "program files (x86)"})


@dataclass(frozen=True)
class Grant:
    """A live or expired owner approval for one folder (rows arrive in F3)."""

    path: str
    mode: GrantMode
    expires_at: int
    revoked_at: int | None = None


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str
    requestable: bool
    resolved: PurePath


@dataclass(frozen=True)
class Roots:
    """Folder locations ``decide()`` needs, passed in so it never touches the disk."""

    inbox: str
    outbox: str
    projects: str
    repo: str
    home: str = ""
    windows: bool = False
    role_roots: tuple[str, ...] = ()


async def _no_grants() -> Sequence[Grant]:
    return ()


# Reason: F3 swaps this for the folder_grants table; until then nothing is granted.
grants_provider: Callable[[], Awaitable[Sequence[Grant]]] = _no_grants


def _pure(path: str | PurePath, windows: bool) -> PurePath:
    # Reason: normpath collapses ".." lexically so decide() stays safe even on raw input.
    if windows:
        return PureWindowsPath(ntpath.normpath(str(path)))
    return PurePosixPath(posixpath.normpath(str(path)))


def _key(path: PurePath, windows: bool) -> tuple[str, ...]:
    return tuple(p.lower() for p in path.parts) if windows else tuple(path.parts)


def _within(path: PurePath, root: str, windows: bool) -> bool:
    if not root:
        return False
    root_key = _key(_pure(root, windows), windows)
    return _key(path, windows)[: len(root_key)] == root_key


def _system_reason(path: PurePath, roots: Roots) -> str | None:
    """Drive roots, OS trees, profile and home roots (section 4, first bullet)."""
    w = roots.windows
    parts = path.parts
    if len(parts) <= 1:
        return "a drive or filesystem root"
    low = [p.lower() for p in parts[1:]]
    if roots.home:
        home = _pure(roots.home, w)
        if _key(path, w) == _key(home, w):
            return "the home folder root"
        if w and _within(path, str(home / "AppData"), w):
            return "the AppData folder"
    if w:
        if low[0] in _WINDOWS_TREES:
            return "a Windows system folder"
        if low[0] == "users" and len(low) == 2:
            return "a user profile root"
        if low[0] == "users" and len(low) >= 3 and low[2] == "appdata":
            return "the AppData folder"
    elif (parts[1] in ("home", "Users") and len(parts) == 3) or parts[1:] == ("root",):
        return "the home folder root"
    return None


def never_reason(path: str | PurePath, mode: Mode, roots: Roots) -> str | None:
    """Why *path* is never allowed for *mode* (section 4), or None.

    Directory names match on every component and file names on the final one,
    both case-insensitively, so ``D:\\Downloads\\.ENV`` cannot dodge the rule.
    """
    p = _pure(path, roots.windows)
    system = _system_reason(p, roots)
    if system:
        return system
    names = [part.lower() for part in p.parts[1:]]
    for i, name in enumerate(names):
        if name in _NEVER_DIRS:
            return f"a protected folder ({name})"
        if name in _BROWSER_DIRS or (
            name == "profiles" and any(v in _BROWSER_VENDORS for v in names[:i])
        ):
            return "a browser profile folder"
        if name == ".git" and mode != "read":
            return "the .git folder (read only)"
    if names and any(fnmatchcase(names[-1], pattern) for pattern in _NEVER_FILES):
        return "a secrets file"
    return None


def decide(
    path: str | PurePath,
    mode: Mode,
    role: str,
    grants: Sequence[Grant],
    roots: Roots,
    now: int | None = None,
) -> Decision:
    """Apply section 3 rules in order; first match wins. Pure: no filesystem access."""
    w = roots.windows
    p = _pure(path, w)
    if not p.is_absolute():
        return Decision(False, "path must be absolute", False, p)
    reason = never_reason(p, mode, roots)
    if reason:
        return Decision(False, f"never allowed: {reason}", False, p)

    if (role or "owner").strip().lower() != "owner":
        # Reason: guests keep today's role read roots from filesystem_policy; they cannot ask.
        ok = mode == "read" and any(_within(p, r, w) for r in roots.role_roots)
        why = "inside the role's read folders" if ok else "outside the allowed read folders"
        return Decision(ok, f"{why} for role '{role}'", False, p)

    for root, modes, name in (
        (roots.inbox, ("read", "index"), "inbox"),
        (roots.outbox, ("read", "write"), "outbox"),
        (roots.projects, ("read", "write"), "projects"),
    ):
        if _within(p, root, w):
            ok = mode in modes
            return Decision(ok, f"{name} folder allows {' and '.join(modes)}", False, p)

    if _within(p, roots.repo, w):
        if mode == "read":
            return Decision(True, "inside the WITS repo", False, p)
        return Decision(False, "repo changes go through the verified-edit pipeline", False, p)

    current = int(time.time()) if now is None else now
    for grant in grants:
        live = grant.revoked_at is None and grant.expires_at > current
        if live and mode in _GRANT_ALLOWS.get(grant.mode, ()) and _within(p, grant.path, w):
            return Decision(True, f"owner grant ({grant.mode})", False, p)

    return Decision(False, "outside the WITS folders; the owner can approve access", True, p)


def _load_config_quietly() -> Any | None:
    try:
        from core.config import load_config

        return load_config()
    except Exception:  # Reason: a broken config must not widen access; defaults apply.
        return None


def roots_for(config: Any | None, role: str = "owner") -> Roots:
    """Build Roots from config (``folders.base``) and the live repo location."""
    base = fp._base_setting(config or _load_config_quietly())
    owner = (role or "owner").strip().lower() == "owner"
    role_roots = () if owner else tuple(str(r) for r in fp.read_roots_for_role(role, config))
    try:
        home = str(Path.home().resolve())
    except (RuntimeError, OSError):
        home = ""
    return Roots(
        inbox=str(inbox_dir(base).resolve()),
        outbox=str(outbox_dir(base).resolve()),
        projects=str(projects_dir(base).resolve()),
        repo=str(fp.project_read_root()),
        home=home,
        windows=os.name == "nt",
        role_roots=role_roots,
    )


def _resolve(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = fp.project_read_root() / candidate
    return candidate.resolve()


def resolve_write_path(file_path: str | Path, config: Any | None = None) -> Path:
    """Where write_file puts *file_path*: relative paths land in the projects folder."""
    candidate = Path(file_path).expanduser()
    if candidate.is_absolute():
        return candidate
    base = fp._base_setting(config or _load_config_quietly())
    return projects_dir(base) / candidate


def enforce(
    path: str | Path,
    mode: Mode,
    role: str = "owner",
    config: Any | None = None,
    grants: Sequence[Grant] = (),
) -> Path:
    """Resolve *path* on disk, decide, and raise PermissionError when refused."""
    resolved = _resolve(path)
    roots = roots_for(config, role)
    decision = decide(resolved, mode, role, grants, roots)
    if decision.allowed:
        return resolved
    if not any(_within(decision.resolved, r, roots.windows) for r in _base_roots(roots)):
        # Reason: F3 moves this to audit_log; until then the logger keeps the trail.
        audit_logger.warning(
            "refused %s %s role=%s: %s", mode, resolved, role or "owner", decision.reason
        )
    raise PermissionError(f"Refusing to {mode} {resolved}: {decision.reason}")


def _base_roots(roots: Roots) -> tuple[str, str, str]:
    return roots.inbox, roots.outbox, roots.projects


async def check(
    path: str | Path, mode: Mode, role: str = "owner", *, config: Any | None = None
) -> Path:
    """Async entry point for tools: loads live grants, then :func:`enforce`."""
    grants = await grants_provider()
    return enforce(path, mode, role=role, config=config, grants=grants)


class IndexGate:
    """Per-root and per-file index decisions for the document indexer."""

    def __init__(self, roots: Roots, grants: Sequence[Grant], corpus: Path | None) -> None:
        self.roots = roots
        self.grants = grants
        self.corpus = corpus.resolve() if corpus else None
        self.skipped = 0

    @classmethod
    async def load(cls, config: Any | None, corpus: Path | None = None) -> IndexGate:
        return cls(roots_for(config), await grants_provider(), corpus)

    def _in_corpus(self, path: Path) -> bool:
        return self.corpus is not None and path.is_relative_to(self.corpus)

    def allows_root(self, root: Path) -> bool:
        resolved = root.resolve()
        if self._in_corpus(resolved):
            return never_reason(resolved, "index", self.roots) is None
        return decide(resolved, "index", "owner", self.grants, self.roots).allowed

    def allows_file(self, path: Path) -> bool:
        """True when *path* may be indexed; refused files are counted in ``skipped``."""
        resolved = path.resolve()
        # Reason: the upload corpus (document_rag.documents_path) is filled by the owner's
        # own upload route, so only the never-list applies there; symlinks out of it do not.
        if self._in_corpus(resolved):
            ok = never_reason(resolved, "index", self.roots) is None
        else:
            ok = decide(resolved, "index", "owner", self.grants, self.roots).allowed
        if not ok:
            self.skipped += 1
        return ok
