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
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import Any, Literal

from core import filesystem_policy as fp
from core.runtime_paths import DEFAULT_BASE

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
_RESERVED_DEVICE = re.compile(r"^(con|prn|aux|nul|com[1-9]|lpt[1-9])(\..*)?$", re.IGNORECASE)
_warned_grants: set[str] = set()


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


# Reason: core.workspace.folder_access.activate() swaps in the live folder_grants reader at
# workspace start-up; with the workspace off nothing is granted. Synchronous callers of
# enforce() (filesystem_policy.resolve_allowed_read_path) never consult it and see no grants.
grants_provider: Callable[[], Awaitable[Sequence[Grant]]] = _no_grants


def _pure(path: str | PurePath, windows: bool) -> PurePath:
    # Reason: normpath collapses ".." lexically so decide() stays safe even on raw input.
    if windows:
        return PureWindowsPath(ntpath.normpath(str(path)))
    return PurePosixPath(posixpath.normpath(str(path)))


def _key(path: PurePath, windows: bool) -> tuple[str, ...]:
    return tuple(p.lower() for p in path.parts) if windows else tuple(path.parts)


def _within(path: PurePath, root: str, windows: bool) -> bool:
    if not isinstance(root, str) or not root.strip():
        return False
    normal = _pure(root, windows)
    # Reason: a relative root such as "." would otherwise match every path.
    if not normal.is_absolute():
        return False
    root_key = _key(normal, windows)
    return _key(path, windows)[: len(root_key)] == root_key


def alias_reason(path: str | PurePath, windows: bool) -> str | None:
    """Refuse path spellings that alias another file or reach outside local drives.

    Windows ignores trailing dots and spaces, treats ``name:stream`` as the same file
    and maps device names, so ``.env.``, ``.env::$DATA`` and ``.git.`` would dodge
    the name rules. UNC paths and the two device namespaces are refused outright.
    """
    text = str(path)
    if text.replace("/", "\\").startswith("\\\\"):
        return "a UNC or device-namespace path"
    if not windows:
        return None
    raw = PureWindowsPath(text)
    for i, part in enumerate(raw.parts):
        if i == 0 and raw.anchor:
            continue
        if part in (".", ".."):
            continue
        if ":" in part or part.endswith((".", " ")) or _RESERVED_DEVICE.match(part):
            return "a Windows alias name (stream, trailing dot or space, or device)"
    return None


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
    alias = alias_reason(path, roots.windows)
    if alias:
        return alias
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

    # Reason: only the exact string "owner" is the owner; "", None or "Owner" are guests.
    if role != "owner":
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
        # Reason: not requestable, since repo writes go through the verified-edit pipeline.
        return Decision(False, "repo changes go through the verified-edit pipeline", False, p)

    current = int(time.time()) if now is None else now
    for grant in grants:
        if (
            _live(grant, current)
            and mode in _GRANT_ALLOWS.get(grant.mode, ())
            and _within(p, grant.path, w)
        ):
            return Decision(True, f"owner grant ({grant.mode})", False, p)

    return Decision(False, "outside the WITS folders; the owner can approve access", True, p)


def _live(grant: Any, now: int) -> bool:
    """True for a well-formed, unexpired, unrevoked grant; malformed ones are ignored."""
    try:
        ok = (
            isinstance(grant.path, str)
            and grant.mode in _GRANT_ALLOWS
            and type(grant.expires_at) is int
            and (grant.revoked_at is None or type(grant.revoked_at) is int)
        )
    except AttributeError:
        ok = False
    if not ok:
        key = repr(grant)
        if key not in _warned_grants:
            _warned_grants.add(key)
            audit_logger.warning("ignoring malformed grant %s", key)
        return False
    return grant.revoked_at is None and grant.expires_at > now


def _load_config_quietly() -> Any | None:
    try:
        from core.config import load_config

        return load_config()
    except Exception:  # Reason: a broken config must not widen access; defaults apply.
        return None


def _anchor() -> Path:
    """Where relative paths and a relative ``folders.base`` resolve: never the cwd."""
    override = os.getenv("WITSV3_PROJECT_ROOT")
    return Path(override) if override else fp.project_read_root()


def _base_dir(config: Any | None) -> Path:
    base = Path(fp._base_setting(config or _load_config_quietly()) or DEFAULT_BASE).expanduser()
    return base if base.is_absolute() else _anchor() / base


def _role_roots(role: Any, config: Any | None) -> tuple[str, ...]:
    if role == "owner":
        return ()
    key = role.strip() if isinstance(role, str) and role.strip() else "guest"
    if key.lower() == "owner":
        key = "guest"
    return tuple(str(r) for r in fp.read_roots_for_role(key, config))


def roots_for(config: Any | None, role: str = "owner") -> Roots:
    """Build Roots from config (``folders.base``) and the live repo location."""
    base = _base_dir(config)
    role_roots = _role_roots(role, config)
    try:
        home = str(Path.home().resolve())
    except (RuntimeError, OSError):
        home = ""
    return Roots(
        inbox=str((base / "inbox").resolve()),
        outbox=str((base / "outbox").resolve()),
        projects=str((base / "projects").resolve()),
        repo=str(fp.project_read_root()),
        home=home,
        windows=os.name == "nt",
        role_roots=role_roots,
    )


def _resolve(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = _anchor() / candidate
    return candidate.resolve()


def resolve_write_path(file_path: str | Path, config: Any | None = None) -> Path:
    """Where write_file puts *file_path*: relative paths land in the projects folder."""
    candidate = Path(file_path).expanduser()
    if candidate.is_absolute():
        return candidate
    return _base_dir(config) / "projects" / candidate


def enforce(
    path: str | Path,
    mode: Mode,
    role: str = "owner",
    config: Any | None = None,
    grants: Sequence[Grant] = (),
) -> Path:
    """Resolve *path* on disk, decide, and raise PermissionError when refused.

    Raises only PermissionError: unusable input (NUL bytes, unknown ``~user``, OS
    errors, malformed grants) is refused rather than surfacing other exceptions.
    """
    try:
        roots = roots_for(config, role)
        # Reason: check the spelling as supplied, since resolve() on Windows strips
        # trailing dots and spaces that alias protected names.
        alias = alias_reason(str(path), roots.windows)
        resolved = _resolve(path)
        decision = (
            Decision(False, f"never allowed: {alias}", False, _pure(resolved, roots.windows))
            if alias
            else decide(resolved, mode, role, grants, roots)
        )
    except (ValueError, RuntimeError, OSError, TypeError) as err:
        audit_logger.warning("refused %s %r role=%r: unusable path (%s)", mode, path, role, err)
        raise PermissionError(f"Refusing to {mode} {path!r}: unusable path") from err
    if decision.allowed:
        return resolved
    if not any(_within(decision.resolved, r, roots.windows) for r in _base_roots(roots)):
        # Reason: F3 moves this to audit_log; until then the logger keeps the trail.
        audit_logger.warning("refused %s %s role=%r: %s", mode, resolved, role, decision.reason)
    raise PermissionError(f"Refusing to {mode} {resolved}: {decision.reason}")


def _base_roots(roots: Roots) -> tuple[str, str, str]:
    return roots.inbox, roots.outbox, roots.projects


async def check(
    path: str | Path, mode: Mode, role: str = "owner", *, config: Any | None = None
) -> Path:
    """Async entry point for tools: loads live grants, then :func:`enforce`."""
    grants = await _load_grants()
    return enforce(path, mode, role=role, config=config, grants=grants)


async def _load_grants() -> Sequence[Grant]:
    try:
        return tuple(await grants_provider())
    except Exception as err:  # Reason: a broken provider must narrow access, never widen it.
        audit_logger.warning("grants provider failed, using no grants: %s", err)
        return ()


class IndexGate:
    """Per-root and per-file index decisions for the document indexer."""

    def __init__(
        self,
        roots: Roots,
        grants: Sequence[Grant],
        corpus: Path | None,
        runtime_var: Path | None = None,
    ) -> None:
        self.roots = roots
        self.grants = grants
        resolved = corpus.resolve() if corpus else None
        var = runtime_var.resolve() if runtime_var else None
        # Reason: the never-list-only exception is for the upload corpus under the repo's
        # var/; a documents_path anywhere else is just another root with the full check.
        inside_var = resolved is not None and var is not None and resolved.is_relative_to(var)
        self.corpus = resolved if inside_var else None
        self.skipped = 0

    @classmethod
    async def load(cls, config: Any | None, corpus: Path | None = None) -> IndexGate:
        return cls(roots_for(config), await _load_grants(), corpus, _anchor() / "var")

    def _in_corpus(self, path: Path) -> bool:
        return self.corpus is not None and path.is_relative_to(self.corpus)

    def allows_root(self, root: Path) -> bool:
        try:
            resolved = root.resolve()
        except (OSError, RuntimeError, ValueError):
            return False
        if self._in_corpus(resolved):
            return never_reason(resolved, "index", self.roots) is None
        return decide(resolved, "index", "owner", self.grants, self.roots).allowed

    def allows_file(self, path: Path) -> bool:
        """True when *path* may be indexed; refused files are counted in ``skipped``."""
        try:
            resolved = path.resolve()
        except (OSError, RuntimeError, ValueError):
            self.skipped += 1
            return False
        # Reason: the upload corpus (document_rag.documents_path) is filled by the owner's
        # own upload route, so only the never-list applies there; symlinks out of it do not.
        if self._in_corpus(resolved):
            ok = never_reason(resolved, "index", self.roots) is None
        else:
            ok = decide(resolved, "index", "owner", self.grants, self.roots).allowed
        if not ok:
            self.skipped += 1
        return ok
