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
from functools import lru_cache
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import Any, Literal

from core import filesystem_policy as fp
from core import runtime_paths

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
    "id_ecdsa*",
    "id_dsa*",
    "credentials",
    "client_secret*.json",
    "*credentials*.json",
    "token*.json",
    "*.keystore",
    "secrets.*",
    ".netrc",
    "_netrc",
    ".pgpass",
    ".npmrc",
    ".pypirc",
    ".git-credentials",
    "*.ppk",
    "*.jks",
    "*.ovpn",
    "*service-account*.json",
    "*service_account*.json",
    "*.kdb",
    "wallet.dat",
    ".htpasswd",
)
# Reason: password lists, not source or policy docs (password_reset.py, password-policy.md).
_PASSWORD_NAME = re.compile(
    r"^passwords?([^.]*)?(\.(txt|csv|tsv|xls|xlsx|json|doc|docx|pdf|xml|yaml|yml))?$"
)
# Reason: WITS opens its own databases directly; no tool may read, write or index them.
_RUNTIME_PRIVATE = (("var", "data"), ("var", "secrets"))
_SNIFF_BYTES = 4096
_KEY_HEADER = re.compile(rb"-----BEGIN[^\r\n]*PRIVATE KEY|PuTTY-User-Key-File-")
_UTF16_BOMS = (b"\xff\xfe", b"\xfe\xff")
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
    documents: str = ""
    # Reason: <repo>/var is refused except the base folders and these (section 4).
    var_allow: tuple[str, ...] = ()
    # Reason: the configured data folder and workspace store, wherever they live.
    private: tuple[str, ...] = ()


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
    runtime = _runtime_reason(p, roots)
    if runtime:
        return runtime
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
    if names and (
        any(fnmatchcase(names[-1], pattern) for pattern in _NEVER_FILES)
        or _PASSWORD_NAME.match(names[-1])
    ):
        return "a secrets file"
    return None


def _runtime_reason(p: PurePath, roots: Roots) -> str | None:
    """WITS's own runtime files: all of ``<repo>/var`` but the allowed folders, and
    the configured data folder (``roots.private``)."""
    w = roots.windows
    if any(_within(p, d, w) for d in roots.private):
        return "WITS runtime data (configured data folder)"
    if not roots.repo:
        return None
    var = _pure(roots.repo, w) / "var"
    for sub in _RUNTIME_PRIVATE:
        if _within(p, str(var.parent.joinpath(*sub)), w):
            return f"WITS runtime data ({'/'.join(sub)})"
    allowed = (*_base_roots(roots), *roots.var_allow)
    if _within(p, str(var), w) and not any(_within(p, a, w) for a in allowed):
        return "WITS runtime data (var)"
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
        # Reason: only the owner's upload and delete routes change documents, never a tool.
        (roots.documents, ("read", "index"), "documents"),
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
    return runtime_paths.base_anchor()


def _base_dir(config: Any | None) -> Path:
    return runtime_paths.base_dir(fp._base_setting(config or _load_config_quietly()))


def _role_roots(role: Any, config: Any | None) -> tuple[str, ...]:
    if role == "owner":
        return ()
    key = role.strip() if isinstance(role, str) and role.strip() else "guest"
    if key.lower() == "owner":
        key = "guest"
    return tuple(str(r) for r in fp.read_roots_for_role(key, config))


def roots_for(config: Any | None, role: str = "owner") -> Roots:
    """Build Roots from config (``folders.base``) and the live repo location."""
    role_roots = _role_roots(role, config)
    config = config or _load_config_quietly()
    base = _base_dir(config)
    repo = _repo_root()
    return Roots(
        inbox=str((base / "inbox").resolve()),
        documents=str((base / "documents").resolve()),
        outbox=str((base / "outbox").resolve()),
        projects=str((base / "projects").resolve()),
        repo=repo,
        home=_home(),
        windows=os.name == "nt",
        role_roots=role_roots,
        # Reason: the guest export route writes to the runtime exports folder.
        var_allow=(str(base.resolve()), str(runtime_paths.exports_dir().resolve())),
        private=_private_dirs(config, repo, base.resolve()),
    )


@lru_cache(maxsize=8)
def _resolved(path: Path) -> str:
    return str(_resolve(path))


def _repo_root() -> str:
    """The repo root, resolved like the paths it is compared with (symlink, 8.3 name)."""
    return _resolved(fp.PROJECT_ROOT)


def _private_dirs(config: Any | None, repo: str, base: Path) -> tuple[str, ...]:
    """The configured data folder and workspace store, which may live outside the repo."""
    runtime = getattr(getattr(config, "runtime_paths", None), "root", None)
    found = [runtime_paths.data_dir(runtime if isinstance(runtime, str) else "var")]
    db = getattr(getattr(config, "workspace", None), "db_path", None)
    if isinstance(db, str) and db.strip():
        store = Path(db).expanduser()
        store = store if store.is_absolute() else runtime_paths.project_root() / store
        found.append(store.parent)
    out: list[str] = []
    for folder in found:
        try:
            resolved = folder.resolve()
        except (OSError, RuntimeError, ValueError):
            continue
        # Reason: a store sitting directly in the repo, base or home must not refuse all of it.
        wide = (Path(repo), base, Path(_home() or "/"))
        if any(w == resolved or w.is_relative_to(resolved) for w in wide):
            continue
        out.append(str(resolved))
    return tuple(out)


def _home() -> str:
    try:
        return str(Path.home().resolve())
    except (RuntimeError, OSError):
        return ""


def is_never(path: str | Path, mode: Mode) -> bool:
    """True when section 4 refuses *path* for *mode*, whatever the role or grants.

    For callers that keep their own folder rule but must still apply the never-list.
    Unusable input counts as never, so a caller cannot be tricked into allowing it.
    """
    windows = os.name == "nt"
    if alias_reason(str(path), windows):
        return True
    try:
        resolved = _resolve(path)
        roots = roots_for(None)
    except (ValueError, RuntimeError, OSError, TypeError):
        return True
    return never_reason(resolved, mode, roots) is not None


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


def _base_roots(roots: Roots) -> tuple[str, ...]:
    return roots.inbox, roots.documents, roots.outbox, roots.projects


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

    def __init__(self, roots: Roots, grants: Sequence[Grant]) -> None:
        self.roots = roots
        self.grants = grants
        self.skipped = 0

    @classmethod
    async def load(cls, config: Any | None) -> IndexGate:
        return cls(roots_for(config), await _load_grants())

    def allows_root(self, root: Path) -> bool:
        try:
            resolved = root.resolve()
        except (OSError, RuntimeError, ValueError):
            return False
        return decide(resolved, "index", "owner", self.grants, self.roots).allowed

    def allows_file(self, path: Path) -> bool:
        """True when *path* may be indexed; refused files are counted in ``skipped``."""
        try:
            resolved = path.resolve()
        except (OSError, RuntimeError, ValueError):
            self.skipped += 1
            return False
        ok = decide(resolved, "index", "owner", self.grants, self.roots).allowed
        if ok and has_private_key(resolved):
            ok = False
        if not ok:
            self.skipped += 1
        return ok


def has_private_key(path: Path) -> bool:
    """True when the first 4 KiB hold a private-key header.

    Unreadable or non-regular files count as True. UTF-16 text (PowerShell ``>``) is
    searched again with its NUL bytes removed.
    """
    try:
        if not path.is_file():
            return True
        with open(path, "rb") as handle:
            head = handle.read(_SNIFF_BYTES)
    except (OSError, ValueError):
        return True
    if _KEY_HEADER.search(head):
        return True
    if head.startswith(_UTF16_BOMS) or b"\x00" in head:
        return _KEY_HEADER.search(head.replace(b"\x00", b"")) is not None
    return False
