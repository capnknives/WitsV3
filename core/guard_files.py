"""Files the verified-edit pipeline must never change: WITS's own safety code (RT-12).

``apply_code_fix`` edits project files and then runs pytest, so an edit to the code that
enforces the edit rules, python execution, roles or the write gate would let the agent
switch its own guards off. This module only decides; core/safe_code_editor.py raises.

Matching is on the normalised path relative to the project root, case-insensitive, with
backslashes read as separators, and 8.3-looking parts (``SAFE_C~1.PY``) judged by text.
"""

from __future__ import annotations

import posixpath
import re
from pathlib import Path, PureWindowsPath

GUARD_FILES = frozenset(
    {
        "core/safe_code_editor.py",
        "core/path_access.py",
        "core/untrusted_content.py",
        "core/sandbox_runner.py",
        "core/docker_sandbox.py",
        "core/injection_guard.py",
        "core/tool_call_guard.py",
        "core/safe_calc.py",
        "core/guard_files.py",
        "core/guest_access.py",
        "core/filesystem_policy.py",
        "core/workspace/permissions.py",
        "agents/tool_tiers.py",
        "agents/turn_context.py",
        "agents/orchestrator_preflight.py",
        "agents/native_tool_orchestrator.py",
        "tools/python_execution_tool.py",
        "tools/self_repair_tools.py",
        "web/file_route_guard.py",
    }
)
GUARD_DIRS = frozenset({"tests/redteam"})
TEST_DIR = "tests"

GUARD_REFUSAL = "guard file: edits to WITS's own safety code are not allowed through apply_code_fix"
TEST_REFUSAL = (
    "test file: edits to test files need the docker sandbox (security.sandbox_mode: docker), "
    "because pytest would run the edited test code on this computer"
)

_SHORT_NAME = re.compile(r"~\d")


def _parts(text: str) -> tuple[str, ...]:
    """Lowercased, normalised path parts; empty when the text leaves the root."""
    norm = posixpath.normpath(text.replace("\\", "/").lower())
    parts = tuple(p for p in norm.split("/") if p not in ("", "."))
    return () if not parts or parts[0] == ".." else parts


def _candidates(file_path: str, resolved_rel: Path) -> list[tuple[str, ...]]:
    """The resolved relative path, plus the requested text when it is relative.

    Reason: on Linux ``core\\x.py`` resolves to one odd file name, and a symlinked folder
    resolves elsewhere, so both spellings are judged and either one refuses.
    """
    out = [_parts(str(resolved_rel))]
    text = str(file_path)
    if not PureWindowsPath(text).anchor and not text.startswith("/"):
        out.append(_parts(text))
    return [p for p in out if p]


def _part_matches(part: str, guarded: str) -> bool:
    """True when part names guarded, directly or as a Windows 8.3 short name.

    Reason: Windows builds ``SAFE_C~1.PY`` from the first six characters of the name
    without dots or spaces, so any ``~digit`` part whose prefix agrees with those is
    treated as the guarded name (over-refusal here only costs an odd spelling).
    """
    if part == guarded:
        return True
    if not _SHORT_NAME.search(part):
        return False
    prefix = part.split("~", 1)[0][:6]
    base = guarded.rsplit(".", 1)[0] if "." in guarded[1:] else guarded
    compact = base.replace(".", "").replace(" ", "")
    return bool(prefix) and compact[:6].startswith(prefix)


def _matches(parts: tuple[str, ...], guarded: str, *, prefix_only: bool) -> bool:
    want = guarded.split("/")
    if len(parts) < len(want) or (not prefix_only and len(parts) != len(want)):
        return False
    return all(_part_matches(p, g) for p, g in zip(parts[: len(want)], want, strict=True))


def is_guard_path(parts: tuple[str, ...]) -> bool:
    """True when normalised parts name a guard file or lie under a guarded directory."""
    return any(_matches(parts, g, prefix_only=False) for g in GUARD_FILES) or any(
        len(parts) > len(d.split("/")) and _matches(parts, d, prefix_only=True) for d in GUARD_DIRS
    )


def is_test_path(parts: tuple[str, ...]) -> bool:
    """True when normalised parts lie under the project's tests/ folder."""
    return len(parts) > 1 and _part_matches(parts[0], TEST_DIR)


def guard_reason(file_path: str, resolved_rel: Path) -> str | None:
    """Return the guard-file refusal message for this edit target, or None."""
    if any(is_guard_path(p) for p in _candidates(file_path, resolved_rel)):
        return GUARD_REFUSAL
    return None


def host_test_reason(file_path: str, resolved_rel: Path, mode: str) -> str | None:
    """Return the test-file refusal when the post-edit pytest would run on the host.

    Args:
        file_path: The path as requested.
        resolved_rel: The resolved path relative to the project root.
        mode: The sandbox mode from ``core.sandbox_runner.sandbox_mode``.
    """
    if str(mode).lower() == "docker":
        return None
    if any(is_test_path(p) for p in _candidates(file_path, resolved_rel)):
        return TEST_REFUSAL
    return None
