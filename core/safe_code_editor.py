"""
Safe, verifiable code-editing pipeline shared by the coding agent and the
self-repair agent.

The core guarantee: a file on disk is never left in a worse state than it
started in. Every edit follows the same sequence:

    1. snapshot the original bytes (or note the file is new)
    2. write the candidate content
    3. run verification (pytest)
    4. on success: commit to git (best-effort — a rejected/failed commit
       still leaves the verified change in the working tree)
       on failure: restore the exact original bytes (or remove the new file)

Git is only ever touched *after* a verified success, so a failed attempt
never leaves stray git state, and a crash mid-write can't corrupt the repo
beyond the single file being edited.
"""

from __future__ import annotations

import asyncio
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("WitsV3.SafeCodeEditor")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Matches a project-relative-looking path ending in a common source
# extension, e.g. "agents/base_agent.py" or "core/config.py:42".
_FILE_MENTION_RE = re.compile(r"\b([\w./\\-]+\.(?:py|ts|js|json|yaml|yml))\b(?::(\d+))?")
_FENCE_RE = re.compile(r"```(?:\w+)?\s*\n?(.*?)```", re.DOTALL)


def extract_file_mention(text: str) -> tuple[str, int | None] | None:
    """Find the first existing-project-file mention in free text.

    Returns (relative_path, line_or_None) or None. Shared by the coding
    agent and the self-repair agent so both target "fix this file" requests
    the same way.
    """
    for match in _FILE_MENTION_RE.finditer(text):
        candidate = match.group(1)
        line = int(match.group(2)) if match.group(2) else None
        try:
            resolved = (PROJECT_ROOT / candidate).resolve()
            resolved.relative_to(PROJECT_ROOT)
        except (ValueError, OSError):
            continue
        if resolved.exists() and resolved.is_file():
            return Path(candidate).as_posix(), line
    return None


def extract_code_from_response(response: str) -> str:
    """Pull code out of a fenced code block if present, else return as-is."""
    from core.json_llm_parser import strip_think_blocks

    cleaned = strip_think_blocks(response or "")
    match = _FENCE_RE.search(cleaned)
    if match:
        return match.group(1).strip()
    if re.search(r"^\s*(import |from |def |class )", cleaned, re.MULTILINE):
        return cleaned.strip()
    return cleaned.strip()


def extract_search_replace_blocks(response: str) -> list[tuple[str, str]]:
    """Parse SEARCH/REPLACE hunks from an LLM repair response."""
    blocks: list[tuple[str, str]] = []
    pattern = re.compile(
        r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE",
        re.DOTALL,
    )
    for match in pattern.finditer(response or ""):
        old, new = match.group(1), match.group(2)
        if old:
            blocks.append((old, new))
    return blocks


def apply_search_replace(original: str, blocks: list[tuple[str, str]]) -> str | None:
    """Apply SEARCH/REPLACE blocks; return None if any block misses."""
    content = original
    for old, new in blocks:
        if old not in content:
            return None
        content = content.replace(old, new, 1)
    return content


PATCH_LINE_THRESHOLD = 150


@dataclass
class EditResult:
    success: bool
    file_path: str
    message: str
    test_output: str = ""
    committed: bool = False
    commit_sha: str | None = None


def resolve_within_project(file_path: str) -> Path:
    """Resolve file_path, refusing anything outside the project root.

    Raises PermissionError if the resolved path escapes PROJECT_ROOT — this
    is the one enforced boundary between "WITS can edit its own code" and
    "WITS can edit arbitrary files on the machine."
    """
    candidate = Path(file_path)
    resolved = (
        candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()
    )
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as err:
        raise PermissionError(
            f"Refusing to edit outside the project directory: {resolved}"
        ) from err
    return resolved


# Files pytest or Python load implicitly (a planted one runs during the very
# pytest verification step), hook and CI config, and interpreter/cache folders.
# Matched case-insensitively; the directories at any depth.
_IMPLICIT_LOAD_NAMES = frozenset(
    {
        "conftest.py",
        "pytest.ini",
        ".pytest.ini",
        "pytest.toml",
        ".pytest.toml",
        "tox.ini",
        "setup.cfg",
        "pyproject.toml",
        "sitecustomize.py",
        "usercustomize.py",
        ".pre-commit-config.yaml",
        # Settings files: an edit here could switch off the sandbox or the write gate.
        "config.yaml",
        "config.local.yaml",
    }
)
_IMPLICIT_LOAD_DIRS = frozenset({".github", ".githooks", ".husky", ".venv", "venv", "__pycache__"})
_RUNTIME_DATA_DIR = "var"


def _refuse_aliases(file_path: str, resolved_rel: Path) -> None:
    """Refuse Windows alias spellings (``name:stream``, trailing dot or space,
    device names) in the requested path and in the resolved relative path.

    Checked with Windows rules on every OS: the owner's PC is Windows, where
    ``conftest.py.`` and ``conftest.py::$DATA`` name the file ``conftest.py``.
    """
    from core.path_access import alias_reason

    for text in (file_path, str(resolved_rel)):
        reason = alias_reason(text, windows=True)
        if reason:
            raise PermissionError(f"Refusing to edit {reason}: {file_path!r}")


def resolve_writable_within_project(file_path: str) -> Path:
    """resolve_within_project plus the write rules of the verified-edit pipeline.

    Also refuses (PermissionError) Windows alias spellings, anything the
    folder-access never-list refuses for write (secrets, keys, .git), anything
    under var/ (runtime data), files pytest/Python load implicitly (conftest.py
    at any depth, pytest/tox/setup config, site/usercustomize.py, *.pth), hook
    config, and anything under .github/, .githooks/, .husky/, .venv/, venv/ or
    __pycache__/ at any depth.
    """
    from core.path_access import is_never

    resolved = resolve_within_project(file_path)
    try:
        rel = resolved.relative_to(PROJECT_ROOT)
    except ValueError as err:
        raise PermissionError(f"Refusing to edit outside the project: {resolved}") from err
    _refuse_aliases(file_path, rel)
    parts = [part.lower() for part in rel.parts]
    name = parts[-1] if parts else ""
    if not parts or is_never(resolved, "write"):
        reason = "a protected or secrets path"
    elif parts[0] == _RUNTIME_DATA_DIR:
        reason = "runtime data under var/"
    elif (
        name in _IMPLICIT_LOAD_NAMES
        or name.endswith(".pth")
        or any(part in _IMPLICIT_LOAD_DIRS for part in parts)
    ):
        reason = "a settings file or one that pytest, Python, git hooks or CI load implicitly"
    else:
        return resolved
    raise PermissionError(f"Refusing to edit {reason}: {resolved}")


def resolve_within(base: Path, file_path: str) -> Path:
    """Resolve project-relative file_path for a write that must stay inside base.

    Raises PermissionError when the result leaves base, uses a Windows alias
    spelling, or is refused for write by the folder-access never-list.
    """
    from core.path_access import is_never

    resolved = resolve_within_project(file_path)
    base_resolved = base.resolve()
    try:
        inner = resolved.relative_to(base_resolved)
    except ValueError as err:
        raise PermissionError(f"Refusing to write outside {base_resolved}: {resolved}") from err
    _refuse_aliases(file_path, inner)
    if not inner.parts or is_never(resolved, "write"):
        raise PermissionError(f"Refusing to write a protected path: {resolved}")
    return resolved


async def run_pytest(
    test_paths: list[str] | None = None, timeout: float = 120.0
) -> tuple[bool, str]:
    """Run pytest as a trusted subprocess (this is orchestration code, not
    arbitrary agent-authored code — it is not run through the sandboxed
    python_execute tool). Returns (passed, tail_of_combined_output).

    Uses --tb=native so failures print as standard Python tracebacks
    ("Traceback (most recent call last): ... File \"...\", line N") instead
    of pytest's own assertion-rewrite style — that's the same format
    tools.self_repair_tools.parse_traceback_issues() already parses from
    var/logs/witsv3.log, so a failing test can feed the same issue-extraction
    path as a logged runtime error.
    """
    args = [sys.executable, "-m", "pytest", "-q", "-o", "addopts=", "--tb=native"]
    args += list(test_paths or [])
    proc = await asyncio.create_subprocess_exec(
        *args,
        cwd=str(PROJECT_ROOT),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return False, f"pytest timed out after {timeout:.0f}s ({' '.join(args)})"
    output = stdout.decode(errors="replace")
    return proc.returncode == 0, output[-4000:]  # bound size for LLM/UI consumption


async def run_py_compile(file_path: Path) -> tuple[bool, str]:
    """Quick syntax check for a single Python file (cheaper than pytest for
    e.g. freshly-scaffolded project files that have no tests yet)."""
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "py_compile",
        str(file_path),
        cwd=str(PROJECT_ROOT),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    stdout, _ = await proc.communicate()
    return proc.returncode == 0, stdout.decode(errors="replace")


async def _git(*args: str) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        "git",
        *args,
        cwd=str(PROJECT_ROOT),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    out, _ = await proc.communicate()
    return proc.returncode, out.decode(errors="replace")


async def apply_verified_edit(
    file_path: str,
    new_content: str,
    *,
    reason: str,
    test_paths: list[str] | None = None,
    timeout: float = 120.0,
    commit: bool = True,
) -> EditResult:
    """Write new_content to file_path, verify with pytest, and either commit
    (on success) or restore the original file exactly (on failure).

    This is the one code path both the coding agent and the self-repair
    agent use to touch source files — new project scaffolding, bug fixes,
    and self-edits to WITS's own code all go through the same guarantee.
    """
    try:
        resolved = resolve_writable_within_project(file_path)
    except PermissionError as e:
        return EditResult(success=False, file_path=file_path, message=str(e))

    original_existed = resolved.exists()
    original_bytes = resolved.read_bytes() if original_existed else None

    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(new_content, encoding="utf-8")

    from core.config import load_config
    from core.sandbox_runner import run_pytest_sandboxed, sandbox_mode

    config = load_config()
    if sandbox_mode(config) == "docker":
        passed, test_output = await run_pytest_sandboxed(test_paths, config=config, timeout=timeout)
    else:
        passed, test_output = await run_pytest(test_paths, timeout=timeout)

    if not passed:
        if original_bytes is not None:
            resolved.write_bytes(original_bytes)
        else:
            resolved.unlink(missing_ok=True)
        logger.warning("Verified edit to %s FAILED tests — reverted. reason=%r", file_path, reason)
        return EditResult(
            success=False,
            file_path=str(resolved),
            message="Verification failed; change reverted to the original file.",
            test_output=test_output,
        )

    committed = False
    commit_sha: str | None = None
    if commit:
        rel = str(resolved.relative_to(PROJECT_ROOT))
        rc, add_out = await _git("add", rel)
        if rc == 0:
            rc, _commit_out = await _git(
                "commit",
                "-m",
                f"self-repair: {reason}\n\nVerified automated edit to {rel} "
                f"(tests passed before commit).\n\n"
                f"Co-Authored-By: WITS Self-Repair <noreply@wits.local>",
            )
            committed = rc == 0
            if committed:
                rc2, sha_out = await _git("rev-parse", "--short", "HEAD")
                commit_sha = sha_out.strip() if rc2 == 0 else None
        else:
            logger.warning("git add failed for %s: %s", rel, add_out)

    logger.info(
        "Verified edit to %s PASSED tests%s. reason=%r",
        file_path,
        " and committed " + (commit_sha or "") if committed else " (not committed)",
        reason,
    )
    return EditResult(
        success=True,
        file_path=str(resolved),
        message="Edit applied and verified."
        + (f" Committed as {commit_sha}." if committed else " Not committed."),
        test_output=test_output,
        committed=committed,
        commit_sha=commit_sha,
    )


async def apply_verified_patch(
    file_path: str,
    patch_response: str,
    *,
    reason: str,
    test_paths: list[str] | None = None,
    timeout: float = 120.0,
    commit: bool = True,
) -> EditResult:
    """Apply SEARCH/REPLACE hunks from patch_response, then verify with pytest."""
    try:
        resolved = resolve_writable_within_project(file_path)
    except PermissionError as e:
        return EditResult(success=False, file_path=file_path, message=str(e))

    if not resolved.exists():
        return EditResult(success=False, file_path=file_path, message="File does not exist.")

    original = resolved.read_text(encoding="utf-8", errors="replace")
    blocks = extract_search_replace_blocks(patch_response)
    if not blocks:
        return EditResult(
            success=False,
            file_path=file_path,
            message="No valid SEARCH/REPLACE blocks found in patch response.",
        )
    patched = apply_search_replace(original, blocks)
    if patched is None:
        return EditResult(
            success=False,
            file_path=file_path,
            message="Patch hunks did not match the current file content.",
        )
    return await apply_verified_edit(
        file_path,
        patched,
        reason=reason,
        test_paths=test_paths,
        timeout=timeout,
        commit=commit,
    )


def guess_related_tests(file_path: str) -> list[str]:
    """Best-effort guess at which pytest paths cover a given source file, so
    verification doesn't have to run the entire suite for every edit.

    Falls back to the whole suite when nothing obviously matches — a slower
    but safe default, since running too few tests risks a false "verified".
    """
    resolved = Path(file_path)
    stem = resolved.stem
    candidates = [
        f"tests/{resolved.parent.name}/test_{stem}.py",
        f"tests/test_{stem}.py",
    ]
    existing = [c for c in candidates if (PROJECT_ROOT / c).exists()]
    return existing or []
