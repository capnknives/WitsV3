"""
Self-repair tools for WitsV3.

These give an agent (the self-repair agent, or the coding agent working on
an existing file) a real detect -> diagnose -> fix -> verify loop instead of
just talking about one:

  - diagnose_log_errors: find real tracebacks/errors in logs/witsv3.log
  - run_test_suite: run pytest and report pass/fail
  - apply_code_fix: write a candidate fix, verify it, commit on success or
    revert to the original bytes on failure (core/safe_code_editor.py)
  - restart_app: deliberately relaunch the process after a verified fix

Safety boundary: apply_code_fix can only ever touch files inside the project
directory (core.safe_code_editor.resolve_within_project), and it never
leaves a failed edit in place — see that module's docstring.
"""

from __future__ import annotations

import asyncio
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.base_tool import BaseTool
from core.safe_code_editor import (
    PROJECT_ROOT,
    apply_verified_edit,
    guess_related_tests,
    run_pytest,
)

logger = logging.getLogger("WitsV3.Tool.SelfRepair")

_TRACEBACK_RE = re.compile(
    r"Traceback \(most recent call last\):\n(?P<body>(?:.*\n)*?)"
    r"(?P<exc_line>\S[^\n]*(?:Error|Exception)[^\n]*)",
)
_FILE_LINE_RE = re.compile(r'File "(?P<file>[^"]+)", line (?P<line>\d+)')
_LEVEL_LINE_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2} [\d:,]+ - (?P<name>\S+) - (?P<level>ERROR|CRITICAL) - (?P<msg>.*)$"
)


def _relative_to_project(path_str: str) -> Optional[str]:
    """Return path relative to PROJECT_ROOT if it's inside the project and
    actually exists, else None (filters out stdlib/site-packages frames)."""
    try:
        p = Path(path_str).resolve()
        rel = p.relative_to(PROJECT_ROOT)
    except (ValueError, OSError):
        return None
    return rel.as_posix() if p.exists() else None


def parse_traceback_issues(log_text: str, max_issues: int) -> List[Dict[str, Any]]:
    """Extract distinct tracebacks (preferred, file/line-resolvable) and bare
    ERROR/CRITICAL lines (message-only, not auto-fixable) from log text.

    Not log-specific despite the name's origin: run_pytest() uses
    --tb=native specifically so a failing test's output is the same
    "Traceback (most recent call last): ... File \"...\", line N" shape as a
    logged runtime error, so this same parser covers both sources — see
    SelfRepairAgent's whole-codebase fallback (no file named, no log issues
    -> run the test suite and parse failures the same way).
    """
    issues: List[Dict[str, Any]] = []
    seen_messages = set()

    for match in _TRACEBACK_RE.finditer(log_text):
        exc_line = match.group("exc_line").strip()
        if exc_line in seen_messages:
            continue
        body = match.group("body")
        file_line = None
        for fm in _FILE_LINE_RE.finditer(body):
            rel = _relative_to_project(fm.group("file"))
            if rel:
                file_line = (rel, int(fm.group("line")))  # keep the last (innermost) project frame
        if file_line:
            issues.append({
                "actionable": True,
                "file": file_line[0],
                "line": file_line[1],
                "message": exc_line,
                "kind": "traceback",
            })
            seen_messages.add(exc_line)

    if len(issues) < max_issues:
        for line in log_text.splitlines():
            m = _LEVEL_LINE_RE.match(line.strip())
            if not m:
                continue
            msg = m.group("msg").strip()
            if msg in seen_messages or any(i["message"] == msg for i in issues):
                continue
            issues.append({
                "actionable": False,
                "file": None,
                "line": None,
                "message": msg,
                "kind": "log_line",
            })
            seen_messages.add(msg)
            if len(issues) >= max_issues * 3:  # keep a bounded pool before truncation below
                break

    # Actionable (file-resolvable) issues first, most recent within each group last-seen-wins already applied above.
    issues.sort(key=lambda i: not i["actionable"])
    return issues[:max_issues]


class DiagnoseLogErrorsTool(BaseTool):
    """Scan logs/witsv3.log for recent tracebacks and errors."""

    def __init__(self):
        super().__init__(
            name="diagnose_log_errors",
            description=(
                "Scan the trailing lines of logs/witsv3.log for real tracebacks and "
                "ERROR/CRITICAL entries. Returns distinct issues with the source file "
                "and line when resolvable from the traceback, so a fix can be targeted "
                "at the right place. Read-only — makes no changes."
            ),
        )
        self.log_path = PROJECT_ROOT / "logs" / "witsv3.log"

    async def execute(self, lines: int = 2000, max_issues: int = 5) -> Dict[str, Any]:
        if not self.log_path.exists():
            return {"success": True, "issues": [], "message": "No log file found — nothing to scan."}

        with open(self.log_path, "r", encoding="utf-8", errors="replace") as f:
            tail = f.readlines()[-max(1, int(lines)):]
        text = "".join(tail)

        issues = parse_traceback_issues(text, max(1, int(max_issues)))
        actionable = sum(1 for i in issues if i["actionable"])
        return {
            "success": True,
            "issues": issues,
            "count": len(issues),
            "message": (
                f"Found {len(issues)} issue(s) in the last {len(tail)} log lines, "
                f"{actionable} with a resolvable file/line."
            ),
        }

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "lines": {"type": "integer", "description": "Trailing log lines to scan", "default": 2000},
                "max_issues": {"type": "integer", "description": "Maximum distinct issues to return", "default": 5},
            },
        }


class RunTestSuiteTool(BaseTool):
    """Run pytest and report pass/fail with a bounded output tail."""

    def __init__(self):
        super().__init__(
            name="run_test_suite",
            description=(
                "Run pytest (optionally scoped to one path) and report whether it "
                "passed, plus a tail of the output. Use this to verify a fix, or to "
                "confirm a suspected bug by reproducing a failure."
            ),
        )

    async def execute(self, test_path: str = "", timeout: float = 120.0) -> Dict[str, Any]:
        paths = [test_path] if test_path else None
        passed, output = await run_pytest(paths, timeout=float(timeout))
        return {
            "success": True,
            "passed": passed,
            "output": output,
            "message": "All tests passed." if passed else "Tests failed — see output.",
        }

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "test_path": {"type": "string", "description": "Specific test file/dir to run (default: whole suite)", "default": ""},
                "timeout": {"type": "number", "description": "Timeout in seconds", "default": 120.0},
            },
        }


class ApplyCodeFixTool(BaseTool):
    """Apply a verified, revert-on-failure edit to a project file."""

    def __init__(self):
        super().__init__(
            name="apply_code_fix",
            description=(
                "Write new_content to file_path inside the project, then run pytest "
                "to verify it. If tests pass, the change is committed to git; if they "
                "fail, the file is restored to its exact original content and nothing "
                "is committed. Use this for every code edit — new files, bug fixes, or "
                "edits to WITS's own source — instead of write_file, since it's the "
                "only path that verifies before keeping a change."
            ),
        )

    async def execute(
        self,
        file_path: str,
        new_content: str,
        reason: str,
        test_path: str = "",
        commit: bool = True,
    ) -> Dict[str, Any]:
        test_paths = [test_path] if test_path else (guess_related_tests(file_path) or None)
        result = await apply_verified_edit(
            file_path, new_content, reason=reason, test_paths=test_paths, commit=commit
        )
        return {
            "success": result.success,
            "file_path": result.file_path,
            "message": result.message,
            "committed": result.committed,
            "commit_sha": result.commit_sha,
            "test_output": result.test_output[-1500:],
        }

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Project-relative path to write"},
                "new_content": {"type": "string", "description": "Full new file content"},
                "reason": {"type": "string", "description": "Short description of what/why, used as the commit message"},
                "test_path": {"type": "string", "description": "Specific test file/dir to verify with (default: best-guess, else whole suite)", "default": ""},
                "commit": {"type": "boolean", "description": "Commit to git on success", "default": True},
            },
            "required": ["file_path", "new_content", "reason"],
        }


def _relaunch(reason: str) -> None:
    logger.warning("Self-triggered restart: %s", reason or "no reason given")
    script_path = sys.argv[0]
    cmd = [sys.executable, script_path]
    cmd += [a for a in sys.argv[1:] if a != "--restart"]
    cmd.append("--restart")
    subprocess.Popen(cmd, cwd=str(PROJECT_ROOT))
    sys.exit(0)


class RestartAppTool(BaseTool):
    """Deliberately restart the running process to pick up a verified fix."""

    def __init__(self):
        super().__init__(
            name="restart_app",
            description=(
                "Restart the WitsV3 process so a verified code change takes effect. "
                "Only call this after apply_code_fix has reported success — never "
                "before verification. The restart is scheduled a couple seconds in "
                "the future so the current response can finish first."
            ),
        )

    async def execute(self, delay_seconds: float = 2.0, reason: str = "") -> Dict[str, Any]:
        delay = max(0.5, min(float(delay_seconds), 30.0))
        loop = asyncio.get_event_loop()
        loop.call_later(delay, _relaunch, reason)
        return {
            "success": True,
            "message": f"Restart scheduled in {delay:.0f}s ({reason or 'manual restart'}).",
        }

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "delay_seconds": {"type": "number", "description": "Seconds to wait before relaunching", "default": 2.0},
                "reason": {"type": "string", "description": "Why the restart is happening", "default": ""},
            },
        }
