"""Self-repair agent for WitsV3.

Real detect -> diagnose -> fix -> verify -> (optionally) restart loop:

  1. Detect: if the request names a specific existing project file, target
     that; otherwise scan logs/witsv3.log for recent tracebacks/errors via
     the diagnose_log_errors tool.
  2. Diagnose: read the target file and ask the LLM for a full corrected
     version.
  3. Fix + verify: apply_code_fix writes the candidate, runs pytest, and
     only commits if tests pass — a failed attempt is reverted to the
     original bytes automatically (core/safe_code_editor.py).
  4. Restart: if a fix was verified and config.self_repair.restart_after_fix
     is on, trigger restart_app so the change takes effect.

Every step streams thinking/action/observation/result so the web UI shows
real progress instead of a single opaque response.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator
from typing import Any

from core.config import WitsV3Config
from core.llm_interface import BaseLLMInterface
from core.memory_manager import MemoryManager
from core.safe_code_editor import (
    PROJECT_ROOT,
    extract_code_from_response,
    extract_file_mention,
)
from core.schemas import ConversationHistory, StreamData

from .base_agent import BaseAgent

__all__ = [
    "SelfRepairAgent",
    "extract_file_mention",
    "extract_code_from_response",
]


class SelfRepairAgent(BaseAgent):
    """Diagnoses and fixes real issues (log errors or a named file), always
    through the verify-before-commit pipeline in core/safe_code_editor.py."""

    def __init__(
        self,
        agent_name: str,
        config: WitsV3Config,
        llm_interface: BaseLLMInterface,
        memory_manager: MemoryManager | None = None,
        tool_registry: Any | None = None,
        **_: Any,
    ) -> None:
        super().__init__(agent_name, config, llm_interface, memory_manager)
        self.tool_registry = tool_registry

    def _tool(self, name: str):
        if not self.tool_registry:
            return None
        return self.tool_registry.get_tool(name)

    async def run(
        self,
        user_input: str,
        conversation_history: ConversationHistory | None = None,
        session_id: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamData, None]:
        if session_id is None:
            session_id = str(uuid.uuid4())

        settings = self.config.self_repair
        if not settings.enabled:
            yield self.stream_result(
                "Self-repair is disabled (self_repair.enabled: false in config.yaml)."
            )
            return

        diagnose_tool = self._tool("diagnose_log_errors")
        fix_tool = self._tool("apply_code_fix")
        if self.tool_registry is None:
            # No tool registry at all (e.g. constructed standalone in a
            # minimal test) — fall back to a plain LLM response so the agent
            # still degrades gracefully instead of crashing.
            yield self.stream_thinking("Analyzing request")
            response = await self.generate_response(user_input)
            yield self.stream_result(response)
            return

        yield self.stream_thinking("Looking for a specific file to target...")
        mention = extract_file_mention(user_input)
        issues: list[dict[str, Any]]
        if mention:
            file_path, line = mention
            issues = [
                {
                    "actionable": True,
                    "file": file_path,
                    "line": line,
                    "message": user_input,
                    "kind": "user_request",
                }
            ]
            yield self.stream_observation(f"Targeting {file_path} as requested.")
        elif diagnose_tool is not None:
            yield self.stream_thinking(
                "No specific file named — scanning logs/witsv3.log for recent errors..."
            )
            diag = await diagnose_tool.execute(
                lines=settings.log_scan_lines, max_issues=settings.max_issues_per_run
            )
            issues = [i for i in diag.get("issues", []) if i.get("actionable")]
            yield self.stream_observation(diag.get("message", "Scanned logs."))
        else:
            issues = []

        if not issues:
            # Neither a named file nor a resolvable log traceback — fall back
            # to running the real test suite and treating any failures as
            # issues. Without this, a vague "find bugs in the codebase"
            # request had no path to actually inspect the codebase at all.
            test_tool = self._tool("run_test_suite")
            if test_tool is not None:
                yield self.stream_thinking(
                    "No log errors either — running the test suite to look for real failures..."
                )
                test_result = await test_tool.execute(timeout=settings.test_timeout_seconds)
                if not test_result.get("passed", True):
                    from tools.self_repair_tools import parse_traceback_issues

                    issues = [
                        i
                        for i in parse_traceback_issues(
                            test_result.get("output", ""), settings.max_issues_per_run
                        )
                        if i.get("actionable")
                    ]
                    yield self.stream_observation(
                        f"Test suite failed — found {len(issues)} resolvable issue(s) in the failures."
                    )
                else:
                    yield self.stream_observation(
                        "Test suite passed — no failing tests to investigate."
                    )

        if not issues:
            yield self.stream_result(
                "No actionable issues found — no resolvable file/line in recent logs, "
                "no failing tests, and the request didn't name a specific file. "
                "Nothing to repair."
            )
            return

        if fix_tool is None:
            yield self.stream_result(
                "apply_code_fix tool is unavailable — cannot safely apply or verify a fix."
            )
            return

        fixed_any = False
        for issue in issues[: settings.max_issues_per_run]:
            file_path = issue["file"]
            async for item in self._repair_one(file_path, issue, fix_tool):
                if item.type == "result" and "Repaired" in item.content:
                    fixed_any = True
                yield item

        if fixed_any and settings.restart_after_fix:
            restart_tool = self._tool("restart_app")
            if restart_tool:
                yield self.stream_action("Restarting to pick up the verified fix...")
                result = await restart_tool.execute(reason="verified self-repair fix")
                yield self.stream_observation(result.get("message", "Restart scheduled."))

    async def _repair_one(
        self, file_path: str, issue: dict[str, Any], fix_tool: Any
    ) -> AsyncGenerator[StreamData, None]:
        full_path = PROJECT_ROOT / file_path
        if not full_path.exists():
            yield self.stream_observation(f"{file_path} does not exist — skipping.")
            return

        yield self.stream_action(f"Reading {file_path} to diagnose: {issue['message'][:200]}")
        original = full_path.read_text(encoding="utf-8", errors="replace")

        prompt = (
            "You are fixing a real bug in a Python project. Output ONLY the full, "
            "corrected file content inside a single ```python code fence — no prose, "
            "no explanation, no partial snippets. Preserve everything that isn't "
            "related to the bug.\n\n"
            f"File: {file_path}\n"
            f"Reported problem: {issue['message']}\n\n"
            f"Current file content:\n```python\n{original}\n```\n"
        )
        proposed = await self.generate_response(prompt, temperature=0.2, max_tokens=4000)
        new_content = extract_code_from_response(proposed)

        if new_content.strip() == original.strip():
            yield self.stream_observation(
                "Proposed fix is identical to the current file — nothing to apply."
            )
            return

        yield self.stream_action(
            f"Applying candidate fix to {file_path} and verifying with tests..."
        )
        result = await fix_tool.execute(
            file_path=file_path, new_content=new_content, reason=issue["message"][:120]
        )

        if result["success"]:
            note = (
                f"committed as {result['commit_sha']}"
                if result.get("committed")
                else "applied but not committed"
            )
            yield self.stream_observation(f"Fix verified — tests passed, {note}.")
            yield self.stream_result(f"Repaired {file_path}: {issue['message'][:200]}")
            await self.store_memory(
                f"Self-repaired {file_path}: {issue['message'][:200]}",
                segment_type="SELF_REPAIR",
                importance=0.7,
                metadata={"file": file_path, "commit_sha": result.get("commit_sha")},
            )
        else:
            tail = result.get("test_output", "")[-500:]
            yield self.stream_observation(
                f"Fix failed verification — reverted to the original file.\n{tail}"
            )
            yield self.stream_result(
                f"Could not safely repair {file_path} — the candidate fix failed tests "
                f"and was reverted. No changes were left in place."
            )
