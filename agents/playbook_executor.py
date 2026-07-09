"""Execute orchestrator playbooks (fixed tool sequences)."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from agents.base_agent import BaseAgent
from core.config import WitsV3Config
from core.llm_interface import BaseLLMInterface
from core.memory_manager import MemoryManager
from core.playbooks import extract_export_basename, get_playbook
from core.runtime_paths import exports_subpath
from core.schemas import ConversationHistory, StreamData


class PlaybookExecutor(BaseAgent):
    """Run a declarative playbook without the JSON ReAct loop."""

    def __init__(
        self,
        agent_name: str,
        config: WitsV3Config,
        llm_interface: BaseLLMInterface,
        memory_manager: MemoryManager | None = None,
        tool_registry: Any | None = None,
    ):
        super().__init__(agent_name, config, llm_interface, memory_manager)
        self.tool_registry = tool_registry

    async def run(
        self,
        user_input: str,
        conversation_history: ConversationHistory | None = None,
        session_id: str | None = None,
        **kwargs,
    ) -> AsyncGenerator[StreamData, None]:
        playbook_id = kwargs.get("playbook_id") or ""
        spec = get_playbook(playbook_id)
        if not spec:
            yield self.stream_error(f"Unknown playbook: {playbook_id}")
            return

        yield self.stream_thinking(f"Running playbook: {playbook_id}")
        last_result: Any = None
        export_path: str | None = None
        file_snippets: list[str] = []
        if any(s.get("args_from") == "export_path" for s in spec.get("steps", [])):
            basename = extract_export_basename(user_input) or "conversation_export.txt"
            export_path = exports_subpath(basename, self.config.runtime_paths.root)

        for step in spec.get("steps", []):
            tool_name = step.get("tool", "")
            args = dict(step.get("args") or {})
            args_from = step.get("args_from")

            if args_from == "export_path":
                if not export_path:
                    basename = extract_export_basename(user_input) or "conversation_export.txt"
                    export_path = exports_subpath(basename, self.config.runtime_paths.root)
                args = {"file_path": export_path, "content": ""}
            elif args_from == "search_query":
                args = {"query": user_input}

            if tool_name == "read_conversation_history" and conversation_history:
                args.setdefault("session_id", session_id)

            yield self.stream_action(f"Playbook step: {tool_name}")
            try:
                if not self.tool_registry:
                    raise RuntimeError("Tool registry not available")
                last_result = await self.tool_registry.execute_tool(tool_name, **args)
                if tool_name == "read_conversation_history" and export_path:
                    content = self._format_history(conversation_history)
                    write_result = await self.tool_registry.execute_tool(
                        "write_file",
                        file_path=export_path,
                        content=content,
                    )
                    last_result = write_result
                    break
            except Exception as e:
                yield self.stream_error(f"Playbook step {tool_name} failed: {e}")
                return

            obs = self._brief_observation(tool_name, last_result)
            yield self.stream_observation(obs)
            snippet = self._extract_content_snippet(tool_name, last_result, step)
            if snippet:
                file_snippets.append(snippet)

        message = await self._synthesize_message(
            spec, user_input, export_path=export_path, file_snippets=file_snippets, last_result=last_result
        )
        yield self.stream_result(message)

    async def _synthesize_message(
        self,
        spec: dict[str, Any],
        user_input: str,
        *,
        export_path: str | None,
        file_snippets: list[str],
        last_result: Any,
    ) -> str:
        template = spec.get("synthesis", "Playbook completed.")
        if export_path:
            message = template.format(path=export_path)
        else:
            message = template

        context = "\n\n---\n\n".join(file_snippets) if file_snippets else ""
        if not context and last_result and isinstance(last_result, dict):
            context = str(last_result.get("content", ""))[:1500]

        if spec.get("synthesis_llm") and context.strip():
            prompt = (
                f"{template}\n\n"
                f"User question: {user_input}\n\n"
                f"Tool results:\n{context[:12000]}\n\n"
                "Write a concise, accurate answer grounded only in the tool results above."
            )
            try:
                return await self.generate_response(prompt, max_tokens=1024, temperature=0.3)
            except Exception as e:
                self.logger.warning(f"Playbook LLM synthesis failed: {e}")

        if context.strip():
            message = f"{message}\n\n{context[:4000]}"
        return message

    @staticmethod
    def _extract_content_snippet(tool_name: str, result: Any, step: dict[str, Any]) -> str:
        if not isinstance(result, dict):
            return ""
        content = str(result.get("content", "") or "").strip()
        if not content:
            return ""
        if tool_name == "read_file":
            path = (step.get("args") or {}).get("file_path", "file")
            return f"### {path}\n{content[:2000]}"
        if tool_name == "document_search":
            return content[:3000]
        if tool_name == "list_directory":
            return content[:1500]
        return content[:1500]

    @staticmethod
    def _format_history(conversation_history: ConversationHistory | None) -> str:
        if not conversation_history:
            return ""
        lines = []
        for msg in conversation_history.messages:
            role = msg.role.upper()
            lines.append(f"[{role}] {msg.content}")
        return "\n\n".join(lines)

    @staticmethod
    def _brief_observation(tool_name: str, result: Any) -> str:
        if isinstance(result, dict):
            if result.get("error"):
                return f"{tool_name}: error — {result['error']}"
            if result.get("success") is False:
                return f"{tool_name}: failed"
        return f"{tool_name}: ok"
