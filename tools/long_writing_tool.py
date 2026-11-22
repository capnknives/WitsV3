"""write_long_text: long-form writing driven by code, not by the chat loop (WP C5a).

The resident model has an 8,192-token context, so a long piece cannot be written part by
part inside one turn. This tool asks for an outline, then writes each section in a fresh
model call that sees only the brief, the outline and the end of the previous section, and
appends each section to ``<projects>/writing/<slug>.md`` as soon as it is written.
"""

from __future__ import annotations

import asyncio
import math
import re
import time
import unicodedata
from pathlib import Path
from typing import Any

import aiofiles

from core.base_tool import BaseTool
from core.llm_interface import LLMMessage
from core.tool_context import tool_user_role

WORDS_PER_SECTION = 450
MIN_SECTION_WORDS = 40
MIN_CALL_SECONDS = 5
TAIL_WORDS = 120
OPENING_WORDS = 60
BRIEF_CHARS = 2000
_MARKER_RE = re.compile(r"^\s*(?:[-*•]|\d+[.):]|#+)\s+")
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_SYSTEM = "You are a skilled writer. Write only what is asked, with no commentary."


def section_count(target_words: int, max_sections: int = 14) -> int:
    """Sections for a piece: ceil(target / 450), clamped to 1..max_sections."""
    return max(1, min(max_sections, 14, math.ceil(target_words / WORDS_PER_SECTION)))


def output_cap(words: int) -> int:
    """num_predict for a section of ``words`` words: about 2.2 tokens a word, 300..1800."""
    return max(300, min(1800, round(words * 2.2)))


def slugify(title: str) -> str:
    """Lowercase ascii words joined by hyphens, at most 60 chars; never empty."""
    text = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", text).strip("-")[:60].strip("-")
    return slug or "untitled"


def parse_outline(text: str, wanted: int, title: str) -> list[str]:
    """Section lines from the model's outline, at most ``wanted``.

    A line counts when it starts with a bullet or number, or reads "heading: summary".
    Preamble such as "Here is the outline:" is skipped. When nothing parses, ``wanted``
    generic sections named from the title are returned instead.
    """
    lines: list[str] = []
    for raw in _THINK_RE.sub("", text or "").splitlines():
        marked = _MARKER_RE.match(raw)
        line = (raw[marked.end() :] if marked else raw).replace("**", "").strip()
        head, sep, rest = line.partition(":")
        if line and (marked or (sep and head.strip() and rest.strip())):
            lines.append(line)
    if not lines:
        return [f"Part {i} of {title}: continue the piece" for i in range(1, wanted + 1)]
    return lines[:wanted]


def count_words(text: str) -> int:
    """Words in ``text``, split on whitespace."""
    return len(text.split())


def _clean(text: str) -> str:
    return _THINK_RE.sub("", text or "").strip()


class LongWritingTool(BaseTool):
    """Write a long piece to a file, one fresh model call per section."""

    def __init__(self) -> None:
        super().__init__(
            name="write_long_text",
            description=(
                "Write a long piece (story, guide, article, report) of 600 to 6,000 words to a "
                "file, section by section. Use it whenever the owner asks for more words than "
                "fit in one reply. Not for short text (answer in the reply) or code (write_file)."
            ),
        )
        self.config: Any = None
        self.llm_interface: Any = None
        # Reason: tests replace the clock to exercise the time limit without waiting.
        self.clock = time.monotonic
        self._deadline = 0.0

    def set_dependencies(self, config, llm_interface=None, memory_manager=None, **kwargs) -> None:
        self.config = config
        self.llm_interface = llm_interface

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "minLength": 3,
                    "maxLength": 120,
                    "description": "short title; becomes the file name",
                },
                "brief": {
                    "type": "string",
                    "minLength": 1,
                    "description": "the owner's request in full: topic, tone, must-haves",
                },
                "target_words": {
                    "type": "integer",
                    "minimum": 300,
                    "maximum": 6000,
                    "description": "words wanted, 300 to 6000; default 1200",
                },
            },
            "required": ["title", "brief"],
        }

    async def execute(
        self, title: str = "", brief: str = "", target_words: int = 1200, **kwargs: Any
    ) -> dict[str, Any]:
        if tool_user_role(kwargs.get("user_role", "owner")) == "guest":
            return {"success": False, "error": "write_long_text is only available to the owner."}
        title, brief = (title or "").strip()[:120], (brief or "").strip()[:BRIEF_CHARS]
        if len(title) < 3 or not brief or self.llm_interface is None:
            reason = "the writer is not ready" if brief and title else "a title and brief"
            return {"success": False, "error": f"Cannot write: needs {reason}."}
        target = max(300, min(6000, int(target_words or 1200)))
        config = self.config or _load_config()
        try:
            path, rel = await _new_file(slugify(title), config)
        except PermissionError as err:
            return {"success": False, "error": str(err)}
        return await self._write(title, brief, target, path, rel, config)

    async def _write(
        self, title: str, brief: str, target: int, path: Path, rel: str, config: Any
    ) -> dict[str, Any]:
        limits = config.conversation.long_writing
        self._deadline = self.clock() + limits.max_seconds
        wanted = section_count(target, limits.max_sections)
        try:
            planned = await self._outline(title, brief, wanted, config)
        except asyncio.TimeoutError:
            planned = ""
        outline = parse_outline(planned, wanted, title)
        per_section = max(1, round(target / len(outline)))
        written: list[str] = []
        timed_out = False
        for idx in range(len(outline)):
            if self.clock() >= self._deadline:
                timed_out = True
                break
            args = (title, brief, outline, idx, written, per_section, config)
            try:
                text = await self._section(*args)
                if count_words(text) < MIN_SECTION_WORDS:
                    text = await self._section(*args)
            except asyncio.TimeoutError:
                # Reason: a section cut off by the time limit is failed, not retried.
                timed_out = True
                break
            if count_words(text) < MIN_SECTION_WORDS:
                continue
            head = f"# {title}\n\n" if not written else "\n\n"
            async with aiofiles.open(path, "a", encoding="utf-8") as f:
                await f.write(head + text)
            written.append(text)
        if not written:
            path.unlink(missing_ok=True)
            why = "the time limit was reached" if timed_out else "the model returned no usable text"
            return {"success": False, "error": f"Nothing was written: {why}."}
        body = "\n\n".join(written)
        words = count_words(body)
        return {
            "success": True,
            "file": rel,
            "words": words,
            "sections": len(written),
            "target_words": target,
            "complete": len(written) == len(outline) and words >= 0.8 * target,
            "opening": " ".join(body.split()[:OPENING_WORDS]),
        }

    async def _call(self, prompt: str, words: int, config: Any) -> str:
        from agents.native_loop_support import resident_model

        # Reason: max_seconds also bounds a single slow call, with a small floor so a call
        # started just before the limit still has a chance to answer.
        remaining = max(MIN_CALL_SECONDS, self._deadline - self.clock())
        reply = await asyncio.wait_for(
            self.llm_interface.generate_chat(
                [
                    LLMMessage(role="system", content=_SYSTEM),
                    LLMMessage(role="user", content=prompt),
                ],
                model=resident_model(config),
                num_ctx=config.ollama_settings.num_ctx,
                think=False,
                num_predict=output_cap(words),
            ),
            remaining,
        )
        return _clean(getattr(reply, "content", "") or "")

    async def _outline(self, title: str, brief: str, wanted: int, config: Any) -> str:
        prompt = (
            f"Plan a piece titled '{title}'.\nBrief: {brief}\n\n"
            f"Write exactly {wanted} numbered lines, one per section, in reading order, each "
            "as 'heading: one-sentence summary'. Output only the lines."
        )
        return await self._call(prompt, 40 * wanted, config)

    async def _section(
        self,
        title: str,
        brief: str,
        outline: list[str],
        idx: int,
        written: list[str],
        words: int,
        config: Any,
    ) -> str:
        plan = "\n".join(f"{i + 1}. {line}" for i, line in enumerate(outline))
        done = "\n".join(f"- {outline[i]}" for i in range(min(idx, len(outline))))
        parts = [f"You are writing '{title}'.\nBrief: {brief}\n\nOutline:\n{plan}"]
        if done:
            parts.append(f"Sections already written (summaries):\n{done}")
        if written:
            tail = " ".join(written[-1].split()[-TAIL_WORDS:])
            parts.append(f"The previous section ended:\n...{tail}")
        ending = (
            "This is the last section: bring the piece to its end."
            if idx == len(outline) - 1
            else "Do not end the piece; later sections follow."
        )
        parts.append(
            f"Write section {idx + 1} of {len(outline)} only ({outline[idx]}), about {words} "
            "words. Prose only, continuing smoothly; no headings unless this kind of text "
            f"needs them, no title, no commentary. {ending}"
        )
        return await self._call("\n\n".join(parts), words, config)


def _load_config() -> Any:
    from core.config import load_config

    return load_config()


async def _new_file(slug: str, config: Any) -> tuple[Path, str]:
    """Reserve a new file under <projects>/writing; never an existing one.

    Raises:
        PermissionError: The path fails the write check write_file uses.
    """
    from core.path_access import check, resolve_write_path

    folder = resolve_write_path("writing", config).resolve()
    for n in range(1, 1000):
        rel = f"writing/{slug}{'' if n == 1 else f'-{n}'}.md"
        path = await check(resolve_write_path(rel, config), "write", config=config)
        if path.parent != folder:
            raise PermissionError(f"Refusing to write {rel}: outside the writing folder")
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.open("x", encoding="utf-8").close()
        except FileExistsError:
            continue
        return path, rel
    raise PermissionError("Refusing to write: too many files with this title")
