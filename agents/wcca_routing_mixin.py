# agents/wcca_routing_mixin.py
"""Orchestrator routing helpers for WitsControlCenterAgent."""

import re
from pathlib import Path
from typing import Any


class OrchestratorRoutingMixin:
    """Document, web-search, and file-write routing for the control center."""

    async def _get_document_inventory(self) -> dict[str, int]:
        """File path -> chunk count for every ingested document (empty if none)."""
        if not self.memory_manager:
            return {}
        try:
            segments = await self.memory_manager.get_recent_memory(
                limit=1_000_000, filter_dict={"type": "DOCUMENT_CHUNK"}
            )
        except Exception as e:
            self.logger.warning(f"Could not list ingested documents: {e}")
            return {}
        counts: dict[str, int] = {}
        for seg in segments:
            fp = seg.metadata.get("file_path")
            if fp:
                counts[fp] = counts.get(fp, 0) + 1
        return counts

    @staticmethod
    def _documents_context(inventory: dict[str, int]) -> str:
        """Prompt block describing which user documents are searchable."""
        if not inventory:
            return "No user documents are currently ingested."
        listing = "\n".join(
            f"- {name} ({count} chunks)" for name, count in sorted(inventory.items())
        )
        return (
            "These user documents are ALREADY ingested and fully accessible via "
            "the document_search tool. Never claim you cannot access them or "
            "have no record of them:\n" + listing
        )

    # Phrases in user messages that imply document_search / file access.
    _DOCUMENT_TOOL_HINTS = (
        "document",
        "my notes",
        "my files",
        "search my",
        "in my memory",
        "remember",
        "ingest",
        "uploaded",
        "read the file",
        "look up",
        "the file",
        "attachment",
        "attached",
    )

    # Phrases that signal the user wants live/external info fetched. Kept
    # deliberately precise so ordinary chat ("how are you today") is NOT routed
    # to the slow orchestrator — "today"/"current" alone are too weak to trust.
    _WEB_SEARCH_SIGNALS = (
        # explicit "go find this online" commands
        "look up",
        "look it up",
        "look that up",
        "look this up",
        "look them up",
        "search for",
        "search the web",
        "web search",
        "search online",
        "search it up",
        "google it",
        "google for",
        "find out",
        "check online",
        "look online",
        "on the internet",
        "on the web",
        "browse the web",
        # real-time / recency signals a local model can't answer from memory
        "latest",
        "most recent",
        "breaking news",
        "in the news",
        "news about",
        "up to date",
        "up-to-date",
        "weather",
        "forecast",
        # common current-fact question patterns
        "who won",
        "who died",
        "who passed away",
        "who is the current",
        "who's the current",
        "what happened to",
        "price of",
        "stock price",
        "exchange rate",
        "score of",
        "release date",
        "when is the next",
        "when does the next",
    )

    def _needs_web_search(self, message: str) -> bool:
        """True if answering needs current/external info or an explicit lookup.

        Such queries must reach the orchestrator (which owns the web_search
        tool) rather than being answered directly from the model's training.
        """
        lowered = message.lower()
        if any(sig in lowered for sig in self._WEB_SEARCH_SIGNALS):
            return True
        # A recent/near-future year (>= 2024) in a question usually implies
        # information past the local model's training cutoff.
        has_question = "?" in message or bool(
            re.search(r"\b(who|what|when|where|which|whose|did|does|is|are|died|won)\b", lowered)
        )
        if has_question and re.search(r"\b(202[4-9]|20[3-9]\d)\b", lowered):
            return True
        return False

    def _doc_routing_hints(self, doc_inventory: dict[str, int]) -> set:
        """Filename and stem tokens that imply a document/tool request."""
        hints: set = set()
        for path in doc_inventory:
            name = Path(path).name.lower()
            hints.add(name)
            hints.update(w for w in re.split(r"[\W_]+", Path(path).stem.lower()) if len(w) >= 4)
        return hints

    # Phrases that signal saving/exporting content to disk.
    _FILE_SAVE_SIGNALS = (
        "save this conversation",
        "save our conversation",
        "save the conversation",
        "save to file",
        "save to a file",
        "save to disk",
        "write to file",
        "write it to",
        "export conversation",
        "log of our conversation",
        "save a log",
        "save this chat",
        "write the story",
        "save the story",
        "save as a file",
    )

    def _needs_file_write(self, message: str) -> bool:
        """True when the user wants content written to a file on disk."""
        lowered = message.lower()
        return any(sig in lowered for sig in self._FILE_SAVE_SIGNALS)

    # Phrases signaling an unambiguous, self-contained request to find/fix
    # real bugs — deliberately specific multi-word phrases (not bare "bug"/
    # "fix", which are too broad and would swallow unrelated chit-chat).
    # 2026-07-08: "find and fix any bugs in your code" was classified as a
    # clarification_question and never reached specialized-agent routing at
    # all (that gate only runs for intent_type == "direct_response"). This
    # signal short-circuits that misclassification the same way
    # _needs_web_search/_needs_file_write already do for their cases.
    _SELF_REPAIR_SIGNALS = (
        "bugs in your code",
        "bugs in the code",
        "bugs in your codebase",
        "bugs in the codebase",
        "bug in your code",
        "bug in the code",
        "fix any bugs",
        "find any bugs",
        "find bugs in",
        "find and fix",
        "search for bugs",
        "look for bugs",
        "scan for bugs",
        "fix bugs in",
        "repair your code",
        "repair your own code",
        "analyze your own code",
        "your own codebase",
    )

    def _needs_self_repair(self, message: str) -> bool:
        """True for an unambiguous "find/fix real bugs" request."""
        lowered = message.lower()
        return any(sig in lowered for sig in self._SELF_REPAIR_SIGNALS)

    # Owner review of family-tester / guest audit logs (routes to guest_audit_summary tool).
    _GUEST_AUDIT_SIGNALS = (
        "guest log",
        "guest logs",
        "guest audit",
        "guest activity",
        "family tester",
        "tester log",
        "tester logs",
        "what did tester",
        "what has tester",
        "summarize tester",
        "other user",
        "other users",
        "guest user",
        "who joined",
        "nephew",
        "nephews",
    )

    def _needs_guest_audit_review(self, message: str) -> bool:
        lowered = message.lower()
        return any(sig in lowered for sig in self._GUEST_AUDIT_SIGNALS)

    async def _requires_orchestrator_for_input(self, user_input: str) -> bool:
        """True when answering requires tools (ingested docs or live web search)."""
        if self._needs_guest_audit_review(user_input):
            return True
        if self._needs_file_write(user_input):
            return True
        doc_inventory = await self._get_document_inventory()
        lowered = user_input.lower()
        doc_hints = self._doc_routing_hints(doc_inventory)
        if any(h in lowered for h in self._DOCUMENT_TOOL_HINTS) or any(
            h in lowered for h in doc_hints
        ):
            return True
        return self._needs_web_search(user_input)

    def _normalize_parsed_intent(self, parsed: dict[str, Any]) -> dict[str, Any]:
        """Fill routing metadata so the handler does not rely on loose defaults."""
        intent_type = parsed.get("type", "goal_defined")
        if intent_type == "goal_defined":
            parsed.setdefault("complexity", "moderate")
            parsed["requires_tools"] = True
            parsed["suggested_response"] = "orchestrator"
        elif intent_type == "clarification_question":
            parsed.setdefault("complexity", "simple")
            parsed["requires_tools"] = False
            parsed["suggested_response"] = "clarification"
        elif intent_type == "direct_response":
            parsed.setdefault("complexity", "simple")
            parsed["requires_tools"] = False
            parsed["suggested_response"] = "direct"
        elif intent_type == "conversation":
            parsed.setdefault("complexity", "simple")
            parsed["requires_tools"] = False
            parsed["suggested_response"] = "direct"
        else:
            parsed.setdefault("complexity", "moderate")
            parsed.setdefault("requires_tools", False)
            parsed.setdefault("suggested_response", "direct")
        return parsed

    def _is_casual_conversation(self, message: str) -> bool:
        """
        Determine if a message is casual conversation.

        Args:
            message: The user's message

        Returns:
            True if it's casual conversation
        """
        lowered = message.lower()
        # Single words match on word boundaries only — a plain substring test
        # made "hi" match inside "things"/"this" and flagged real requests
        # as small talk.
        words = set(re.findall(r"[a-z']+", lowered))
        casual_words = {
            "hello",
            "hi",
            "hey",
            "thanks",
            "appreciate",
            "nice",
            "cool",
            "great",
            "bye",
            "goodbye",
        }
        casual_phrases = (
            "how are you",
            "what's up",
            "how's it going",
            "thank you",
            "see you",
            "talk to you",
        )

        # Short messages are usually casual
        if len(message.split()) < 6:
            return True

        # Check for casual indicators
        if words & casual_words or any(phrase in lowered for phrase in casual_phrases):
            return True

        # Short questions are usually casual
        question_words = {"what", "how", "why", "when", "where", "who"}
        if ("?" in message or words & question_words) and len(message.split()) < 10:
            return True

        # Longer messages are less likely to be casual
        return False
