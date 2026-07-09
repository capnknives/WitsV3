# agents/wcca_routing_mixin.py
"""Orchestrator routing helpers for WitsControlCenterAgent."""

import re
from pathlib import Path
from typing import Any

from core.schemas import ConversationHistory


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

    # Same short-circuit idea as _needs_self_repair, for story/book requests.
    # 2026-07-08 finding: "write the equivalent to a 100 page story, about a
    # knight..." was classified by the LLM intent step as ordinary
    # conversation/direct_response, which returns before specialized-agent
    # routing is ever considered — this forces delegation to book_writing
    # regardless of that classification, the same way _needs_self_repair
    # already does for bug-hunt requests.
    # A bare fiction noun isn't enough on its own — "what's the story with
    # this bug?" is ordinary chat, not a writing request — so this requires
    # an authoring verb to co-occur with a fiction noun anywhere in the
    # message.
    _STORY_NOUN_RE = re.compile(
        r"\b(story|novel|screenplay|fanfiction|fan fiction|short story|poem|tale)\b",
        re.IGNORECASE,
    )
    _AUTHORING_VERB_RE = re.compile(
        r"\b(write|writing|wrote|create|creating|compose|composing|tell|telling|"
        r"craft|crafting|pen|penning)\b",
        re.IGNORECASE,
    )

    def _needs_story_writing(self, message: str) -> bool:
        """True for an unambiguous creative-writing request."""
        return bool(self._STORY_NOUN_RE.search(message) and self._AUTHORING_VERB_RE.search(message))

    # Short follow-ups that mean "keep going with what we were just doing" —
    # meaningless on their own, but should resume whichever specialized agent
    # handled the previous turn in this session rather than being judged in
    # isolation (which was landing on casual chat or a fresh, unrelated
    # orchestrator run). Mirrors the _pending_clarifications merge pattern.
    _AGENT_CONTINUATION_RE = re.compile(
        r"^\s*(okay,?\s*|ok,?\s*|alright,?\s*|so\s+)*"
        r"(please\s+)?"
        r"(make it|do it|write it( all)?|go ahead|finish it|finish (it|the (story|book|chapter))|"
        r"continue( writing)?|keep (going|writing)|"
        r"write the (whole|entire|full) (story|book|thing)|"
        r"save (it|this)( now)?( to disk)?)\s*\.?\s*$",
        re.IGNORECASE,
    )

    def _is_agent_continuation_phrase(self, message: str) -> bool:
        """True for a short "keep going" reply with no new content of its own."""
        return bool(self._AGENT_CONTINUATION_RE.match(message.strip()))

    # Owner guest admin: audit summaries + account roster (guest_audit_summary / guest_accounts_list).
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

    _GUEST_PROFILE_SIGNALS = (
        "interested in",
        "what does",
        "what do we know about",
        "what do you know about",
        "what does the system know",
        "what does wits know",
        "what have we learned about",
        "what have you learned about",
        "guest profile",
        "user profile",
        "their hobbies",
        "their interests",
    )

    # Generic "ask about a person" phrasing. On its own this is too broad to
    # force guest-profile routing (e.g. "tell me about the weather"), so it
    # only counts when paired with an actual registered guest's name via
    # _message_mentions_guest_name — see _needs_guest_profile_review.
    _GUEST_PROFILE_PERSON_QUERY_SIGNALS = (
        "tell me about",
        "who is",
        "what is",
        "what's",
        "know about",
        "into",
    )

    _GUEST_ACCOUNTS_SIGNALS = (
        "list guest",
        "list guests",
        "list all guest",
        "list active guest",
        "guest accounts",
        "guest account",
        "active guest",
        "active guests",
        "registered guest",
        "registered guests",
        "who are the guests",
        "show me guests",
        "family testers",
        "tester accounts",
        "set age band",
        "age band",
        "as a teen",
        "as an adult",
        "as a child",
        "make guest",
        "set guest",
        "set sean",
        "protection tier",
    )

    def _needs_guest_audit_review(self, message: str) -> bool:
        lowered = message.lower()
        return any(sig in lowered for sig in self._GUEST_AUDIT_SIGNALS)

    def _needs_guest_accounts_list(self, message: str) -> bool:
        lowered = message.lower()
        return any(sig in lowered for sig in self._GUEST_ACCOUNTS_SIGNALS)

    def _message_mentions_guest_name(self, message: str) -> bool:
        from core.guest_access import GuestRegistry

        lowered = message.lower()
        for guest in GuestRegistry().list_active_guests():
            name = (guest.get("display_name") or "").strip().lower()
            if name and name in lowered:
                return True
        return False

    def _extract_guest_name_for_profile_query(self, message: str) -> str | None:
        from core.guest_access import GuestRegistry

        lowered = message.lower()
        matches: list[tuple[float, str]] = []
        for guest in GuestRegistry().list_active_guests():
            name = (guest.get("display_name") or "").strip()
            if name and name.lower() in lowered:
                matches.append((float(guest.get("last_seen") or 0), name))
        if matches:
            return max(matches, key=lambda x: x[0])[1]
        return None

    def _needs_guest_profile_review(self, message: str) -> bool:
        lowered = message.lower()
        mentions_known_guest = self._message_mentions_guest_name(message)

        # A message naming an actual registered guest plus any "ask about a
        # person" phrasing is unambiguous, regardless of which name it is —
        # don't require the name to be hardcoded into a phrase list.
        if mentions_known_guest and any(
            sig in lowered for sig in self._GUEST_PROFILE_PERSON_QUERY_SIGNALS
        ):
            return True

        if not any(sig in lowered for sig in self._GUEST_PROFILE_SIGNALS):
            return False
        if mentions_known_guest:
            return True
        return any(
            w in lowered
            for w in (
                "guest",
                "tester",
                "nephew",
                "family",
                "user",
                "profile",
                "interest",
                "hobbies",
            )
        )

    def _needs_guest_admin_review(self, message: str) -> bool:
        if self._needs_guest_profile_review(message):
            return True
        return self._needs_guest_audit_review(message) or self._needs_guest_accounts_list(message)

    # Owner asks about accumulated cross-session knowledge (recurring bugs,
    # durable project facts) — distinct from guest-profile signals above,
    # which require either a registered guest name or explicit guest/tester
    # wording, so there's no overlap with these project-level phrases.
    _KNOWLEDGE_LOG_SIGNALS = (
        "recurring bug",
        "recurring bugs",
        "recurring error",
        "recurring errors",
        "recurring issue",
        "recurring issues",
        "keeps happening",
        "keeps breaking",
        "keep happening",
        "keep breaking",
        "what bugs keep",
        "what do you know about this project",
        "what do you know about the project",
        "project facts",
        "accumulated knowledge",
    )

    def _needs_knowledge_log_review(self, message: str) -> bool:
        lowered = message.lower()
        return any(sig in lowered for sig in self._KNOWLEDGE_LOG_SIGNALS)

    async def _requires_orchestrator_for_input(self, user_input: str) -> bool:
        """True when answering requires tools (ingested docs or live web search)."""
        if self._needs_guest_admin_review(user_input):
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

    def _messages_before_current_turn(
        self, conversation_history: ConversationHistory | None
    ) -> list:
        """History excluding the in-flight user message (already appended in web/CLI)."""
        if not conversation_history or not conversation_history.messages:
            return []
        messages = conversation_history.messages
        if messages[-1].role == "user":
            return messages[:-1]
        return messages

    def _last_assistant_message(self, conversation_history: ConversationHistory | None) -> str:
        for msg in reversed(self._messages_before_current_turn(conversation_history)):
            if msg.role == "assistant":
                return msg.content or ""
        return ""

    def _last_user_message_before_current(
        self, conversation_history: ConversationHistory | None
    ) -> str:
        prior = self._messages_before_current_turn(conversation_history)
        for msg in reversed(prior):
            if msg.role == "user":
                return msg.content or ""
        return ""

    _GENERIC_ASSISTANT_QUESTIONS = (
        "how can i help",
        "how may i assist",
        "what can i do for you",
        "anything else i can",
        "how are you",
        "what would you like to work on",
    )

    def _assistant_message_awaited_reply(self, text: str) -> bool:
        """True when the assistant's last turn invited a task-shaping user answer."""
        t = (text or "").strip()
        if not t:
            return False
        lowered = t.lower()
        if any(marker in lowered for marker in self._GENERIC_ASSISTANT_QUESTIONS):
            return False
        if t.endswith("?"):
            return True
        markers = (
            "which one",
            "could you clarify",
            "please clarify",
            "let me know",
            "tell me which",
            "what would you like",
            "can you specify",
            "more details",
            "which file",
            "which report",
            "which document",
            "what do you mean",
            "look it up",
            "would you like me to",
        )
        return any(marker in lowered for marker in markers)

    @staticmethod
    def _is_short_follow_up_reply(message: str) -> bool:
        """Short reply that likely continues prior context instead of starting small talk."""
        stripped = (message or "").strip()
        if not stripped:
            return False
        words = stripped.split()
        if len(words) > 12:
            return False
        lowered = stripped.lower().rstrip("!.,")
        affirmatives = {
            "yes",
            "yeah",
            "yep",
            "yup",
            "sure",
            "ok",
            "okay",
            "no",
            "nope",
            "go ahead",
            "do it",
            "do that",
        }
        if lowered in affirmatives:
            return True
        referential_phrases = (
            "summarize it",
            "look it up",
            "that one",
            "this one",
            "the same",
            "the audit",
            "the report",
            "the file",
            "the first",
            "the second",
            "same one",
            "that report",
            "that file",
        )
        if any(phrase in lowered for phrase in referential_phrases):
            return True
        if len(words) <= 6 and any(token in lowered for token in ("it", "that", "this", "one")):
            return True
        return False

    def _prior_turn_was_task_context(
        self, conversation_history: ConversationHistory | None
    ) -> bool:
        """Prior user turn looked like a task rather than pure small talk."""
        prior_user = self._last_user_message_before_current(conversation_history)
        if not prior_user:
            return False
        if self._needs_web_search(prior_user) or self._needs_file_write(prior_user):
            return True
        lowered = prior_user.lower()
        task_verbs = (
            "summarize",
            "search",
            "find",
            "fix",
            "write",
            "read",
            "analyze",
            "explain",
            "compare",
            "list",
            "show",
            "check",
            "audit",
            "report",
            "document",
        )
        return any(verb in lowered for verb in task_verbs)

    def _is_conversation_follow_up(
        self, user_input: str, conversation_history: ConversationHistory | None
    ) -> bool:
        """True when the current message continues an in-progress exchange."""
        if not conversation_history or len(conversation_history.messages) < 2:
            return False
        last_assistant = self._last_assistant_message(conversation_history)
        if not last_assistant:
            return False
        if self._assistant_message_awaited_reply(last_assistant):
            return True
        if self._is_short_follow_up_reply(user_input) and self._prior_turn_was_task_context(
            conversation_history
        ):
            return True
        return False

    def _follow_up_routing_message(
        self, user_input: str, conversation_history: ConversationHistory | None
    ) -> str:
        """Combine prior user turn with a short follow-up for routing heuristics."""
        prior_user = self._last_user_message_before_current(conversation_history)
        if prior_user and self._is_short_follow_up_reply(user_input):
            return f"{prior_user} {user_input}"
        return user_input

    def _orchestrator_follow_up_intent(self, notes: str) -> dict[str, Any]:
        return {
            "type": "task",
            "complexity": "moderate",
            "requires_tools": True,
            "suggested_response": "orchestrator",
            "notes": notes,
            "confidence": 0.85,
        }

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

    def _is_casual_conversation(
        self,
        message: str,
        conversation_history: ConversationHistory | None = None,
    ) -> bool:
        """
        Determine if a message is casual conversation.

        Args:
            message: The user's message
            conversation_history: Optional prior turns (current user message may
                already be appended)

        Returns:
            True if it's casual conversation
        """
        if conversation_history and self._is_conversation_follow_up(message, conversation_history):
            return False

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
