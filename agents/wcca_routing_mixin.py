# agents/wcca_routing_mixin.py
"""Orchestrator routing helpers for WitsControlCenterAgent."""

from typing import Any

from agents import routing_classifier as rc
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

    # Re-export signal lists for tests and backward compatibility.
    _DOCUMENT_TOOL_HINTS = rc.DOCUMENT_TOOL_HINTS
    _WEB_SEARCH_SIGNALS = rc.WEB_SEARCH_SIGNALS
    _FILE_SAVE_SIGNALS = rc.FILE_SAVE_SIGNALS
    _SELF_REPAIR_SIGNALS = rc.SELF_REPAIR_SIGNALS
    _STORY_NOUN_RE = rc.STORY_NOUN_RE
    _AUTHORING_VERB_RE = rc.AUTHORING_VERB_RE
    _AGENT_CONTINUATION_RE = rc.AGENT_CONTINUATION_RE
    _GUEST_AUDIT_SIGNALS = rc.GUEST_AUDIT_SIGNALS
    _GUEST_PROFILE_SIGNALS = rc.GUEST_PROFILE_SIGNALS
    _GUEST_PROFILE_PERSON_QUERY_SIGNALS = rc.GUEST_PROFILE_PERSON_QUERY_SIGNALS
    _GUEST_ACCOUNTS_SIGNALS = rc.GUEST_ACCOUNTS_SIGNALS
    _GUEST_CHAT_ACTIVITY_SIGNALS = rc.GUEST_CHAT_ACTIVITY_SIGNALS
    _KNOWLEDGE_LOG_SIGNALS = rc.KNOWLEDGE_LOG_SIGNALS
    _GENERIC_ASSISTANT_QUESTIONS = rc.GENERIC_ASSISTANT_QUESTIONS

    def _needs_web_search(self, message: str) -> bool:
        return rc.needs_web_search(message)

    def _doc_routing_hints(self, doc_inventory: dict[str, int]) -> set:
        return rc.doc_routing_hints(doc_inventory)

    def _needs_file_write(self, message: str) -> bool:
        return rc.needs_file_write(message)

    def _needs_self_repair(self, message: str) -> bool:
        return rc.needs_self_repair(message)

    def _needs_story_writing(self, message: str) -> bool:
        return rc.needs_story_writing(message)

    def _is_agent_continuation_phrase(self, message: str) -> bool:
        return rc.is_agent_continuation_phrase(message)

    def _needs_guest_chat_history(self, message: str) -> bool:
        return rc.needs_guest_chat_history(message)

    def _needs_guest_audit_review(self, message: str) -> bool:
        return rc.needs_guest_audit_review(message)

    def _needs_guest_accounts_list(self, message: str) -> bool:
        return rc.needs_guest_accounts_list(message)

    def _message_mentions_guest_name(self, message: str) -> bool:
        return rc.message_mentions_guest_name(message)

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
        return rc.needs_guest_profile_review(message)

    def _needs_guest_admin_review(self, message: str) -> bool:
        return rc.needs_guest_admin_review(message)

    def _needs_knowledge_log_review(self, message: str) -> bool:
        return rc.needs_knowledge_log_review(message)

    async def _requires_orchestrator_for_input(self, user_input: str) -> bool:
        if self._needs_guest_admin_review(user_input):
            return True
        if self._needs_file_write(user_input):
            return True
        doc_inventory = await self._get_document_inventory()
        return rc.requires_orchestrator(user_input, doc_inventory)

    def _messages_before_current_turn(
        self, conversation_history: ConversationHistory | None
    ) -> list:
        return rc._messages_before_current_turn(conversation_history)

    def _last_assistant_message(self, conversation_history: ConversationHistory | None) -> str:
        return rc._last_assistant_message(conversation_history)

    def _last_user_message_before_current(
        self, conversation_history: ConversationHistory | None
    ) -> str:
        return rc._last_user_message_before_current(conversation_history)

    def _assistant_message_awaited_reply(self, text: str) -> bool:
        return rc._assistant_message_awaited_reply(text)

    def _is_short_follow_up_reply(self, message: str) -> bool:
        return rc.is_short_follow_up_reply(message)

    def _prior_turn_was_task_context(
        self, conversation_history: ConversationHistory | None
    ) -> bool:
        return rc._prior_turn_was_task_context(conversation_history)

    def _is_conversation_follow_up(
        self, user_input: str, conversation_history: ConversationHistory | None
    ) -> bool:
        return rc.is_conversation_follow_up(user_input, conversation_history)

    def _follow_up_routing_message(
        self, user_input: str, conversation_history: ConversationHistory | None
    ) -> str:
        return rc.follow_up_routing_message(user_input, conversation_history)

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

    def _is_pure_greeting(self, message: str) -> bool:
        """True only for explicit greetings/thanks/bye — never by message length."""
        return rc.is_pure_greeting(message)

    def _is_casual_conversation(
        self,
        message: str,
        conversation_history: ConversationHistory | None = None,
    ) -> bool:
        """Deprecated alias — greeting whitelist only (no length shortcut)."""
        if conversation_history and self._is_conversation_follow_up(message, conversation_history):
            return False
        return self._is_pure_greeting(message)

    async def _classify_routing(
        self,
        user_input: str,
        conversation_history: ConversationHistory | None,
        session_id: str | None = None,
    ) -> rc.RouteDecision:
        doc_inventory = await self._get_document_inventory()
        ctx = rc.RoutingContext(
            message=user_input,
            user_role=getattr(self, "_request_user_role", "owner"),
            doc_inventory=doc_inventory,
            session_id=session_id,
            active_specialized_agent=(
                self._active_specialized_agent.get(session_id)
                if session_id and hasattr(self, "_active_specialized_agent")
                else None
            ),
            conversation_history=conversation_history,
            greeting_only_direct=self.config.routing.greeting_only_direct,
        )
        return rc.classify_message(ctx)

    @staticmethod
    def _message_references_documents(message: str, doc_inventory: dict[str, int]) -> bool:
        lowered = message.lower()
        if any(h in lowered for h in rc.DOCUMENT_TOOL_HINTS):
            return True
        return any(h in lowered for h in rc.doc_routing_hints(doc_inventory))
