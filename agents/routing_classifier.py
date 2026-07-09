"""Deterministic-first routing for WitsControlCenterAgent.

Consolidates keyword/heuristic routing into a single ordered classifier so
intent analysis does not rely on brittle length-based casual bypasses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from core.schemas import ConversationHistory

RouteDestination = Literal[
    "orchestrator",
    "self_repair",
    "book_writing",
    "coding",
    "greeting",
    "needs_intent",
    "guest_chat_history",
    "guest_profile",
    "knowledge_log",
    "read_file_direct",
    "playbook",
    "continuation",
]

# --- shared signal lists (single source of truth for routing heuristics) ---

DOCUMENT_TOOL_HINTS = (
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

WEB_SEARCH_SIGNALS = (
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
    "latest",
    "most recent",
    "breaking news",
    "in the news",
    "news about",
    "up to date",
    "up-to-date",
    "weather",
    "forecast",
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

FILE_SAVE_SIGNALS = (
    "save this conversation",
    "save our conversation",
    "save the conversation",
    "save a copy",
    "save a copy of",
    "save to file",
    "save to a file",
    "save to disk",
    "save as",
    "write to file",
    "write it to",
    "export conversation",
    "export this chat",
    "log of our conversation",
    "save a log",
    "save this chat",
    "write the story",
    "save the story",
    "save as a file",
)

CODEBASE_INTRO_SIGNALS = (
    "your codebase",
    "your own files",
    "your own code",
    "tell me about yourself",
    "look at your code",
    "look at your files",
    "look at your own files",
    "read your own files",
    "actually look at your",
    "wits v3",
    "this codebase",
    "about your code",
    "about the codebase",
    "about yourself",
    "this project",
)

CODEBASE_AUTHORING_VERBS = (
    "create",
    "write",
    "generate",
    "build",
    "implement",
    "scaffold",
)

SELF_REPAIR_SIGNALS = (
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
    "run self repair",
    "run self-repair",
    "run a self repair",
    "run a self-repair",
    "run the self repair",
    "run the self-repair",
    "self repair",
    "self-repair",
    "start self repair",
    "start self-repair",
    "do a self repair",
    "do a self-repair",
    "perform self repair",
    "perform self-repair",
    "initiate self repair",
    "initiate self-repair",
    "repair yourself",
    "diagnose errors",
    "diagnose the log",
    "diagnose log",
    "diagnose application log",
    "run tests",
    "run the test suite",
    "run pytest",
    "failing tests",
)

MATH_CALCULATOR_RE = re.compile(
    r"\b(square[- ]?root|sqrt|calculate|compute|what is|what's)\b.*\b(\d+[\d.,]*)\b"
    r"|\b(\d+[\d.,]*)\s*[\+\-\*\/\^]\s*(\d+[\d.,]*)",
    re.IGNORECASE,
)

EXPLICIT_READ_FILE_RE = re.compile(
    r"\bread\s+(?:the\s+)?(?:file\s+)?([^\s?\"']+\.(?:py|md|txt|json|yaml|yml|log|html|csv|pdf))\b",
    re.IGNORECASE,
)

DOWNLOADS_READ_SIGNALS = (
    "downloads",
    r"d:\downloads",
    "my downloads",
    "competitive landscape",
)

DOWNLOADS_FILE_RE = re.compile(
    r"(?:downloads[/\\]|d:\\downloads[/\\])?([^\s?\"']+\.(?:pdf|md|txt|docx?|csv|json|yaml|yml))\b",
    re.IGNORECASE,
)

STORY_NOUN_RE = re.compile(
    r"\b(story|novel|screenplay|fanfiction|fan fiction|short story|poem|tale)\b",
    re.IGNORECASE,
)
AUTHORING_VERB_RE = re.compile(
    r"\b(write|writing|wrote|create|creating|compose|composing|tell|telling|"
    r"craft|crafting|pen|penning)\b",
    re.IGNORECASE,
)

AGENT_CONTINUATION_RE = re.compile(
    r"^\s*(okay,?\s*|ok,?\s*|alright,?\s*|so\s+)*"
    r"(please\s+)?"
    r"(make it|do it|write it( all)?|go ahead|finish it|finish (it|the (story|book|chapter))|"
    r"continue( writing)?|keep (going|writing)|"
    r"write the (whole|entire|full) (story|book|thing)|"
    r"save (it|this)( now)?( to disk)?)\s*\.?\s*$",
    re.IGNORECASE,
)

GREETING_WORDS = frozenset(
    {"hello", "hi", "hey", "thanks", "appreciate", "nice", "cool", "great", "bye", "goodbye"}
)
GREETING_PHRASES = (
    "how are you",
    "what's up",
    "how's it going",
    "thank you",
    "see you",
    "talk to you",
)

GUEST_AUDIT_SIGNALS = (
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

GUEST_PROFILE_SIGNALS = (
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

GUEST_PROFILE_PERSON_QUERY_SIGNALS = (
    "tell me about",
    "who is",
    "what is",
    "what's",
    "know about",
    "into",
)

GUEST_ACCOUNTS_SIGNALS = (
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

GUEST_CHAT_ACTIVITY_SIGNALS = (
    "chat with",
    "chatted",
    "chat about",
    "chatting about",
    "talk with",
    "talked",
    "talk about",
    "talking about",
    "talk to you",
    "talked to you",
    "discuss",
    "discussed",
    "say to you",
    "said to you",
    "ask you",
    "asked you",
    "asking about",
    "conversation with",
    "conversations with",
    "message you",
    "messaged you",
)

KNOWLEDGE_LOG_SIGNALS = (
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

GENERIC_ASSISTANT_QUESTIONS = (
    "how can i help",
    "how may i assist",
    "what can i do for you",
    "anything else i can",
    "how are you",
    "what would you like to work on",
)


@dataclass
class RouteDecision:
    destination: RouteDestination
    reason: str
    orchestrator_notes: str = ""
    preferred_tool: str = ""
    playbook_id: str = ""
    file_path: str = ""

    def to_intent(self) -> dict[str, Any] | None:
        """Map to WCCA intent dict, or None when the caller should run the intent LLM."""
        if self.destination == "greeting":
            return {
                "type": "conversation",
                "complexity": "simple",
                "requires_tools": False,
                "suggested_response": "direct",
                "notes": self.reason,
                "confidence": 0.95,
            }
        if self.destination == "self_repair":
            return {
                "type": "goal_defined",
                "complexity": "moderate",
                "requires_tools": True,
                "suggested_response": "specialized",
                "specialized_agent": "self_repair",
                "notes": self.reason,
                "confidence": 0.9,
            }
        if self.destination == "book_writing":
            return {
                "type": "goal_defined",
                "complexity": "moderate",
                "requires_tools": True,
                "suggested_response": "specialized",
                "specialized_agent": "book_writing",
                "notes": self.reason,
                "confidence": 0.9,
            }
        if self.destination == "coding":
            return {
                "type": "goal_defined",
                "complexity": "moderate",
                "requires_tools": True,
                "suggested_response": "specialized",
                "specialized_agent": "coding",
                "notes": self.reason,
                "confidence": 0.85,
            }
        if self.destination == "orchestrator":
            intent: dict[str, Any] = {
                "type": "task",
                "complexity": "moderate",
                "requires_tools": True,
                "suggested_response": "orchestrator",
                "notes": self.orchestrator_notes or self.reason,
                "confidence": 0.85,
            }
            if self.preferred_tool:
                intent["preferred_tool"] = self.preferred_tool
            return intent
        if self.destination in ("guest_chat_history", "guest_profile", "knowledge_log"):
            return {
                "type": "task",
                "complexity": "moderate",
                "requires_tools": True,
                "suggested_response": "direct_tool",
                "routing_destination": self.destination,
                "notes": self.reason,
                "confidence": 0.9,
            }
        if self.destination == "read_file_direct":
            return {
                "type": "task",
                "complexity": "simple",
                "requires_tools": True,
                "suggested_response": "direct_tool",
                "routing_destination": "read_file_direct",
                "file_path": self.file_path,
                "notes": self.reason,
                "confidence": 0.9,
            }
        if self.destination == "playbook":
            return {
                "type": "goal_defined",
                "complexity": "moderate",
                "requires_tools": True,
                "suggested_response": "playbook",
                "playbook_id": self.playbook_id,
                "notes": self.reason,
                "confidence": 0.9,
            }
        if self.destination == "continuation":
            return {
                "type": "task",
                "complexity": "simple",
                "requires_tools": False,
                "suggested_response": "continuation",
                "notes": self.reason,
                "confidence": 0.9,
            }
        return None


@dataclass
class RoutingContext:
    message: str
    user_role: str = "owner"
    doc_inventory: dict[str, int] = field(default_factory=dict)
    session_id: str | None = None
    active_specialized_agent: str | None = None
    last_task_route: str | None = None
    conversation_history: ConversationHistory | None = None
    greeting_only_direct: bool = True


def doc_routing_hints(doc_inventory: dict[str, int]) -> set[str]:
    hints: set[str] = set()
    for path in doc_inventory:
        name = Path(path).name.lower()
        hints.add(name)
        hints.update(w for w in re.split(r"[\W_]+", Path(path).stem.lower()) if len(w) >= 4)
    return hints


def needs_math_calculator(message: str) -> bool:
    """Pure math that should use calculator, not web_search."""
    if needs_web_search(message):
        return False
    return bool(MATH_CALCULATOR_RE.search(message))


def needs_explicit_file_read(message: str) -> str | None:
    """Return file path when user names a specific file to read."""
    match = EXPLICIT_READ_FILE_RE.search(message)
    if match:
        return match.group(1)
    return None


def needs_downloads_file_read(message: str) -> str | None:
    """Return Downloads-relative path when user asks about files in Downloads."""
    lowered = message.lower()
    if not any(sig in lowered for sig in DOWNLOADS_READ_SIGNALS):
        return None
    match = DOWNLOADS_FILE_RE.search(message)
    if match:
        name = match.group(1).lstrip("/\\")
        return name
    # Generic "list downloads" — no specific file
    if any(p in lowered for p in ("list", "show", "what's in", "whats in")):
        return "__downloads_dir__"
    return None


def needs_web_search(message: str) -> bool:
    lowered = message.lower()
    if any(sig in lowered for sig in WEB_SEARCH_SIGNALS):
        return True
    has_question = "?" in message or bool(
        re.search(r"\b(who|what|when|where|which|whose|did|does|is|are|died|won)\b", lowered)
    )
    if has_question and re.search(r"\b(202[4-9]|20[3-9]\d)\b", lowered):
        return True
    return False


def needs_file_write(message: str) -> bool:
    lowered = message.lower()
    if needs_story_writing(message) and not any(
        s in lowered
        for s in ("conversation", "chat", "copy of", "export conversation", "log of our")
    ):
        return False
    return any(sig in lowered for sig in FILE_SAVE_SIGNALS)


def needs_codebase_intro(message: str) -> bool:
    lowered = message.lower()
    return any(sig in lowered for sig in CODEBASE_INTRO_SIGNALS)


def is_code_introspection_not_authoring(message: str) -> bool:
    """True when the user wants to read/describe the project, not author new code."""
    lowered = message.lower()
    intro_hints = (
        "codebase",
        "your code",
        "your files",
        "your own files",
        "your own code",
        "yourself",
        "this project",
        "wits",
        "about you",
        "look at your",
        "read your",
    )
    if not any(h in lowered for h in intro_hints):
        return False
    return not any(v in lowered for v in CODEBASE_AUTHORING_VERBS)


def needs_self_repair(message: str) -> bool:
    lowered = message.lower()
    return any(sig in lowered for sig in SELF_REPAIR_SIGNALS)


def needs_story_writing(message: str) -> bool:
    return bool(STORY_NOUN_RE.search(message) and AUTHORING_VERB_RE.search(message))


def is_agent_continuation_phrase(message: str) -> bool:
    return bool(AGENT_CONTINUATION_RE.match(message.strip()))


def message_mentions_guest_name(message: str) -> bool:
    from core.guest_access import GuestRegistry

    lowered = message.lower()
    for guest in GuestRegistry().list_active_guests():
        name = (guest.get("display_name") or "").strip().lower()
        if name and name in lowered:
            return True
    return False


def needs_guest_chat_history(message: str) -> bool:
    lowered = message.lower()
    if not message_mentions_guest_name(message):
        return False
    return any(sig in lowered for sig in GUEST_CHAT_ACTIVITY_SIGNALS)


def needs_guest_audit_review(message: str) -> bool:
    lowered = message.lower()
    if any(sig in lowered for sig in GUEST_AUDIT_SIGNALS):
        return True
    return needs_guest_chat_history(message)


def needs_guest_accounts_list(message: str) -> bool:
    lowered = message.lower()
    return any(sig in lowered for sig in GUEST_ACCOUNTS_SIGNALS)


def needs_guest_profile_review(message: str) -> bool:
    lowered = message.lower()
    mentions_known_guest = message_mentions_guest_name(message)
    if mentions_known_guest and any(
        sig in lowered for sig in GUEST_PROFILE_PERSON_QUERY_SIGNALS
    ):
        return True
    if not any(sig in lowered for sig in GUEST_PROFILE_SIGNALS):
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


def needs_guest_admin_review(message: str) -> bool:
    if needs_guest_profile_review(message):
        return True
    return needs_guest_audit_review(message) or needs_guest_accounts_list(message)


def needs_knowledge_log_review(message: str) -> bool:
    lowered = message.lower()
    return any(sig in lowered for sig in KNOWLEDGE_LOG_SIGNALS)


def requires_orchestrator(message: str, doc_inventory: dict[str, int]) -> bool:
    if needs_guest_admin_review(message):
        return True
    if needs_file_write(message):
        return True
    if needs_math_calculator(message):
        return True
    lowered = message.lower()
    doc_hints = doc_routing_hints(doc_inventory)
    if any(h in lowered for h in DOCUMENT_TOOL_HINTS) or any(h in lowered for h in doc_hints):
        return True
    return needs_web_search(message)


def is_pure_greeting(message: str) -> bool:
    """True only for explicit greetings/thanks/bye — never by message length."""
    lowered = message.lower()
    words = set(re.findall(r"[a-z']+", lowered))
    if words & GREETING_WORDS:
        return True
    return any(phrase in lowered for phrase in GREETING_PHRASES)


def _messages_before_current_turn(conversation_history: ConversationHistory | None) -> list:
    if not conversation_history or not conversation_history.messages:
        return []
    messages = conversation_history.messages
    if messages[-1].role == "user":
        return messages[:-1]
    return messages


def _last_assistant_message(conversation_history: ConversationHistory | None) -> str:
    for msg in reversed(_messages_before_current_turn(conversation_history)):
        if msg.role == "assistant":
            return msg.content or ""
    return ""


def _last_user_message_before_current(conversation_history: ConversationHistory | None) -> str:
    prior = _messages_before_current_turn(conversation_history)
    for msg in reversed(prior):
        if msg.role == "user":
            return msg.content or ""
    return ""


def _assistant_message_awaited_reply(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    lowered = t.lower()
    if any(marker in lowered for marker in GENERIC_ASSISTANT_QUESTIONS):
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


def is_short_follow_up_reply(message: str) -> bool:
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


def _prior_turn_was_task_context(conversation_history: ConversationHistory | None) -> bool:
    prior_user = _last_user_message_before_current(conversation_history)
    if not prior_user:
        return False
    if needs_web_search(prior_user) or needs_file_write(prior_user):
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


def is_conversation_follow_up(
    user_input: str, conversation_history: ConversationHistory | None
) -> bool:
    if not conversation_history or len(conversation_history.messages) < 2:
        return False
    last_assistant = _last_assistant_message(conversation_history)
    if not last_assistant:
        return False
    if _assistant_message_awaited_reply(last_assistant):
        return True
    if is_short_follow_up_reply(user_input) and _prior_turn_was_task_context(conversation_history):
        return True
    return False


def follow_up_routing_message(
    user_input: str, conversation_history: ConversationHistory | None
) -> str:
    prior_user = _last_user_message_before_current(conversation_history)
    if prior_user and is_short_follow_up_reply(user_input):
        return f"{prior_user} {user_input}"
    return user_input


def classify_message(ctx: RoutingContext) -> RouteDecision:
    """Single ordered routing decision before the intent LLM."""
    message = ctx.message
    role = ctx.user_role

    if role == "owner" and needs_guest_chat_history(message):
        return RouteDecision(
            "guest_chat_history",
            "Owner guest chat history query — direct audit tool.",
        )

    if role == "owner" and needs_guest_profile_review(message):
        return RouteDecision(
            "guest_profile",
            "Owner guest profile query — direct profile tool.",
        )

    if role == "owner" and needs_knowledge_log_review(message):
        return RouteDecision(
            "knowledge_log",
            "Owner project knowledge query — direct knowledge log tool.",
        )

    if needs_file_write(message):
        from core.playbooks import match_playbook

        pb = match_playbook(message, doc_inventory=ctx.doc_inventory)
        if pb:
            return RouteDecision(
                "playbook",
                f"Save/export playbook — {pb}",
                playbook_id=pb,
            )
        return RouteDecision(
            "orchestrator",
            "Save/export to file — routing to orchestrator.",
            "Save/export to file — routing to orchestrator for "
            "read_conversation_history + write_file.",
        )

    if needs_codebase_intro(message):
        return RouteDecision(
            "playbook",
            "Codebase introspection — codebase_tour playbook.",
            playbook_id="codebase_tour",
        )

    explicit_path = needs_explicit_file_read(message)
    if explicit_path and role != "guest":
        return RouteDecision(
            "read_file_direct",
            f"Explicit file read — {explicit_path}",
            file_path=explicit_path,
        )

    downloads_path = needs_downloads_file_read(message)
    if downloads_path and role != "guest":
        return RouteDecision(
            "read_file_direct",
            f"Downloads file read — {downloads_path}",
            file_path=downloads_path,
        )

    from core.playbooks import match_playbook

    playbook_id = match_playbook(message, doc_inventory=ctx.doc_inventory)
    if playbook_id:
        return RouteDecision(
            "playbook",
            f"Playbook match — {playbook_id}",
            playbook_id=playbook_id,
        )

    if role != "guest" and needs_self_repair(message):
        return RouteDecision(
            "self_repair",
            "Self-repair signals matched — route to self-repair agent.",
        )

    if needs_story_writing(message):
        return RouteDecision(
            "book_writing",
            "Creative writing signals matched — route to book-writing agent.",
        )

    if requires_orchestrator(message, ctx.doc_inventory):
        notes = "Routing to orchestrator for tool use."
        preferred = "calculator" if needs_math_calculator(message) else ""
        if needs_web_search(message):
            notes = (
                "Needs current/external info or an explicit lookup — "
                "routing to orchestrator for web_search."
            )
        elif any(h in message.lower() for h in DOCUMENT_TOOL_HINTS):
            notes = "References user documents/files — routing to orchestrator."
        return RouteDecision(
            "orchestrator",
            "Tool/orchestrator guard matched.",
            notes,
            preferred_tool=preferred,
        )

    if (
        ctx.last_task_route in ("orchestrator", "playbook")
        and is_agent_continuation_phrase(message)
    ):
        return RouteDecision(
            "continuation",
            f"Continuation phrase — resume prior {ctx.last_task_route} task.",
        )

    if ctx.conversation_history and is_conversation_follow_up(message, ctx.conversation_history):
        routing_message = follow_up_routing_message(message, ctx.conversation_history)
        if requires_orchestrator(routing_message, ctx.doc_inventory):
            return RouteDecision(
                "orchestrator",
                "Follow-up to a prior task — routing to orchestrator.",
                "Follow-up to a prior task — routing to orchestrator with conversation context.",
            )

    if (
        ctx.active_specialized_agent
        and is_agent_continuation_phrase(message)
    ):
        agent = ctx.active_specialized_agent
        if agent == "self_repair":
            return RouteDecision(
                "self_repair",
                "Continuation phrase — resume self-repair agent.",
            )
        if agent == "book_writing":
            return RouteDecision(
                "book_writing",
                "Continuation phrase — resume book-writing agent.",
            )
        if agent == "coding":
            return RouteDecision(
                "coding",
                "Continuation phrase — resume coding agent.",
            )

    if ctx.greeting_only_direct and is_pure_greeting(message):
        if ctx.conversation_history and is_conversation_follow_up(
            message, ctx.conversation_history
        ):
            pass
        else:
            return RouteDecision(
                "greeting",
                "Explicit greeting or thanks — direct reply.",
            )

    return RouteDecision("needs_intent", "No deterministic match — use slim intent LLM.")
