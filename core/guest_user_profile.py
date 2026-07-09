"""Per-guest interest/fact profiles built from casual chat (separate from global memory)."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.runtime_paths import guest_user_profiles_dir

from core.json_llm_parser import parse_json_object

logger = logging.getLogger("WitsV3.GuestUserProfile")

def default_profile_dir() -> Path:
    return guest_user_profiles_dir()


DEFAULT_PROFILE_DIR = default_profile_dir
MAX_FACTS = 40
MAX_FACT_LEN = 200

# Known interest/topic keywords (lowercase). Counted when mentioned in user messages.
INTEREST_KEYWORDS: dict[str, str] = {
    "minecraft": "Minecraft",
    "roblox": "Roblox",
    "fortnite": "Fortnite",
    "pokemon": "Pokémon",
    "zelda": "Zelda",
    "coding": "Coding",
    "programming": "Programming",
    "python": "Python",
    "math": "Math",
    "science": "Science",
    "homework": "School / homework",
    "basketball": "Basketball",
    "soccer": "Soccer",
    "football": "Football",
    "music": "Music",
    "guitar": "Guitar",
    "drawing": "Drawing",
    "art": "Art",
    "reading": "Reading",
    "books": "Books",
    "anime": "Anime",
    "marvel": "Marvel",
    "star wars": "Star Wars",
    "lego": "LEGO",
    "redstone": "Minecraft redstone",
    "survival mode": "Minecraft survival",
    "hardcore": "Minecraft hardcore",
}

# Capture short self-reported facts from user phrasing. Capture up to sentence-ending
# punctuation (not just \w\s) so clauses like "I am Richard's wife" aren't truncated
# into a false claim ("I am Richard") at the apostrophe.
_FACT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bi(?:'m| am)\s+(?:a\s+)?([^.!?\n]{2,80})", re.I),
    re.compile(r"\bi like\s+([^.!?\n]{3,80})", re.I),
    re.compile(r"\bi love\s+([^.!?\n]{3,80})", re.I),
    re.compile(r"\bmy favorite\s+([^.!?\n]{3,80})", re.I),
    re.compile(r"\bi play\s+([^.!?\n]{3,80})", re.I),
    re.compile(r"\bi'm interested in\s+([^.!?\n]{3,80})", re.I),
)

# Words that plausibly open a question, used to tell statements from questions
# when logging a conversation-derived fact (see update_from_turn).
_QUESTION_STARTERS: frozenset[str] = frozenset(
    {
        "who",
        "what",
        "when",
        "where",
        "why",
        "how",
        "which",
        "whose",
        "is",
        "are",
        "am",
        "was",
        "were",
        "do",
        "does",
        "did",
        "can",
        "could",
        "will",
        "would",
        "should",
        "shall",
        "may",
        "might",
        "has",
        "have",
        "had",
    }
)


def _is_question(text: str) -> bool:
    """Heuristic: trailing '?' or a leading interrogative word."""
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.endswith("?"):
        return True
    first_word = stripped.split(None, 1)[0].strip(".,!\"'").lower()
    return first_word in _QUESTION_STARTERS


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _empty_profile(guest_id: str, display_name: str) -> dict[str, Any]:
    now = _now_iso()
    return {
        "guest_id": guest_id,
        "display_name": display_name,
        "created_at": now,
        "updated_at": now,
        "turn_count": 0,
        "interests": {},
        "facts": [],
        "recent_topics": [],
    }


class GuestUserProfileStore:
    """JSON document per guest under data/guest_user_profiles/<guest_id>.json."""

    def __init__(self, base_dir: Path | str | None = None):
        self.base_dir = Path(base_dir) if base_dir else default_profile_dir()

    def _path(self, guest_id: str) -> Path:
        return self.base_dir / f"{guest_id}.json"

    @staticmethod
    def _merge_profile_data(into: dict[str, Any], other: dict[str, Any]) -> None:
        into["turn_count"] = int(into.get("turn_count", 0)) + int(other.get("turn_count", 0))
        interests: dict[str, int] = into.setdefault("interests", {})
        for label, count in (other.get("interests") or {}).items():
            interests[label] = interests.get(label, 0) + int(count)
        topics: list[str] = list(into.get("recent_topics") or [])
        for topic in other.get("recent_topics") or []:
            if topic not in topics:
                topics.append(topic)
        into["recent_topics"] = topics[-12:]
        facts: list[dict[str, Any]] = list(into.get("facts") or [])
        seen = {f.get("text") for f in facts}
        for fact in other.get("facts") or []:
            text = fact.get("text")
            if text and text not in seen:
                facts.append(fact)
                seen.add(text)
        into["facts"] = facts[-MAX_FACTS:]

    def delete_profile(self, guest_id: str) -> bool:
        path = self._path(guest_id)
        if path.is_file():
            path.unlink()
            return True
        return False

    def merge_profiles(
        self, *, target_guest_id: str, source_guest_id: str, display_name: str
    ) -> dict[str, Any]:
        """Merge source profile JSON into target and remove source file."""
        target = self.load(target_guest_id, display_name)
        source = self.load(source_guest_id, display_name)
        self._merge_profile_data(target, source)
        target["guest_id"] = target_guest_id
        target["display_name"] = display_name
        self.save(target)
        self.delete_profile(source_guest_id)
        return target

    def consolidate_for_display_name(
        self, registry: Any, display_name: str
    ) -> dict[str, Any] | None:
        """Merge duplicate account profiles into the canonical guest_id for this name."""
        accounts = registry.find_all_by_display_name(display_name)
        if not accounts:
            return None
        canonical = max(accounts, key=lambda g: float(g.get("last_seen") or 0))
        cid = canonical["guest_id"]
        name = canonical.get("display_name", display_name)
        merged = self.load(cid, name)
        for acct in accounts:
            if acct["guest_id"] == cid:
                continue
            other = self.load(acct["guest_id"], name)
            self._merge_profile_data(merged, other)
            self.delete_profile(acct["guest_id"])
        merged["guest_id"] = cid
        merged["display_name"] = name
        self.save(merged)
        return merged

    def load_merged_for_display_name(
        self, display_name: str, registry: Any | None = None
    ) -> dict[str, Any] | None:
        from core.guest_access import GuestRegistry

        reg = registry or GuestRegistry()
        accounts = reg.find_all_by_display_name(display_name)
        if not accounts:
            return None
        canonical = max(accounts, key=lambda g: float(g.get("last_seen") or 0))
        name = canonical.get("display_name", display_name)
        merged = self.load(canonical["guest_id"], name)
        for acct in accounts:
            if acct["guest_id"] == canonical["guest_id"]:
                continue
            self._merge_profile_data(merged, self.load(acct["guest_id"], name))
        merged["guest_id"] = canonical["guest_id"]
        merged["display_name"] = name
        return merged

    def load(self, guest_id: str, display_name: str = "Guest") -> dict[str, Any]:
        path = self._path(guest_id)
        if not path.is_file():
            return _empty_profile(guest_id, display_name)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            data.setdefault("guest_id", guest_id)
            data.setdefault("interests", {})
            data.setdefault("facts", [])
            data.setdefault("recent_topics", [])
            return data
        except Exception as e:
            logger.warning("Failed to load guest user profile %s: %s", guest_id[:8], e)
            return _empty_profile(guest_id, display_name)

    def save(self, profile: dict[str, Any]) -> Path:
        guest_id = profile["guest_id"]
        profile["updated_at"] = _now_iso()
        path = self._path(guest_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
        return path

    def update_from_turn(
        self,
        *,
        guest_id: str,
        display_name: str,
        user_message: str,
        assistant_message: str | None = None,
    ) -> dict[str, Any]:
        """Extract interests/facts from a completed guest turn."""
        from core.guest_access import GuestRegistry

        reg = GuestRegistry()
        acct = reg.find_by_display_name(display_name)
        canonical_id = acct["guest_id"] if acct else guest_id
        profile = self.load(canonical_id, display_name)
        profile["display_name"] = display_name
        profile["turn_count"] = int(profile.get("turn_count", 0)) + 1

        normalized = user_message.lower()
        interests: dict[str, int] = profile.get("interests") or {}
        for keyword, label in INTEREST_KEYWORDS.items():
            if keyword in normalized:
                interests[label] = interests.get(label, 0) + 1
        profile["interests"] = interests

        topics: list[str] = profile.get("recent_topics") or []
        for keyword, label in INTEREST_KEYWORDS.items():
            if keyword in normalized and label not in topics:
                topics.append(label)
        profile["recent_topics"] = topics[-12:]

        facts: list[dict[str, Any]] = list(profile.get("facts") or [])
        for pattern in _FACT_PATTERNS:
            match = pattern.search(user_message)
            if match:
                snippet = match.group(0).strip()[:MAX_FACT_LEN]
                if snippet and not any(f.get("text") == snippet for f in facts):
                    facts.append({"text": snippet, "ts": _now_iso(), "source": "user_message"})
        profile["facts"] = facts[-MAX_FACTS:]

        if assistant_message and len(user_message) > 20:
            summary = user_message.strip()[:120]
            label = "Asked" if _is_question(user_message) else "Said"
            note = f"{label}: {summary}"
            if not any(f.get("text") == note for f in facts):
                facts.append({"text": note, "ts": _now_iso(), "source": "conversation"})
            profile["facts"] = facts[-MAX_FACTS:]

        self._save_canonical(profile, display_name)
        logger.info(
            "Updated guest user profile for %s (turns=%s, interests=%s)",
            display_name,
            profile["turn_count"],
            len(interests),
        )
        return profile

    def set_facts(self, *, guest_id: str, display_name: str, facts: list[str]) -> dict[str, Any]:
        """Owner-edited replacement of the facts list (e.g. to remove/correct a wrong entry).

        Existing timestamp/source are preserved for facts whose text is unchanged;
        new or edited lines are recorded with source "owner_edit".
        """
        from core.guest_access import GuestRegistry

        reg = GuestRegistry()
        acct = reg.find_by_display_name(display_name)
        canonical_id = acct["guest_id"] if acct else guest_id
        profile = self.load(canonical_id, display_name)
        profile["display_name"] = display_name

        existing_by_text = {f.get("text"): f for f in (profile.get("facts") or [])}
        new_facts: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw in facts:
            text = raw.strip()[:MAX_FACT_LEN]
            if not text or text in seen:
                continue
            seen.add(text)
            prior = existing_by_text.get(text)
            if prior:
                new_facts.append(prior)
            else:
                new_facts.append({"text": text, "ts": _now_iso(), "source": "owner_edit"})
        profile["facts"] = new_facts[-MAX_FACTS:]

        self._save_canonical(profile, display_name)
        logger.info(
            "Owner edited facts for guest profile %s (count=%s)", display_name, len(new_facts)
        )
        return profile

    def _save_canonical(
        self, profile: dict[str, Any], display_name: str, registry: Any | None = None
    ) -> Path:
        """Persist under the canonical guest_id for this display name."""
        from core.guest_access import GuestRegistry

        reg = registry or GuestRegistry()
        acct = reg.find_by_display_name(display_name)
        if acct:
            profile["guest_id"] = acct["guest_id"]
        profile["display_name"] = display_name
        return self.save(profile)

    def _merge_llm_extraction(self, profile: dict[str, Any], extracted: dict[str, Any]) -> None:
        interests: dict[str, int] = profile.get("interests") or {}
        for raw in extracted.get("interests") or []:
            label = str(raw).strip()[:60]
            if label:
                interests[label] = interests.get(label, 0) + 1
        profile["interests"] = interests

        topics: list[str] = profile.get("recent_topics") or []
        for raw in extracted.get("interests") or []:
            label = str(raw).strip()[:60]
            if label and label not in topics:
                topics.append(label)
        profile["recent_topics"] = topics[-12:]

        facts: list[dict[str, Any]] = list(profile.get("facts") or [])
        for raw in extracted.get("facts") or []:
            text = str(raw).strip()[:MAX_FACT_LEN]
            if text and not any(f.get("text") == text for f in facts):
                facts.append({"text": text, "ts": _now_iso(), "source": "llm"})
        profile["facts"] = facts[-MAX_FACTS:]

    async def update_from_turn_async(
        self,
        *,
        guest_id: str,
        display_name: str,
        user_message: str,
        assistant_message: str | None = None,
        llm_interface: Any | None = None,
        config: Any | None = None,
    ) -> dict[str, Any]:
        """Heuristic update plus optional LLM extraction for richer profiles."""
        profile = self.update_from_turn(
            guest_id=guest_id,
            display_name=display_name,
            user_message=user_message,
            assistant_message=assistant_message,
        )
        if not llm_interface or not config:
            return profile
        guest_cfg = getattr(getattr(config, "web_ui", None), "guest_access", None)
        if guest_cfg is not None and not getattr(guest_cfg, "profile_llm_extraction", True):
            return profile
        if len(user_message.strip()) < 8:
            return profile

        from core.model_router import ModelRouter

        router = ModelRouter(config)
        model = router.route(user_message, default=config.ollama_settings.default_model)
        reply = (assistant_message or "")[:400]
        prompt = (
            f"Extract interests and personal facts the guest revealed about themselves.\n"
            f"Guest name: {display_name}\n"
            f"User message: {user_message[:500]}\n"
            f"Assistant reply: {reply}\n\n"
            'Return JSON only: {"interests": ["topic"], "facts": ["short fact"]}\n'
            "0-4 interests, 0-3 facts. Empty arrays if nothing personal was shared."
        )
        try:
            raw = await llm_interface.generate_text(
                prompt,
                model=model,
                temperature=0.2,
                max_tokens=256,
                format="json",
            )

            def _validate_extraction(parsed: dict[str, Any]) -> dict[str, Any]:
                interests = parsed.get("interests", [])
                facts = parsed.get("facts", [])
                if not isinstance(interests, list):
                    interests = []
                if not isinstance(facts, list):
                    facts = []
                return {"interests": interests, "facts": facts}

            parsed = parse_json_object(
                raw,
                _validate_extraction,
                logger=logger,
                fallback=lambda _r, _e: {"interests": [], "facts": []},
            )
            if parsed.get("interests") or parsed.get("facts"):
                self._merge_llm_extraction(profile, parsed)
                self._save_canonical(profile, display_name, registry=None)
                logger.info(
                    "LLM profile extraction for %s: +%s interests, +%s facts",
                    display_name,
                    len(parsed.get("interests") or []),
                    len(parsed.get("facts") or []),
                )
        except Exception as e:
            logger.warning("LLM guest profile extraction failed for %s: %s", display_name, e)
        return profile

    def list_profile_summaries(self, registry: Any | None = None) -> list[dict[str, Any]]:
        """One summary per display_name (merged across duplicate guest_ids)."""
        from core.guest_access import GuestRegistry

        reg = registry or GuestRegistry()
        seen_names: set[str] = set()
        out: list[dict[str, Any]] = []
        for acct in reg.list_active_guests():
            name = (acct.get("display_name") or "Guest").strip()
            key = name.lower()
            if key in seen_names:
                continue
            seen_names.add(key)
            profile = self.load_merged_for_display_name(name, reg) or _empty_profile(
                acct["guest_id"], name
            )
            interests: dict[str, int] = profile.get("interests") or {}
            top = sorted(interests.items(), key=lambda x: (-x[1], x[0]))[:5]
            dupes = reg.find_all_by_display_name(name)
            out.append(
                {
                    "guest_id": profile.get("guest_id", acct["guest_id"]),
                    "display_name": name,
                    "turn_count": profile.get("turn_count", 0),
                    "updated_at": profile.get("updated_at"),
                    "top_interests": [{"label": k, "count": v} for k, v in top],
                    "fact_count": len(profile.get("facts") or []),
                    "duplicate_accounts": len(dupes),
                    "account_ids": [g["guest_id"] for g in dupes],
                }
            )
        return sorted(out, key=lambda x: x["display_name"].lower())

    def format_owner_summary(
        self,
        *,
        guest_id: str | None = None,
        display_name: str | None = None,
        registry: Any | None = None,
    ) -> str:
        from core.guest_access import GuestRegistry

        reg = registry or GuestRegistry()
        if guest_id:
            acct = reg.get(guest_id)
        elif display_name:
            acct = reg.find_by_display_name(display_name)
        else:
            return "Provide display_name or guest_id for a guest profile summary."

        if not acct:
            return f"No guest account found for {display_name or guest_id}."

        name = acct.get("display_name", "Guest")
        if display_name and not guest_id:
            profile = self.load_merged_for_display_name(name, reg) or self.load(
                acct["guest_id"], name
            )
            dupes = reg.find_all_by_display_name(name)
            if len(dupes) > 1:
                profile = self.consolidate_for_display_name(reg, name) or profile
        else:
            profile = self.load(acct["guest_id"], name)
        gid = profile.get("guest_id", acct["guest_id"])
        age = acct.get("age_band", "teen")

        if profile.get("turn_count", 0) == 0 and not profile.get("interests"):
            return (
                f"Guest profile: {name} (id {gid[:8]}…, age_band={age})\n"
                "No conversation-derived interests yet — they need to chat more first."
            )

        lines = [
            f"Guest profile: {name} (id {gid}, age_band={age})",
            f"Conversation turns recorded: {profile.get('turn_count', 0)}",
            f"Last updated: {profile.get('updated_at', 'unknown')}",
        ]

        interests: dict[str, int] = profile.get("interests") or {}
        if interests:
            ranked = sorted(interests.items(), key=lambda x: (-x[1], x[0]))
            lines.append("Top interests (mention count):")
            for label, count in ranked[:10]:
                lines.append(f"  - {label}: {count}")
        else:
            lines.append("Top interests: (none detected yet)")

        facts: list[dict[str, Any]] = profile.get("facts") or []
        if facts:
            lines.append("Notes from conversation:")
            for fact in facts[-8:]:
                lines.append(f"  - {fact.get('text', '')}")

        recent = profile.get("recent_topics") or []
        if recent:
            lines.append(f"Recent topic areas: {', '.join(recent[-8:])}")

        return "\n".join(lines)

    def personalization_block(
        self, guest_id: str, display_name: str = "Guest", registry: Any | None = None
    ) -> str:
        """Short context for guest chat personalization (not shown to guest)."""
        profile = self.load_merged_for_display_name(display_name, registry) or self.load(
            guest_id, display_name
        )
        interests: dict[str, int] = profile.get("interests") or {}
        if not interests and not profile.get("facts"):
            return ""

        parts = [f"[Guest personalization for {display_name}]"]
        if interests:
            top = sorted(interests.items(), key=lambda x: -x[1])[:5]
            parts.append("Known interests: " + ", ".join(f"{k} ({v})" for k, v in top))
        facts = profile.get("facts") or []
        if facts:
            snippets = [f["text"] for f in facts[-3:] if f.get("text")]
            if snippets:
                parts.append("Remember: " + "; ".join(snippets))
        parts.append("Use this to tailor replies warmly — do not recite it verbatim.")
        return "\n".join(parts)
