"""Per-guest interest/fact profiles built from casual chat (separate from global memory)."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.json_llm_parser import parse_json_object

logger = logging.getLogger("WitsV3.GuestUserProfile")

DEFAULT_PROFILE_DIR = Path("data/guest_user_profiles")
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

# Capture short self-reported facts from user phrasing.
_FACT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bi(?:'m| am)\s+(?:a\s+)?(\w[\w\s]{2,40})", re.I),
    re.compile(r"\bi like\s+(.{3,60})", re.I),
    re.compile(r"\bi love\s+(.{3,60})", re.I),
    re.compile(r"\bmy favorite\s+(.{3,60})", re.I),
    re.compile(r"\bi play\s+(.{3,60})", re.I),
    re.compile(r"\bi'm interested in\s+(.{3,60})", re.I),
)


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
        self.base_dir = Path(base_dir) if base_dir else DEFAULT_PROFILE_DIR

    def _path(self, guest_id: str) -> Path:
        return self.base_dir / f"{guest_id}.json"

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
        profile = self.load(guest_id, display_name)
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
            note = f"Asked: {summary}"
            if not any(f.get("text") == note for f in facts):
                facts.append({"text": note, "ts": _now_iso(), "source": "conversation"})
            profile["facts"] = facts[-MAX_FACTS:]

        self.save(profile)
        logger.info(
            "Updated guest user profile for %s (turns=%s, interests=%s)",
            display_name,
            profile["turn_count"],
            len(interests),
        )
        return profile

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
                self.save(profile)
                logger.info(
                    "LLM profile extraction for %s: +%s interests, +%s facts",
                    display_name,
                    len(parsed.get("interests") or []),
                    len(parsed.get("facts") or []),
                )
        except Exception as e:
            logger.warning("LLM guest profile extraction failed for %s: %s", display_name, e)
        return profile

    def list_profile_summaries(self) -> list[dict[str, Any]]:
        """All stored user profiles (for owner settings UI)."""
        if not self.base_dir.is_dir():
            return []
        out: list[dict[str, Any]] = []
        for path in sorted(self.base_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                interests: dict[str, int] = data.get("interests") or {}
                top = sorted(interests.items(), key=lambda x: (-x[1], x[0]))[:5]
                out.append(
                    {
                        "guest_id": data.get("guest_id", path.stem),
                        "display_name": data.get("display_name", "Guest"),
                        "turn_count": data.get("turn_count", 0),
                        "updated_at": data.get("updated_at"),
                        "top_interests": [{"label": k, "count": v} for k, v in top],
                        "fact_count": len(data.get("facts") or []),
                    }
                )
            except Exception as e:
                logger.warning("Skipping corrupt profile %s: %s", path.name, e)
        return out

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

        gid = acct["guest_id"]
        name = acct.get("display_name", "Guest")
        profile = self.load(gid, name)
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

    def personalization_block(self, guest_id: str, display_name: str = "Guest") -> str:
        """Short context for guest chat personalization (not shown to guest)."""
        profile = self.load(guest_id, display_name)
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
