"""Personality questionnaire routes for the WitsV3 web UI."""

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from web.schemas import PersonalityAnswers

logger = logging.getLogger("WitsV3.WebUI")


def register_personality_routes(app: FastAPI, system) -> None:
    """Register personality GET/POST/DELETE API routes on the FastAPI app."""

    @app.get("/api/personality")
    async def get_personality():
        from core.personality_manager import PersonalityManager

        pm = PersonalityManager(config=system.config)
        profile = pm.personality_profile or {}
        comm = profile.get("communication", {})
        personas = [
            r.get("name")
            for r in profile.get("persona_layers", {}).get("available_roles", [])
            if isinstance(r, dict) and r.get("name")
        ]
        return {
            "identity_label": profile.get("identity_label", ""),
            "default_role": profile.get("default_role", ""),
            "tone": comm.get("tone", ""),
            "language_level": comm.get("language_level", ""),
            "verbosity": comm.get("verbosity", ""),
            "structure_preference": comm.get("structure_preference", ""),
            "humor": comm.get("humor", ""),
            "default_persona": profile.get("persona_layers", {}).get("default_persona", ""),
            "core_directives": profile.get("core_directives", []),
            "available_personas": personas,
            "system_prompt": pm.get_system_prompt(),
        }

    @app.post("/api/personality")
    async def save_personality(answers: PersonalityAnswers):
        import yaml

        from core.personality_manager import PersonalityManager, reload_personality_manager

        overrides: dict[str, Any] = {}
        comm: dict[str, Any] = {}

        def clean(value: str | None) -> str | None:
            return value.strip() if value and value.strip() else None

        if clean(answers.identity_label):
            overrides["identity_label"] = clean(answers.identity_label)
        if clean(answers.default_role):
            overrides["default_role"] = clean(answers.default_role)
        for field in ("tone", "language_level", "verbosity", "structure_preference", "humor"):
            value = clean(getattr(answers, field))
            if value:
                comm[field] = value
        if comm:
            overrides["communication"] = comm
        if clean(answers.default_persona):
            overrides["persona_layers"] = {"default_persona": clean(answers.default_persona)}
        if answers.core_directives is not None:
            directives = [d.strip() for d in answers.core_directives if d.strip()]
            if directives:
                overrides["core_directives"] = directives

        if not overrides:
            return JSONResponse({"detail": "no answers provided"}, status_code=400)

        overrides_path = Path(
            PersonalityManager.overrides_path_for(system.config.personality.profile_path)
        )
        overrides_path.parent.mkdir(parents=True, exist_ok=True)
        header = (
            "# Written by the WITS web UI personality questionnaire (/personality).\n"
            "# Merged over config/wits_personality.yaml at load - delete this file\n"
            "# to revert to the base profile.\n"
        )
        overrides_path.write_text(
            header + yaml.safe_dump(overrides, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

        # Apply immediately: agents fetch the personality manager per call
        pm = reload_personality_manager(config=system.config)
        logger.info(f"Personality overrides saved to {overrides_path}")
        return {"saved": True, "system_prompt": pm.get_system_prompt()}

    @app.delete("/api/personality")
    async def reset_personality():
        from core.personality_manager import PersonalityManager, reload_personality_manager

        overrides_path = Path(
            PersonalityManager.overrides_path_for(system.config.personality.profile_path)
        )
        existed = overrides_path.exists()
        if existed:
            overrides_path.unlink()
        pm = reload_personality_manager(config=system.config)
        return {"reset": existed, "system_prompt": pm.get_system_prompt()}
