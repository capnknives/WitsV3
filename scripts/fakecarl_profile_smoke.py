#!/usr/bin/env python3
"""Live FakeCarl smoke: register, share personal facts in chat, verify guest profile."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

_env_paths = [ROOT / ".env", ROOT.parent / "WitsV3" / ".env"]
for _env in _env_paths:
    if _env.is_file():
        for line in _env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

from httpx import ASGITransport, AsyncClient

from core.guest_access import GuestRegistry
from core.guest_user_profile import GuestUserProfileStore
from core.runtime_paths import data_dir, ensure_runtime_layout
from tests.web.test_web_server import FakeSystem, _parse_sse
from tools.guest_profile_tool import GuestUserProfileSummaryTool
from web.server import create_app

NAME = "FakeCarl"
DEVICE = f"fakecarl-smoke-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}"

# Self-disclosure turns — keywords chosen to match INTEREST_KEYWORDS + fact patterns.
INTRO_TURNS: list[tuple[str, list[str]]] = [
    (
        "Hi! I'm FakeCarl. I love playing guitar and I'm learning Python programming.",
        ["guitar", "python", "programming"],
    ),
    (
        "My favorite game is Minecraft survival mode — I build redstone farms every weekend.",
        ["minecraft", "redstone", "survival"],
    ),
    (
        "I'm really into science and reading sci-fi books before bed.",
        ["science", "reading", "books"],
    ),
    (
        "I play soccer after school and I like drawing anime characters in my sketchbook.",
        ["soccer", "drawing", "anime", "art"],
    ),
    (
        "I'm interested in coding robots with LEGO and I enjoy math homework when it's puzzles.",
        ["coding", "lego", "math"],
    ),
]

MIN_TURNS = len(INTRO_TURNS)
MIN_INTEREST_HITS = 4
MIN_FACTS = 2


def _load_env() -> tuple[str, str, str]:
    invite = os.getenv("WITSV3_GUEST_INVITE", "")
    secret = os.getenv("WITSV3_GUEST_SECRET", "")
    web_token = os.getenv("WITSV3_WEB_TOKEN", "")
    if not invite or not secret:
        raise SystemExit("FAIL: WITSV3_GUEST_INVITE and WITSV3_GUEST_SECRET required in .env")
    return invite, secret, web_token


def _interest_blob(profile: dict) -> str:
    interests = profile.get("interests") or {}
    return " ".join(interests.keys()).lower()


def _facts_blob(profile: dict) -> str:
    return " ".join((f.get("text") or "") for f in (profile.get("facts") or [])).lower()


def _count_keyword_hits(blob: str, keywords: list[str]) -> int:
    return sum(1 for kw in keywords if kw in blob)


def _validate_profile(profile: dict, owner_summary: str) -> list[str]:
    errors: list[str] = []
    turns = int(profile.get("turn_count", 0))
    if turns < MIN_TURNS:
        errors.append(f"turn_count={turns}, expected >={MIN_TURNS}")

    interests = profile.get("interests") or {}
    if len(interests) < MIN_INTEREST_HITS:
        errors.append(
            f"only {len(interests)} interests detected, expected >={MIN_INTEREST_HITS}: {list(interests)}"
        )

    facts = profile.get("facts") or []
    if len(facts) < MIN_FACTS:
        errors.append(f"only {len(facts)} facts, expected >={MIN_FACTS}")

    combined = f"{_interest_blob(profile)} {_facts_blob(profile)} {owner_summary.lower()}"
    expected_all = sorted({kw for _, kws in INTRO_TURNS for kw in kws})
    hits = [kw for kw in expected_all if kw in combined]
    if len(hits) < MIN_INTEREST_HITS:
        errors.append(
            f"profile missing expected topics (got {len(hits)}/{MIN_INTEREST_HITS}): {hits}"
        )

    if NAME.lower() not in owner_summary.lower():
        errors.append("owner summary does not mention FakeCarl")

    return errors


async def _guest_chat(client: AsyncClient, token: str, message: str, session_id: str) -> str:
    res = await client.post(
        "/api/chat",
        json={"message": message, "session_id": session_id},
        headers={"Authorization": f"Bearer {token}"},
    )
    if res.status_code != 200:
        raise RuntimeError(f"chat failed {res.status_code}: {res.text[:300]}")
    events = _parse_sse(res.text)
    return events[-1][1].get("final", "")


async def _run_inprocess(*, profile_llm: bool, settle_s: float) -> int:
    invite, _, web_token = _load_env()
    ensure_runtime_layout()
    data_dir().mkdir(parents=True, exist_ok=True)

    system = FakeSystem(ROOT)
    system.config.web_ui.guest_access.enabled = True
    system.config.web_ui.guest_access.audit_chat = True
    system.config.web_ui.guest_access.profile_llm_extraction = profile_llm

    app = create_app(system)
    transport = ASGITransport(app=app)

    print("=" * 72)
    print("  FAKECARL PROFILE LIVE SMOKE")
    print(f"  Runtime: {ROOT}")
    print(f"  Device:  {DEVICE}")
    print(f"  LLM extraction: {profile_llm}")
    print("=" * 72)

    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        reg = await client.post(
            "/api/guest/register",
            json={
                "invite_code": invite,
                "display_name": NAME,
                "device_id": DEVICE,
            },
        )
        if reg.status_code != 200:
            print(f"FAIL register: {reg.status_code} {reg.text}")
            return 1
        body = reg.json()
        token = body["guest_token"]
        guest_id = body["guest_id"]
        print(f"\nRegistered {NAME} guest_id={guest_id}\n")

        session_id = str(uuid.uuid4())
        for i, (message, keywords) in enumerate(INTRO_TURNS, 1):
            print(f"--- Turn {i} ---")
            print(f"  > {message[:72]}{'…' if len(message) > 72 else ''}")
            final = await _guest_chat(client, token, message, session_id)
            print(f"  < {final[:100]}{'…' if len(final) > 100 else ''}")
            print(f"  (expect keywords: {', '.join(keywords)})")

        print(f"\nWaiting {settle_s:.1f}s for profile tasks to finish…")
        await asyncio.sleep(settle_s)

        reg_check = GuestRegistry()
        acct = reg_check.find_by_display_name(NAME)
        if not acct:
            print("FAIL: FakeCarl not in guest registry")
            return 1

        store = GuestUserProfileStore()
        profile = store.load_merged_for_display_name(NAME, reg_check) or store.load(
            guest_id, NAME
        )
        profile_path = store._path(profile.get("guest_id", guest_id))
        print(f"\n--- Profile file ---\n  {profile_path}")

        tool = GuestUserProfileSummaryTool()
        owner_summary = await tool.execute(display_name=NAME, user_role="owner")

        print("\n--- Owner profile summary ---")
        print(owner_summary)

        print("\n--- Profile JSON (interests + facts) ---")
        print(json.dumps(
            {
                "turn_count": profile.get("turn_count"),
                "interests": profile.get("interests"),
                "facts": profile.get("facts"),
            },
            indent=2,
            ensure_ascii=False,
        ))

        errors = _validate_profile(profile, owner_summary)
        if errors:
            print("\nFAIL profile validation:")
            for err in errors:
                print(f"  - {err}")
            return 1

        owner_res = await client.get(
            "/api/guest/admin/profile",
            params={"display_name": NAME},
            headers={"Authorization": f"Bearer {web_token}"},
        )
        if owner_res.status_code != 200:
            print(f"WARN owner profile API: {owner_res.status_code}")

        report = {
            "guest_id": guest_id,
            "display_name": NAME,
            "device_id": DEVICE,
            "turns_sent": len(INTRO_TURNS),
            "profile_path": str(profile_path),
            "profile": profile,
            "owner_summary": owner_summary,
            "validated_at": datetime.now(timezone.utc).isoformat(),
        }
        report_path = data_dir() / "fakecarl_profile_smoke_report.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nReport saved: {report_path}")
        print("\n=== FAKECARL PROFILE SMOKE PASSED ===")
        return 0


async def _run_live_server(base_url: str, *, profile_llm: bool, settle_s: float) -> int:
    """Optional: hit a running WitsV3 Web UI (python run_web.py)."""
    invite, _, web_token = _load_env()
    print("=" * 72)
    print("  FAKECARL PROFILE LIVE SMOKE (remote server)")
    print(f"  URL: {base_url}")
    print("=" * 72)

    async with AsyncClient(base_url=base_url.rstrip("/"), timeout=120.0) as client:
        reg = await client.post(
            "/api/guest/register",
            json={
                "invite_code": invite,
                "display_name": NAME,
                "device_id": DEVICE,
            },
        )
        if reg.status_code != 200:
            print(f"FAIL register: {reg.status_code} {reg.text}")
            return 1
        token = reg.json()["guest_token"]
        guest_id = reg.json()["guest_id"]
        print(f"Registered {NAME} guest_id={guest_id}")

        session_id = str(uuid.uuid4())
        for i, (message, _) in enumerate(INTRO_TURNS, 1):
            print(f"--- Turn {i} ---")
            final = await _guest_chat(client, token, message, session_id)
            print(f"  < {final[:80]}…")

        await asyncio.sleep(settle_s)

        prof = await client.get(
            "/api/guest/admin/profile",
            params={"display_name": NAME},
            headers={"Authorization": f"Bearer {web_token}"},
        )
        if prof.status_code != 200:
            print(f"FAIL owner profile API: {prof.status_code} {prof.text}")
            return 1
        payload = prof.json()
        profile = payload.get("profile") or {}
        owner_summary = payload.get("summary") or ""
        print("\n--- Owner profile summary ---")
        print(owner_summary)

        errors = _validate_profile(profile, owner_summary)
        if errors:
            print("\nFAIL profile validation:")
            for err in errors:
                print(f"  - {err}")
            return 1

        print("\n=== FAKECARL PROFILE SMOKE PASSED ===")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="FakeCarl guest profile live smoke test")
    parser.add_argument(
        "--url",
        default="",
        help="Live server base URL (e.g. http://localhost:8000). Default: in-process app.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM profile extraction (heuristic-only, faster/deterministic).",
    )
    parser.add_argument(
        "--settle",
        type=float,
        default=3.0,
        help="Seconds to wait after chats for async profile updates (default 3).",
    )
    args = parser.parse_args()

    profile_llm = not args.no_llm
    if args.url:
        return asyncio.run(
            _run_live_server(args.url, profile_llm=profile_llm, settle_s=args.settle)
        )
    return asyncio.run(
        _run_inprocess(profile_llm=profile_llm, settle_s=args.settle)
    )


if __name__ == "__main__":
    raise SystemExit(main())
