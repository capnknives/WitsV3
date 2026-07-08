#!/usr/bin/env python3
"""Short TESTER smoke: 2 chats + verify [TESTER] appears in server logs."""

from __future__ import annotations

import logging
import os
import sys
import uuid
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
        break

from fastapi.testclient import TestClient

from tests.web.test_web_server import FakeSystem, _parse_sse
from web.server import create_app

NAME = "TESTER"
DEVICE = f"label-smoke-{uuid.uuid4().hex[:8]}"


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record.getMessage())


def main() -> int:
    invite = os.getenv("WITSV3_GUEST_INVITE", "")
    secret = os.getenv("WITSV3_GUEST_SECRET", "")
    if not invite or not secret:
        print("FAIL: WITSV3_GUEST_INVITE and WITSV3_GUEST_SECRET required")
        return 1

    (ROOT / "data").mkdir(exist_ok=True)
    system = FakeSystem(ROOT)
    system.config.web_ui.guest_access.enabled = True
    system.config.web_ui.guest_access.audit_chat = True
    system.config.web_ui.guest_access.content_policy_enabled = True

    capture = _Capture()
    for name in ("uvicorn.access", "WitsV3.WebUI"):
        log = logging.getLogger(name)
        log.addHandler(capture)
        log.setLevel(logging.INFO)

    client = TestClient(create_app(system))
    reg = client.post(
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
    print(f"Registered {NAME}")

    for msg in ("what is 2+2?", "I love Minecraft redstone"):
        res = client.post(
            "/api/chat",
            json={"message": msg, "session_id": str(uuid.uuid4())},
            headers={"Authorization": f"Bearer {token}"},
        )
        if res.status_code != 200:
            print(f"FAIL chat: {res.status_code}")
            return 1
        final = _parse_sse(res.text)[-1][1].get("final", "")
        print(f"  > {msg[:50]}")
        print(f"    {final[:100]}{'…' if len(final) > 100 else ''}")

    blob = "\n".join(capture.records)
    if f"[{NAME}]" not in blob:
        print("FAIL: no [TESTER] label in captured logs")
        print("--- log sample ---")
        print(blob[-2000:] if blob else "(empty)")
        return 1

    hits = [line for line in capture.records if f"[{NAME}]" in line]
    print(f"\nOK found {len(hits)} log line(s) with [{NAME}]:")
    for line in hits[:6]:
        print(f"  {line}")
    print("\n=== LABEL SMOKE PASSED ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
