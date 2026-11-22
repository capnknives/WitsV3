"""Static checks for the slash menu in web/static/app.js (no JS test runner in this repo)."""

import re
from pathlib import Path

APP_JS = Path(__file__).resolve().parents[2] / "web" / "static" / "app.js"


def _source() -> str:
    return APP_JS.read_text(encoding="utf-8")


def test_token_form_submit_reloads_slash_commands():
    src = _source()
    start = src.index('$("#token-form").addEventListener("submit"')
    # Region ends at the next top-level comment banner, function or listener.
    nxt = re.search(
        r"^(?:/\* -|async function |function |\$\(|\w+\.addEventListener)", src[start + 10 :], re.M
    )
    end = start + 10 + nxt.start() if nxt else len(src)
    assert "loadSlashCommands(" in src[start:end]


def test_empty_list_message_present():
    assert "Commands could not be loaded. Check your sign-in." in _source()


def test_no_matching_commands_message_kept():
    assert "No matching commands" in _source()
