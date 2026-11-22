"""Tests for scripts/show_owner_token_qr.py."""

import importlib.util
from pathlib import Path

import pytest
import qrcode

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "show_owner_token_qr.py"
spec = importlib.util.spec_from_file_location("show_owner_token_qr", SCRIPT)
qr_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qr_script)

TOKEN = "s3cret-Token_value.123456789"


def write_env(root: Path, body: str) -> None:
    (root / ".env").write_text(body, encoding="utf-8")


def test_parse_token_handles_comments_quotes_and_duplicates(tmp_path):
    write_env(
        tmp_path,
        "# comment\n\nOTHER=1\nWITSV3_WEB_TOKEN=first\n  # WITSV3_WEB_TOKEN=nope\n"
        f'WITSV3_WEB_TOKEN = "  {TOKEN}  "\n',
    )
    assert qr_script.parse_token(tmp_path / ".env") == TOKEN


def test_parse_token_single_quotes(tmp_path):
    write_env(tmp_path, f"WITSV3_WEB_TOKEN='{TOKEN}'\n")
    assert qr_script.parse_token(tmp_path / ".env") == TOKEN


def test_missing_env_returns_one(tmp_path, capsys):
    assert qr_script.main(["--no-wait"], root=tmp_path) == 1
    out = capsys.readouterr().out
    assert "█" not in out and "▀" not in out and "▄" not in out


def test_empty_token_returns_one(tmp_path, capsys):
    write_env(tmp_path, "WITSV3_WEB_TOKEN=\nOTHER=abc123secret\n")
    assert qr_script.main(["--no-wait"], root=tmp_path) == 1
    out = capsys.readouterr().out
    assert "abc123secret" not in out and "█" not in out


def test_qr_shown_without_token_text(tmp_path, capsys):
    write_env(tmp_path, f"WITSV3_WEB_TOKEN={TOKEN}\n")
    assert qr_script.main(["--no-wait"], root=tmp_path) == 0
    out = capsys.readouterr().out
    assert any(ch in out for ch in "█▀▄")
    assert TOKEN not in out


def test_text_flag_prints_token(tmp_path, capsys):
    write_env(tmp_path, f"WITSV3_WEB_TOKEN={TOKEN}\n")
    assert qr_script.main(["--text", "--no-wait"], root=tmp_path) == 0
    assert TOKEN in capsys.readouterr().out


def test_qr_encodes_bare_token(tmp_path, capsys):
    write_env(tmp_path, f"WITSV3_WEB_TOKEN={TOKEN}\n")
    assert qr_script.main(["--no-wait"], root=tmp_path) == 0
    out = capsys.readouterr().out
    expected = qrcode.QRCode(border=2)
    expected.add_data(TOKEN)
    assert qr_script.render_qr(TOKEN, "utf-8") in out
    # A different payload (e.g. a URL) must draw a different matrix.
    other = qrcode.QRCode(border=2)
    other.add_data(f"http://localhost/?owner_token={TOKEN}")
    other.make()
    expected.make()
    assert expected.get_matrix() != other.get_matrix()


def test_ascii_fallback_when_encoding_cannot_show_blocks():
    art = qr_script.render_qr(TOKEN, "cp1252")
    assert "#" in art
    assert not any(ch in art for ch in "█▀▄")
    art.encode("cp1252")


def test_source_never_logs_or_writes_files():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "import logging" not in source
    assert "logging." not in source
    for mode in ('"w"', "'w'", '"a"', "'a'", '"wb"', "write_text", "write_bytes", "tempfile"):
        assert mode not in source


@pytest.mark.parametrize("flag", ["--no-wait"])
def test_default_root_is_script_parent(flag):
    assert qr_script.PROJECT_ROOT == SCRIPT.parent.parent
