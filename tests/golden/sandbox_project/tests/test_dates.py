import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from parser import parse_date  # noqa: E402


def test_parse_date():
    assert parse_date("2026-11-22") == date(2026, 11, 22)
