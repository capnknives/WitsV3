"""Tests for SEARCH/REPLACE patch helpers in the verified-edit pipeline."""

from core.safe_code_editor import apply_search_replace, extract_search_replace_blocks


def test_extract_search_replace_blocks_parses_hunks():
    response = """<<<<<<< SEARCH
def old():
    pass
=======
def new():
    return 1
>>>>>>> REPLACE"""
    blocks = extract_search_replace_blocks(response)
    assert len(blocks) == 1
    assert "def old" in blocks[0][0]
    assert "def new" in blocks[0][1]


def test_apply_search_replace_applies_in_order():
    original = "alpha\nbeta\ngamma\n"
    blocks = [("beta", "BETA"), ("alpha", "ALPHA")]
    result = apply_search_replace(original, blocks)
    assert result == "ALPHA\nBETA\ngamma\n"


def test_apply_search_replace_returns_none_on_miss():
    assert apply_search_replace("hello", [("missing", "x")]) is None
