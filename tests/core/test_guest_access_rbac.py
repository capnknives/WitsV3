"""RBAC role tool matrix tests."""

from core.guest_access import is_tool_allowed, resolve_effective_role, tools_for_role


def test_owner_has_all_tools():
    assert tools_for_role("owner") is None
    assert is_tool_allowed("write_file", "owner")


def test_family_kid_cannot_write():
    allowed = tools_for_role("family_kid")
    assert allowed is not None
    assert "read_file" in allowed
    assert "write_file" not in allowed


def test_family_adult_has_downloads_tools():
    allowed = tools_for_role("family_adult")
    assert allowed is not None
    assert "read_file" in allowed
    assert "list_directory" in allowed


def test_resolve_effective_role_from_age_band():
    assert resolve_effective_role("guest", {"age_band": "adult"}) == "family_adult"
    assert resolve_effective_role("guest", {"age_band": "child"}) == "family_kid"
    assert resolve_effective_role("owner", None) == "owner"
