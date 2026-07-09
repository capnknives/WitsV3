"""Tests for web chat slash commands."""

from web.slash_commands import list_slash_commands

pytest_plugins = ("tests.web.test_web_server", "tests.web.test_guest_access")


def test_list_slash_commands_owner_includes_shutdown():
    cmds = list_slash_commands("owner")
    by_id = {c["id"]: c for c in cmds}
    assert "shutdown" in by_id
    assert "/stop" in by_id["shutdown"]["aliases"]
    assert by_id["shutdown"]["dispatch"] == "chat"
    assert "new" in by_id
    assert by_id["new"]["dispatch"] == "client"


def test_list_slash_commands_guest_hides_owner_only():
    cmds = list_slash_commands("guest")
    ids = {c["id"] for c in cmds}
    assert "new" in ids
    assert "export" in ids
    assert "shutdown" not in ids
    assert "memory" not in ids
    assert "settings" not in ids


def test_api_commands_owner(client_auth):
    client, _ = client_auth
    body = client.get("/api/commands", headers={"Authorization": "Bearer sekrit"}).json()
    ids = {c["id"] for c in body["commands"]}
    assert "help" in ids
    assert "restart" in ids
    assert all("description" in c for c in body["commands"])


def test_api_commands_noauth(client_noauth):
    client, _ = client_noauth
    res = client.get("/api/commands")
    assert res.status_code == 200
    assert isinstance(res.json()["commands"], list)


def test_api_commands_guest(guest_env):
    client, _ = guest_env
    from tests.web.test_guest_access import _register

    token = _register(client).json()["guest_token"]
    body = client.get(
        "/api/commands",
        headers={"Authorization": f"Bearer {token}"},
    ).json()
    ids = {c["id"] for c in body["commands"]}
    assert "tools" in ids
    assert "shutdown" not in ids
