"""RT-6: the never-list is name-based; secrets under other names or content slip through."""

import pytest

from core.path_access import IndexGate, Roots, decide

WIN = Roots(
    inbox=r"D:\WITS\inbox",
    documents=r"D:\WITS\documents",
    outbox=r"D:\WITS\outbox",
    projects=r"D:\WITS\projects",
    repo=r"C:\Code\WitsV3",
    home=r"C:\Users\capta",
    windows=True,
)
POSIX = Roots(
    inbox="/srv/wits/inbox",
    documents="/srv/wits/documents",
    outbox="/srv/wits/outbox",
    projects="/srv/wits/projects",
    repo="/opt/witsv3",
    home="/home/andy",
)

NAMES = ["passwords.txt", ".netrc", "key.ppk", "service-account.json"]


@pytest.mark.xfail(strict=True, reason="RT-6: secret-bearing names are not in the never-list")
@pytest.mark.parametrize("mode", ["index", "read"])
@pytest.mark.parametrize("name", NAMES)
def test_secret_names_in_inbox_refused(name, mode):
    d = decide(f"/srv/wits/inbox/{name}", mode, "owner", [], POSIX, now=1000)
    assert not d.allowed


@pytest.mark.xfail(strict=True, reason="RT-6: var/data/workspace.db in repo is readable by owner")
def test_workspace_db_read_refused():
    d = decide("/opt/witsv3/var/data/workspace.db", "read", "owner", [], POSIX, now=1000)
    assert not d.allowed


@pytest.mark.xfail(
    strict=True, reason="RT-6: private-key content under an innocent name is indexed"
)
def test_private_key_content_not_indexed(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    f = inbox / "notes.txt"
    f.write_text("-----BEGIN OPENSSH PRIVATE KEY-----\nabc\n")
    roots = Roots(
        inbox=str(inbox.resolve()),
        documents=str(tmp_path / "d"),
        outbox=str(tmp_path / "o"),
        projects=str(tmp_path / "p"),
        repo=str(tmp_path / "r"),
        home=str(tmp_path / "h"),
    )
    assert not IndexGate(roots, []).allows_file(f)


# Pure decision only: on a non-Windows host this does not exercise Path.resolve() on 8.3 names.
@pytest.mark.parametrize("path", [r"C:\PROGRA~1\x.txt", r"C:\Users\capta\APPDAT~1\x"])
def test_windows_short_names_refused(path):
    d = decide(path, "read", "owner", [], WIN, now=1000)
    assert not d.allowed
