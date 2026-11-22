"""Acceptance tests for core.path_access (folder-access contract sections 3, 4 and 6)."""

import logging
from types import SimpleNamespace

import pytest

import core.filesystem_policy as fp
import core.path_access as pa
from core import runtime_paths
from core.path_access import Grant, Roots, check, decide

FAR = 4_000_000_000  # an expiry far in the future
WIN = Roots(
    inbox=r"D:\WITS\inbox",
    outbox=r"D:\WITS\outbox",
    projects=r"D:\WITS\projects",
    repo=r"C:\Code\WitsV3",
    home=r"C:\Users\andy",
    windows=True,
)
POSIX = Roots(
    inbox="/srv/wits/inbox",
    outbox="/srv/wits/outbox",
    projects="/srv/wits/projects",
    repo="/opt/witsv3",
    home="/home/andy",
)
ALL_WIN = [Grant("C:\\", "read_write", FAR), Grant("D:\\", "index", FAR)]
ALL_POSIX = [Grant("/", "read_write", FAR)]


def _win(path, mode="read", role="owner", grants=(), roots=WIN):
    return decide(path, mode, role, grants, roots, now=1000)


def _posix(path, mode="read", role="owner", grants=()):
    return decide(path, mode, role, grants, POSIX, now=1000)


# ------------------------------------------------------------ section 3, in order


def test_rule1_never_beats_everything_and_is_not_requestable():
    d = _win(r"D:\WITS\inbox\.env", grants=ALL_WIN)
    assert not d.allowed and not d.requestable


def test_rule2_guest_keeps_role_roots_and_cannot_request():
    guest = Roots(**{**WIN.__dict__, "role_roots": (r"C:\Code\WitsV3",)})
    assert _win(r"C:\Code\WitsV3\docs\a.md", role="guest", roots=guest).allowed
    for path, mode in [(r"D:\WITS\inbox\a.txt", "read"), (r"C:\Code\WitsV3\a.md", "write")]:
        d = _win(path, mode, role="guest", grants=ALL_WIN, roots=guest)
        assert not d.allowed and not d.requestable


@pytest.mark.parametrize(
    "folder,allowed",
    [
        ("inbox", {"read", "index"}),
        ("outbox", {"read", "write"}),
        ("projects", {"read", "write"}),
    ],
)
def test_rule3_base_folders_follow_the_table(folder, allowed):
    for mode in ("read", "write", "index"):
        assert _win(rf"D:\WITS\{folder}\notes.txt", mode).allowed is (mode in allowed)
        assert _posix(f"/srv/wits/{folder}/notes.txt", mode).allowed is (mode in allowed)


def test_rule4_repo_is_read_only_here():
    assert _posix("/opt/witsv3/core/config.py").allowed
    for mode in ("write", "index"):
        d = _posix("/opt/witsv3/core/config.py", mode, grants=ALL_POSIX)
        assert not d.allowed and not d.requestable


@pytest.mark.parametrize(
    "grant_mode,allowed",
    [
        ("read", {"read"}),
        ("read_write", {"read", "write"}),
        ("index", {"read", "index"}),
    ],
)
def test_rule5_live_grant_allows_its_modes(grant_mode, allowed):
    grants = [Grant(r"E:\Photos", grant_mode, FAR)]
    for mode in ("read", "write", "index"):
        assert _win(r"E:\Photos\2026\a.txt", mode, grants=grants).allowed is (mode in allowed)


def test_rule5_expired_or_revoked_grant_allows_nothing():
    for grant in (Grant("/data", "read", 999), Grant("/data", "read", FAR, revoked_at=5)):
        assert not _posix("/data/a.txt", grants=[grant]).allowed


def test_rule6_everything_else_is_requestable():
    d = _posix("/data/a.txt")
    assert not d.allowed and d.requestable


# ------------------------------------------------------------ section 4 bullets


@pytest.mark.parametrize(
    "path",
    [
        "D:\\",
        r"C:\Windows\System32\drivers\etc\hosts",
        r"C:\Program Files\App\a.txt",
        r"C:\Program Files (x86)\App\a.txt",
        r"C:\Users\andy",
        r"C:\Users\x",
        r"C:\Users\andy\AppData\Local\a.txt",
        r"C:\Users\x\AppData\Roaming\a.txt",
    ],
)
def test_never_system_and_profile_roots_on_windows(path):
    for mode in ("read", "write", "index"):
        d = _win(path, mode, grants=ALL_WIN)
        assert not d.allowed and not d.requestable


@pytest.mark.parametrize("path", ["/", "/home/andy", "/home/other", "/root"])
def test_never_home_and_filesystem_root_on_posix(path):
    assert not _posix(path, grants=ALL_POSIX).allowed


@pytest.mark.parametrize("name", [".ssh", ".gnupg", ".aws", ".azure", ".kube"])
def test_never_protected_folders_on_any_component(name):
    for mode in ("read", "write", "index"):
        assert not _posix(f"/data/{name}/config", mode, grants=ALL_POSIX).allowed
        assert not _win(rf"E:\Backup\{name.upper()}\x", mode, grants=ALL_WIN).allowed


def test_never_git_for_write_and_index_but_repo_git_reads():
    assert _posix("/opt/witsv3/.git/HEAD").allowed
    for path in ("/opt/witsv3/.git/HEAD", "/srv/wits/projects/app/.git/config"):
        for mode in ("write", "index"):
            d = _posix(path, mode, grants=ALL_POSIX)
            assert not d.allowed and not d.requestable


@pytest.mark.parametrize(
    "path",
    [
        r"E:\Chrome\User Data\Default\Cookies",
        "/data/.mozilla/firefox/abc/places.sqlite",
        r"E:\Mozilla\Firefox\Profiles\abc\logins.json",
    ],
)
def test_never_browser_profiles(path):
    flavour = _win if path.startswith("E:") else _posix
    assert not flavour(path, grants=ALL_WIN + ALL_POSIX).allowed


def test_profiles_outside_a_browser_vendor_is_not_never():
    assert _posix("/data/Profiles/a.txt", grants=ALL_POSIX).allowed


@pytest.mark.parametrize(
    "name",
    [
        ".env",
        ".env.local",
        "server.pem",
        "site.key",
        "cert.pfx",
        "cert.p12",
        "vault.kdbx",
        "id_rsa",
        "id_rsa.pub",
        "id_ed25519",
        "client_secret_123.json",
        "gcp-credentials-prod.json",
        "token.json",
        "release.keystore",
        "secrets.yaml",
    ],
)
def test_never_secret_file_names_even_in_base_and_repo(name):
    for path in (f"/srv/wits/inbox/{name}", f"/opt/witsv3/{name}", f"/data/{name}"):
        d = _posix(path, grants=ALL_POSIX)
        assert not d.allowed and not d.requestable


def test_case_differences_on_windows_cannot_dodge_never_rules():
    for path in (
        r"D:\Downloads\.ENV",
        r"d:\wits\INBOX\Client_Secret_X.JSON",
        r"c:\users\ANDY\appdata\local\x.txt",
        r"c:\WINDOWS\win.ini",
        r"e:\backup\.SSH\id",
    ):
        assert not _win(path, grants=ALL_WIN).allowed, path
    assert _win(r"d:\wits\INBOX\notes.txt").allowed  # base match is case-insensitive too


def test_lexical_dotdot_and_relative_paths_are_refused():
    assert not _win(r"D:\WITS\inbox\..\..\Secret\x.txt").allowed
    assert not _posix("/srv/wits/outbox/../../../etc/passwd", "write").allowed
    d = _posix("notes.txt")
    assert not d.allowed and not d.requestable


# ------------------------------------------------------------ check() on disk


@pytest.fixture
def env(tmp_path, monkeypatch):
    base, repo, outside = tmp_path / "base", tmp_path / "repo", tmp_path / "outside"
    for folder in (base / "inbox", base / "outbox", base / "projects", repo, outside):
        folder.mkdir(parents=True)
    monkeypatch.setattr(fp, "PROJECT_ROOT", repo)
    monkeypatch.setattr(runtime_paths, "PROJECT_ROOT", repo)
    config = SimpleNamespace(folders=SimpleNamespace(base=str(base)), security=None)
    monkeypatch.setattr("core.config.load_config", lambda *a, **k: config)
    return SimpleNamespace(base=base, repo=repo, outside=outside, config=config)


async def test_symlink_escape_from_base_is_refused(env):
    (env.outside / "secret.txt").write_text("x")
    (env.base / "inbox" / "link.txt").symlink_to(env.outside / "secret.txt")
    (env.base / "inbox" / "dir").symlink_to(env.outside, target_is_directory=True)
    for path in (env.base / "inbox" / "link.txt", env.base / "inbox" / "dir" / "secret.txt"):
        with pytest.raises(PermissionError):
            await check(path, "read", config=env.config)


async def test_dotdot_escape_from_base_is_refused(env):
    (env.outside / "x.txt").write_text("x")
    with pytest.raises(PermissionError):
        await check(env.base / "outbox" / ".." / ".." / "outside" / "x.txt", "read")


async def test_secret_files_refused_for_read(env):
    (env.repo / ".env").write_text("KEY=1")
    (env.base / "inbox" / "client_secret_x.json").write_text("{}")
    from tools.file_tools import FileReadTool

    for path in (env.repo / ".env", env.base / "inbox" / "client_secret_x.json"):
        with pytest.raises(PermissionError):
            await check(path, "read", config=env.config)
        assert (await FileReadTool().execute(file_path=str(path))).startswith("Error:")


async def test_grants_provider_is_injectable_and_refusals_are_audited(env, monkeypatch, caplog):
    target = env.outside / "a.txt"
    target.write_text("x")
    caplog.set_level(logging.WARNING, logger="WitsV3.PathAccess")
    with pytest.raises(PermissionError):
        await check(target, "read", config=env.config)
    assert any("refused read" in r.getMessage() for r in caplog.records)

    caplog.clear()
    with pytest.raises(PermissionError):
        await check(env.base / "inbox" / ".env", "read", config=env.config)
    assert not caplog.records  # inside a base folder: no audit line

    async def grants():
        return [Grant(str(env.outside), "read", FAR)]

    monkeypatch.setattr(pa, "grants_provider", grants)
    assert await check(target, "read", config=env.config) == target.resolve()


async def test_write_file_refuses_repo_and_writes_outbox_and_projects(env):
    from tools.file_tools import FileWriteTool

    tool = FileWriteTool()
    result = await tool.execute(file_path=str(env.repo / "core.py"), content="x")
    assert result.startswith("Error:") and not (env.repo / "core.py").exists()

    assert "Successfully" in await tool.execute(file_path="notes/a.txt", content="p")
    assert (env.base / "projects" / "notes" / "a.txt").read_text() == "p"
    assert "Successfully" in await tool.execute(
        file_path=str(env.base / "outbox/b.txt"), content="o"
    )
    assert (env.base / "outbox" / "b.txt").read_text() == "o"


class _EmbedLLM:
    async def get_embedding(self, text, model=None):
        return [1.0] + [0.0] * 383


async def test_indexer_skips_and_counts_protected_files(env, tmp_path):
    from core.config import WitsV3Config
    from core.memory_manager import MemoryManager
    from tools.document_tools import DocumentIngestTool

    corpus = env.repo / "var" / "user_files"
    corpus.mkdir(parents=True)
    (corpus / ".env").write_text("KEY=1")
    (corpus / "notes.md").write_text("hello notes")
    (env.base / "inbox" / "client_secret_x.json").write_text("{}")
    (env.outside / "free.md").write_text("not granted")

    config = WitsV3Config()
    config.folders.base = str(env.base)
    config.memory_manager.backend = "basic"
    config.memory_manager.memory_file_path = str(tmp_path / "memory.json")
    config.document_rag.documents_path = str(corpus)
    config.document_rag.file_extensions = [".md", ".json", ".env", ""]
    config.security.document_ingest_roots = [str(env.outside)]
    tool = DocumentIngestTool()
    tool.set_dependencies(
        config, _EmbedLLM(), MemoryManager(config=config, llm_interface=_EmbedLLM())
    )

    summary = await tool.execute()
    assert summary["files_skipped_protected"] == 2
    assert set(summary["searchable_files"]) == {"notes.md"}  # ungranted root not indexed
    assert "skipped 2 protected file(s)" in summary["message"]


# ------------------------------------------------------------ review fixes


@pytest.mark.parametrize(
    "path,mode",
    [
        ("D:\\WITS\\inbox\\.env ", "read"),
        (r"D:\WITS\inbox\.env::$DATA", "read"),
        (r"D:\WITS\inbox\.env:x", "read"),
        ("D:\\WITS\\inbox\\server.pem.", "read"),
        (r"D:\WITS\projects\.git.\config", "write"),
        (r"D:\WITS\projects\.git::$INDEX_ALLOCATION\config", "write"),
        (r"D:\WITS\outbox\NUL", "write"),
        (r"D:\WITS\outbox\com1.txt", "write"),
        (r"D:\WITS\inbox\lpt9", "read"),
    ],
)
def test_windows_alias_names_are_never(path, mode):
    d = _win(path, mode, grants=ALL_WIN)
    assert not d.allowed and not d.requestable


@pytest.mark.parametrize(
    "path", [r"E:\x\.ssh.\id", r"C:\Windows.\win.ini", r"C:\Users\andy\AppData.\Local\x"]
)
def test_windows_alias_names_do_not_dodge_never_even_with_grant(path):
    grants = ALL_WIN + [Grant("E:\\", "read_write", FAR)]
    assert not _win(path, grants=grants).allowed


@pytest.mark.parametrize(
    "path",
    [
        r"\\server\share\a.txt",
        "//server/share/a.txt",
        r"\\?\D:\WITS\inbox\a.txt",
        "//?/D:/WITS/inbox/a.txt",
        r"\\.\PhysicalDrive0",
        "//./PhysicalDrive0",
    ],
)
def test_unc_and_device_paths_are_never(path):
    d = _win(path, grants=ALL_WIN)
    assert not d.allowed and not d.requestable
    assert not _posix(path.replace("\\", "/"), grants=ALL_POSIX).allowed


def test_relative_grant_or_role_root_matches_nothing():
    for root in (".", "", "x"):
        assert not _win(
            r"E:\private\a.txt", "write", grants=[Grant(root, "read_write", FAR)]
        ).allowed
        assert not _posix(
            "/private/a.txt", "write", grants=[Grant(root, "read_write", FAR)]
        ).allowed
    guest = Roots(**{**POSIX.__dict__, "role_roots": (".",)})
    assert not decide("/private/a.txt", "read", "guest", (), guest).allowed


def test_only_exact_owner_string_is_owner():
    guest = Roots(**{**POSIX.__dict__, "role_roots": ("/opt/witsv3",)})
    for role in ("", None, "Owner", " owner", "family_adult"):
        assert not decide("/srv/wits/outbox/a.txt", "write", role, (), guest).allowed
        assert not decide("/srv/wits/inbox/a.txt", "read", role, (), guest).allowed
    assert decide("/srv/wits/outbox/a.txt", "write", "owner", (), guest).allowed


async def test_malformed_input_raises_only_permission_error(env, monkeypatch, caplog):
    for bad in ("/tmp/a\x00b", "~nosuchuser_wits_xyz/x"):
        with pytest.raises(PermissionError):
            await check(bad, "read", config=env.config)

    async def bad_grants():
        return [Grant(str(env.outside), "read", None)]

    (env.outside / "a.txt").write_text("x")
    monkeypatch.setattr(pa, "grants_provider", bad_grants)
    caplog.set_level(logging.WARNING, logger="WitsV3.PathAccess")
    for _ in range(2):
        with pytest.raises(PermissionError):
            await check(env.outside / "a.txt", "read", config=env.config)
    assert sum("malformed grant" in r.getMessage() for r in caplog.records) == 1


async def test_supplied_alias_spelling_refused_before_resolution(env, monkeypatch):
    monkeypatch.setattr(pa.os, "name", "nt")
    with pytest.raises(PermissionError):
        pa.enforce("D:\\WITS\\inbox\\.env.", "read", config=env.config)


async def test_relative_base_and_target_anchor_to_project_root_not_cwd(env, tmp_path, monkeypatch):
    elsewhere = tmp_path / "cwd"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.delenv("WITSV3_PROJECT_ROOT", raising=False)
    config = SimpleNamespace(folders=SimpleNamespace(base="var/wits_files"), security=None)
    target = pa.resolve_write_path("a.txt", config)
    assert target == env.repo / "var" / "wits_files" / "projects" / "a.txt"
    assert await check(env.repo / "var/wits_files/outbox/b.txt", "write", config=config)
    with pytest.raises(PermissionError):
        await check(elsewhere / "var/wits_files/outbox/b.txt", "write", config=config)


async def test_corpus_exception_only_inside_repo_var(env, tmp_path):
    from core.config import WitsV3Config
    from core.memory_manager import MemoryManager
    from tools.document_tools import DocumentIngestTool

    corpus = tmp_path / "elsewhere_docs"
    corpus.mkdir()
    (corpus / "notes.md").write_text("hello")
    config = WitsV3Config()
    config.folders.base = str(env.base)
    config.memory_manager.backend = "basic"
    config.memory_manager.memory_file_path = str(tmp_path / "memory.json")
    config.document_rag.documents_path = str(corpus)
    config.security.document_ingest_roots = []
    tool = DocumentIngestTool()
    tool.set_dependencies(
        config, _EmbedLLM(), MemoryManager(config=config, llm_interface=_EmbedLLM())
    )
    summary = await tool.execute()
    assert summary["searchable_files"] == {} and summary["files_skipped_protected"] == 1


def test_write_verification_uses_the_path_write_file_returned(tmp_path):
    from agents.orchestrator_tool_helpers import OrchestratorToolHelpersMixin as M

    written = tmp_path / "chat.txt"
    written.write_text("hello world")
    obs = f"Tool write_file result: Successfully written to file: {written}"
    assert M._write_result_verified("chat.txt", obs, min_bytes=5)
    assert not M._write_result_verified(
        "chat.txt", "Tool write_file result: Successfully", min_bytes=1
    )
    assert not M._write_result_verified("other.txt", obs, min_bytes=5)


def test_runtime_paths_and_check_share_one_anchor_not_cwd(env, tmp_path, monkeypatch):
    elsewhere = tmp_path / "cwd"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.delenv("WITSV3_PROJECT_ROOT", raising=False)
    config = SimpleNamespace(folders=SimpleNamespace(base="var/wits_files"), security=None)
    assert str(runtime_paths.inbox_dir("var/wits_files")) == pa.roots_for(config).inbox
    created = runtime_paths.ensure_base_folders("var/wits_files")
    assert created[0] == env.repo / "var" / "wits_files" / "inbox" and created[0].is_dir()
    assert not (elsewhere / "var").exists()
    monkeypatch.setenv("WITSV3_PROJECT_ROOT", str(tmp_path / "pinned"))
    assert runtime_paths.outbox_dir("rel") == tmp_path / "pinned" / "rel" / "outbox"
    assert runtime_paths.outbox_dir(str(tmp_path / "abs")) == tmp_path / "abs" / "outbox"


@pytest.mark.parametrize("name", [".env", "id_rsa", "client_secret_x.json", "secrets.txt"])
def test_is_never_answers_section_4_alone(tmp_path, name):
    assert pa.is_never(tmp_path / name, "index")
    assert pa.is_never(tmp_path / ".ssh" / "notes.txt", "read")
    assert not pa.is_never(tmp_path / "notes.txt", "write")
