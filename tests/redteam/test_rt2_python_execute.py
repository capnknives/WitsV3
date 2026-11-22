"""Red-team RT-2: python_execute sandbox escapes (modes off/subprocess/docker argv)."""

import socket
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.sandbox_runner import UNSANDBOXED_REFUSAL
from tools.python_execution_tool import PythonExecutionTool

SECRET = "RT-FAKE-SECRET-VALUE-12345"


def _cfg(mode, allow_unsandboxed=False):
    # Default opt-in is false, as in core/config.py.
    return SimpleNamespace(
        security=SimpleNamespace(
            sandbox_mode=mode,
            python_execution_network_access=False,
            python_execution_subprocess_access=False,
            python_execution_allow_unsandboxed=allow_unsandboxed,
        )
    )


def _patch_cfg(monkeypatch, cfg):
    # execute() imports core.config.load_config; the restricted-script builder
    # uses the module-level name, so both are patched.
    monkeypatch.setattr("core.config.load_config", lambda *a, **k: cfg)
    monkeypatch.setattr("tools.python_execution_tool.load_config", lambda *a, **k: cfg)


@pytest.fixture(params=["off", "subprocess"])
def tool(request, monkeypatch):
    cfg = _cfg(request.param)
    monkeypatch.setattr("core.config.load_config", lambda *a, **k: cfg)
    monkeypatch.setattr("tools.python_execution_tool.load_config", lambda *a, **k: cfg)
    return PythonExecutionTool()


@pytest.fixture
def off_tool(monkeypatch):
    cfg = _cfg("off")
    monkeypatch.setattr("core.config.load_config", lambda *a, **k: cfg)
    monkeypatch.setattr("tools.python_execution_tool.load_config", lambda *a, **k: cfg)
    return PythonExecutionTool()


async def _run(t, code):
    res = await t.execute(code=code, timeout=20)
    return (res.get("output") or "") + (res.get("error") or "")


async def test_a_cannot_read_secret_file(tool, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(f"TOKEN={SECRET}\n")
    out = await _run(tool, f"print(open({str(env_file)!r}).read())")
    assert SECRET not in out
    assert UNSANDBOXED_REFUSAL in out


async def test_b_env_secret_not_inherited(tool, monkeypatch):
    monkeypatch.setenv("WITSV3_WEB_TOKEN", SECRET)
    out = await _run(tool, "import os; print(os.environ.get('WITSV3_WEB_TOKEN'))")
    assert SECRET not in out


async def test_c_cannot_list_home(off_tool, tmp_path, monkeypatch):
    home = tmp_path / "fakehome"
    home.mkdir()
    (home / "rt_marker_file.txt").write_text("x")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    out = await _run(off_tool, f"import os; print(os.listdir({str(home)!r}))")
    assert "rt_marker_file" not in out
    assert UNSANDBOXED_REFUSAL in out


async def test_d_cannot_write_outside(tool, tmp_path):
    target = tmp_path / "outside.txt"
    out = await _run(tool, f"open({str(target)!r}, 'w').write('pwned')")
    assert not target.exists()
    assert UNSANDBOXED_REFUSAL in out


class _Listener:
    def __init__(self):
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(5)
        self.sock.settimeout(0.3)
        self.port = self.sock.getsockname()[1]
        self.hits = 0
        self._stop = False
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self):
        while not self._stop:
            try:
                c, _ = self.sock.accept()
                self.hits += 1
                c.close()
            except OSError:
                pass

    def close(self):
        self._stop = True
        self._t.join(2)
        self.sock.close()


@pytest.fixture
def listener():
    lst = _Listener()
    yield lst
    lst.close()


async def test_e_plain_socket_blocked(off_tool, listener):
    code = (
        "import socket\n"
        "s = socket.socket()\n"
        f"s.connect(('127.0.0.1', {listener.port}))\n"
        "print('connected')\n"
    )
    out = await _run(off_tool, code)
    assert listener.hits == 0
    assert UNSANDBOXED_REFUSAL in out


async def test_e_plain_socket_blocked_opt_in(monkeypatch, listener):
    # Opted-in host run: the restricted socket wrapper must still block it.
    _patch_cfg(monkeypatch, _cfg("off", allow_unsandboxed=True))
    code = (
        "import socket\n"
        "s = socket.socket()\n"
        f"s.connect(('127.0.0.1', {listener.port}))\n"
        "print('connected')\n"
    )
    out = await _run(PythonExecutionTool(), code)
    assert listener.hits == 0
    assert "Network access is not allowed" in out
    assert "connected" not in out


async def test_e_raw_socket_bypass(off_tool, listener):
    code = (
        "import _socket\n"
        "s = _socket.socket()\n"
        f"s.connect(('127.0.0.1', {listener.port}))\n"
        "print('connected')\n"
    )
    out = await _run(off_tool, code)
    assert listener.hits == 0
    assert UNSANDBOXED_REFUSAL in out


async def test_e_ctypes_bypass(off_tool, listener):
    code = (
        "import ctypes, ctypes.util, struct\n"
        "libc = ctypes.CDLL(None)\n"
        "fd = libc.socket(2, 1, 0)\n"
        f"addr = struct.pack('=H', 2) + struct.pack('>H', {listener.port}) + bytes([127,0,0,1]) + bytes(8)\n"
        "libc.connect(fd, addr, len(addr))\n"
        "print('connected')\n"
    )
    out = await _run(off_tool, code)
    assert listener.hits == 0
    assert UNSANDBOXED_REFUSAL in out


async def test_f_docker_argv_safe(monkeypatch, tmp_path):
    import core.docker_sandbox as ds
    import core.sandbox_runner as sr

    captured = {}

    class FakeProc:
        returncode = 0

        async def communicate(self):
            return b"ok", b""

    async def fake_exec(*argv, **kwargs):
        captured["argv"] = list(argv)
        return FakeProc()

    monkeypatch.setattr(ds, "_docker_exe", lambda: "docker")
    monkeypatch.setattr(ds, "_docker_env", lambda: {})
    monkeypatch.setattr(sr.asyncio, "create_subprocess_exec", fake_exec)
    await sr.run_python_sandboxed("print(1)", config=_cfg("docker"), timeout=5)

    argv = captured["argv"]
    assert "--network=none" in argv
    for flag in (
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit=128",
        "--memory=512m",
        "--memory-swap=512m",
        "--user=65534:65534",
        "--read-only",
        "--tmpfs=/tmp:rw,noexec,nosuid,size=64m",
    ):
        assert flag in argv
        assert argv.index(flag) < argv.index("witsv3-sandbox")
    assert argv[argv.index("-e") + 1] == "HOME=/tmp"
    assert argv.index("-e") < argv.index("witsv3-sandbox")
    assert "--privileged" not in argv
    assert argv.count("-v") == 1
    mount = argv[argv.index("-v") + 1]
    host, _, rest = mount.partition(":/sandbox")
    assert rest == ":ro"
    host_path = Path(host)
    assert "wits_docker_sandbox_" in host_path.name
    assert Path.home() not in host_path.parents and host_path != Path.home()
    assert Path(__file__).resolve().parents[2] not in [host_path, *host_path.parents]


def test_default_config_refuses_unsandboxed():
    from core.config import WitsV3Config

    assert WitsV3Config().security.python_execution_allow_unsandboxed is False


async def test_opt_in_off_mode_runs(monkeypatch):
    _patch_cfg(monkeypatch, _cfg("off", allow_unsandboxed=True))
    res = await PythonExecutionTool().execute(code="print(1+1)", timeout=20)
    assert res["success"] is True
    assert res["output"].strip() == "2"


@pytest.mark.parametrize("mode", ["off", "subprocess"])
async def test_opt_in_env_secret_not_inherited(monkeypatch, mode):
    _patch_cfg(monkeypatch, _cfg(mode, allow_unsandboxed=True))
    monkeypatch.setenv("RT_FAKE_SECRET_ENV", SECRET)
    code = "import os; print('ran', os.environ.get('RT_FAKE_SECRET_ENV'))"
    res = await PythonExecutionTool().execute(code=code, timeout=20)
    assert "ran None" in (res.get("output") or "")
    assert SECRET not in (res.get("output") or "") + (res.get("error") or "")


async def test_subprocess_runner_refuses_without_opt_in():
    import core.sandbox_runner as sr

    res = await sr.run_python_sandboxed("print(1)", config=_cfg("subprocess"), timeout=5)
    assert not res.success
    assert "python_execute is disabled" in res.error


async def test_unknown_mode_refused_even_with_opt_in(monkeypatch):
    _patch_cfg(monkeypatch, _cfg("of", allow_unsandboxed=True))
    res = await PythonExecutionTool().execute(code="print('ran')", timeout=20)
    assert res["success"] is False
    assert "unknown sandbox_mode" in res["error"]
    assert "ran" not in res["output"]


async def test_docker_start_failure_no_host_fallback(monkeypatch):
    import core.docker_sandbox as ds
    import core.sandbox_runner as sr

    async def boom(*argv, **kwargs):
        raise FileNotFoundError("docker")

    monkeypatch.setattr(ds, "_docker_exe", lambda: "docker")
    monkeypatch.setattr(ds, "_docker_env", lambda: {})
    monkeypatch.setattr(sr.asyncio, "create_subprocess_exec", boom)
    _patch_cfg(monkeypatch, _cfg("docker", allow_unsandboxed=True))
    res = await PythonExecutionTool().execute(code="print('ran')", timeout=5)
    assert res["success"] is False
    assert "docker failed to start" in res["error"]


def _docker_image_ready():
    import shutil
    import subprocess

    if not shutil.which("docker"):
        return False
    try:
        probe = ["docker", "image", "inspect", "witsv3-sandbox"]
        return subprocess.run(probe, capture_output=True, timeout=15).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


@pytest.mark.skipif(not _docker_image_ready(), reason="no docker daemon or witsv3-sandbox image")
async def test_real_docker_run_reads_script():
    import core.sandbox_runner as sr

    res = await sr.run_python_sandboxed(
        "import os; print('docker-ok', os.getuid())", config=_cfg("docker"), timeout=120
    )
    assert res.success, res.output
    assert "docker-ok 65534" in res.output


def test_minimal_child_env_temp_is_workdir(monkeypatch, tmp_path):
    from core.sandbox_runner import minimal_child_env

    for key in ("TEMP", "TMP", "TMPDIR"):
        monkeypatch.setenv(key, "/host/shared/tmp")
    env = minimal_child_env(str(tmp_path), {"PYTHONUNBUFFERED": "1"})
    assert all(env[k] == str(tmp_path) for k in ("TEMP", "TMP", "TMPDIR", "HOME"))
