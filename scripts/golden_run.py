"""Run the golden turn set and write a results table.

Golden cases exercise file tools and ingest, so by default they run in a sandbox: every
runtime path and read/ingest root points inside DIR and the runner refuses to start if any
resolved path is outside it. Running against real data needs both live flags.

Usage:
    python scripts/golden_run.py --sandbox D:/golden-sandbox --runs 3 --tag chat
    python scripts/golden_run.py --live --i-understand-this-touches-real-data
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import inspect
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path, PurePosixPath, PureWindowsPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.golden import GoldenCase, evaluate, load_cases, tools_seen  # noqa: E402

try:  # C1a may not be merged yet
    from core import turn_metrics
except ImportError:
    turn_metrics = None


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    p = argparse.ArgumentParser(description="Run the WitsV3 golden turn set.")
    p.add_argument("--runs", type=int, default=1, help="runs per case (default 1)")
    p.add_argument("--only", default="", help="comma-separated case ids")
    p.add_argument("--tag", default="", help="only cases with this tag")
    p.add_argument("--cases", default=str(ROOT / "tests/golden/turns.yaml"))
    p.add_argument("--out", default=str(ROOT / "docs/benchmarks/golden.md"))
    p.add_argument("--sandbox", default="", help="DIR that holds all data for this run")
    p.add_argument("--live", action="store_true", help="run against real data (needs ack flag)")
    p.add_argument("--i-understand-this-touches-real-data", action="store_true", dest="ack")
    p.add_argument("--timeout-seconds", type=float, default=180.0, help="per-case limit")
    return p


class SandboxError(RuntimeError):
    """Raised when a path cannot be confined to the sandbox directory."""


# Reason: matched by field name so a newly added path setting is covered automatically.
_PATHISH = re.compile(r"path|dir|root|file|folder", re.IGNORECASE)
_SEEDS = {
    "notes.txt": "Shopping list: milk, eggs, bread.\n",
    "todo.txt": "1. water the plants\n2. call the plumber\n",
    "readme.txt": "Sandbox file for golden runs.\n",
    "docs/guide.txt": "A short guide that lives in the docs folder.\n",
}


def _is_inside(value: str, sandbox: Path) -> bool:
    """True if value (relative values resolve against the sandbox) stays inside it."""
    # Reason: a Windows path such as D:\\Downloads looks relative on POSIX; never treat it so.
    win = PureWindowsPath(value)
    if (win.drive or win.root) and not PurePosixPath(value).is_absolute():
        return os.name == "nt" and Path(value).resolve().is_relative_to(sandbox.resolve())
    resolved = (sandbox / value).resolve()
    return resolved.is_relative_to(sandbox.resolve())


def _walk_values(obj, prefix: str = ""):
    """Yield (dotted name, string value) for every path-like setting in a config tree."""
    from pydantic import BaseModel

    if isinstance(obj, BaseModel):
        items = [(n, getattr(obj, n)) for n in type(obj).model_fields]
    elif isinstance(obj, dict):
        items = [(str(k), v) for k, v in obj.items()]
    else:
        return
    for name, value in items:
        dotted = f"{prefix}{name}"
        if isinstance(value, (BaseModel, dict)):
            yield from _walk_values(value, dotted + ".")
        elif _PATHISH.search(name):
            for v in value if isinstance(value, list) else [value]:
                if isinstance(v, str):
                    yield dotted, v


def _runtime_path_values(root: str) -> list[tuple[str, str]]:
    """Call every zero-argument core.runtime_paths function that returns a Path."""
    from core import runtime_paths

    found = []
    for name, fn in inspect.getmembers(runtime_paths, inspect.isfunction):
        if name.startswith("_") or fn.__module__ != runtime_paths.__name__:
            continue
        sig = inspect.signature(fn).parameters
        if any(q.default is q.empty for q in sig.values()):
            continue
        out = fn(root=root) if "root" in sig else fn()
        if isinstance(out, Path):
            found.append((f"runtime_paths.{name}", str(out)))
    return found


def find_escapes(config, sandbox: Path) -> list[str]:
    """List every resolved path setting that is not inside the sandbox."""
    values = list(_walk_values(config))
    values += _runtime_path_values(config.runtime_paths.root)
    bad = [f"{n}={v}" for n, v in values if not _is_inside(v, sandbox)]
    cwd = Path.cwd().resolve()
    if cwd != sandbox.resolve():
        bad.append(f"cwd={cwd}")  # relative paths resolve from cwd
    return bad


def _rebase(obj, sandbox: Path) -> None:
    """Move every path-like setting that points outside the sandbox into it."""
    from pydantic import BaseModel

    items = (
        [(n, getattr(obj, n)) for n in type(obj).model_fields]
        if isinstance(obj, BaseModel)
        else list(obj.items())
    )
    for name, value in items:
        if isinstance(value, (BaseModel, dict)):
            _rebase(value, sandbox)
        elif _PATHISH.search(name) and isinstance(value, (str, list)):
            values = value if isinstance(value, list) else [value]
            if not all(isinstance(v, str) for v in values):
                continue
            moved = [
                (
                    v
                    if _is_inside(v, sandbox)
                    else str(sandbox / "var" / "redirected" / (PureWindowsPath(v).name or "x"))
                )
                for v in values
            ]
            new = moved if isinstance(value, list) else moved[0]
            if isinstance(obj, BaseModel):
                setattr(obj, name, new)
            else:
                obj[name] = new


def _seed_sandbox(sandbox: Path) -> Path:
    """Create the sandbox layout, copy config and the throwaway project; return its path."""
    for rel, text in _SEEDS.items():
        target = sandbox / "files" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    for sub in ("data", "logs", "sessions", "workspace", "exports", "cache", "user_files"):
        (sandbox / "var" / sub).mkdir(parents=True, exist_ok=True)
    # Reason: relative config/ reads resolve from the sandbox cwd, and must not reach the repo.
    if (ROOT / "config").is_dir():
        shutil.copytree(ROOT / "config", sandbox / "config", dirs_exist_ok=True)
    project = sandbox / "project"
    shutil.copytree(
        ROOT / "tests/golden/sandbox_project",
        project,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("conftest.py", "__pycache__"),
    )
    # Reason: without its own repo, git would walk up and commit into whatever contains DIR.
    for cmd in (["init", "-q"], ["add", "-A"], ["commit", "-qm", "seed"]):
        subprocess.run(
            ["git", "-c", "user.name=golden", "-c", "user.email=golden@localhost", *cmd],
            cwd=project,
            check=False,
            capture_output=True,
        )
    return project


def _write_sandbox_config(config, sandbox: Path):
    """Write the rebased in-memory config to sandbox/config.yaml and reload it from there.

    Reason: later load_config() calls (e.g. turn metrics) run from the sandbox cwd; without
    a file there they print "not found" and silently fall back to defaults. Dumping the
    rebased config, not copying the repo file, keeps every path inside the sandbox.

    Returns:
        The config as a fresh load_config() from the sandbox cwd sees it.
    """
    import yaml

    from core.config import load_config

    text = yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False, allow_unicode=True)
    (sandbox / "config.yaml").write_text(text, encoding="utf-8")
    return load_config()


@contextlib.contextmanager
def enter_sandbox(config, sandbox: Path) -> Iterator[Path]:
    """Confine config, environment, cwd and the verified-edit root to sandbox, then restore.

    Raises:
        SandboxError: if any resolved path is still outside the sandbox. Nothing stays
            changed in that case.
    """
    sandbox = sandbox.resolve()
    sandbox.mkdir(parents=True, exist_ok=True)
    project = _seed_sandbox(sandbox)
    saved_env = os.environ.get("WITSV3_PROJECT_ROOT")
    saved_cwd = os.getcwd()
    patched: list[tuple[object, object]] = []
    try:
        _rebase(config, sandbox)
        files = str(sandbox / "files")
        config.document_rag.documents_path = files
        config.document_rag.auto_ingest_on_startup = False
        config.security.filesystem_read_roots = [files]
        config.security.document_ingest_roots = [files]
        config.self_repair.restart_after_fix = False
        config.self_repair.daily_schedule_enabled = False
        config.auto_restart_on_file_change = False
        os.environ["WITSV3_PROJECT_ROOT"] = str(sandbox)
        os.chdir(sandbox)
        # Reason: safe_code_editor.PROJECT_ROOT is a module constant copied by value into
        # other modules, so every holder of the old value is repointed (and restored).
        from core import safe_code_editor

        old = safe_code_editor.PROJECT_ROOT
        for mod in list(sys.modules.values()):
            if getattr(mod, "PROJECT_ROOT", None) == old:
                patched.append((mod, old))
                mod.PROJECT_ROOT = project
        escapes = find_escapes(config, sandbox)
        if escapes:
            raise SandboxError("paths outside the sandbox: " + "; ".join(escapes))
        escapes = find_escapes(_write_sandbox_config(config, sandbox), sandbox)
        if escapes:
            raise SandboxError(
                "reloaded config has paths outside the sandbox: " + "; ".join(escapes)
            )
        yield project
    finally:
        for mod, old in patched:
            mod.PROJECT_ROOT = old
        os.chdir(saved_cwd)
        if saved_env is None:
            os.environ.pop("WITSV3_PROJECT_ROOT", None)
        else:
            os.environ["WITSV3_PROJECT_ROOT"] = saved_env


async def run_once(
    control_center, case: GoldenCase
) -> tuple[bool, float | None, list[str], list[str]]:
    """Run one case; return (passed, first_result_ms, tools seen, unobserved)."""
    turn = None
    if turn_metrics is not None:
        turn = turn_metrics.start_turn("golden", None, len(case.message))
    start = time.perf_counter()
    first_result: float | None = None
    events = []
    try:
        history = None
        if case.history:
            from core.schemas import ConversationHistory

            history = ConversationHistory(session_id="golden")
            for turn_ in case.history:
                history.add_message(turn_["role"], turn_["content"])
            history.add_message("user", case.message)
        async for ev in control_center.run(case.message, conversation_history=history):
            events.append(ev)
            if turn_metrics is not None:
                turn_metrics.note_event(ev.type)
            if first_result is None and ev.type in ("result", "error"):
                first_result = (time.perf_counter() - start) * 1000
    finally:
        if turn is not None:
            errored = any(e.type == "error" for e in events)
            await turn_metrics.finish_turn(turn, "error" if errored else "ok")
    names = [c["name"] for c in turn.tool_calls] if turn is not None else None
    res = evaluate(case, events, names)
    return (
        res.passed,
        first_result,
        names if names is not None else tools_seen(events),
        res.unobserved,
    )


async def run_cases(control_center, cases, args) -> int:
    """Run cases, rewriting this run's table after each one; return the exit code."""
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prefix = out.read_text(encoding="utf-8") if out.exists() else ""
    stamp = time.strftime("%Y-%m-%d %H:%M")
    header = "| case | pass | median first_result_ms | tools seen | unobserved |\n|---|---|---|---|---|\n"
    rows, failed_all = [], False
    for case in cases:
        results = []
        for _ in range(args.runs):
            try:
                results.append(
                    await asyncio.wait_for(run_once(control_center, case), args.timeout_seconds)
                )
            except asyncio.TimeoutError:
                results.append((False, None, [], ["timeout"]))
        passes = sum(r[0] for r in results)
        times = [r[1] for r in results if r[1] is not None]
        med = f"{statistics.median(times):.0f}" if times else "-"
        tools = sorted({t for r in results for t in r[2]})
        unobs = sorted({u for r in results for u in r[3]})
        if (
            case.max_first_result_ms
            and times
            and statistics.median(times) > case.max_first_result_ms
        ):
            unobs.append("slow")
        rows.append(
            f"| {case.id} | {passes}/{args.runs} | {med} | {', '.join(tools) or '-'} "
            f"| {', '.join(unobs) or '-'} |"
        )
        failed_all = failed_all or passes == 0
        # Reason: a crash or Ctrl-C mid-run still leaves every finished row on disk.
        section = f"\n## Golden run {stamp}\n\n{header}" + "\n".join(rows) + "\n"
        out.write_text(prefix + section, encoding="utf-8")
    return 1 if failed_all else 0


async def main_async(args: argparse.Namespace) -> int:
    """Build the system (sandboxed unless live), run selected cases, return the exit code."""
    from core.config import load_config  # live system, imported lazily

    cases = load_cases(args.cases)
    only = {s for s in args.only.split(",") if s}
    cases = [
        c for c in cases if (not only or c.id in only) and (not args.tag or args.tag in c.tags)
    ]
    config = load_config(str(ROOT / "config.yaml"))
    with contextlib.ExitStack() as stack:
        if not args.live:
            try:
                stack.enter_context(enter_sandbox(config, Path(args.sandbox)))
            except SandboxError as exc:
                print(f"refusing to start: {exc}", file=sys.stderr)
                return 3
        from run import WitsV3System  # after the sandbox: run.py opens its log at import

        system = WitsV3System(config)
        await system.initialize()
        return await run_cases(system.control_center, cases, args)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; exits 2 before building anything if no safe mode was chosen."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.live and not args.ack:
        print("--live also needs --i-understand-this-touches-real-data", file=sys.stderr)
        return 2
    if not args.live and not args.sandbox:
        print("--sandbox DIR is required (or --live with the acknowledgement)", file=sys.stderr)
        return 2
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
