#!/usr/bin/env python3
"""Count or remove stored INTENT_ANALYSIS memory segments.

Dry run by default: prints the count and changes nothing. With ``--apply`` it first copies the
memory file to a timestamped backup next to it, verifies the copy, and only then rewrites the
file without the INTENT_ANALYSIS segments. Nothing else is ever removed.

Covers the backends that keep segments in one JSON file: ``basic``, ``faiss_cpu``, ``faiss_gpu``.
(FAISS rebuilds its vector index from the segments on next start when the counts differ.)
Not covered: ``neural`` and ``supabase`` stores.

Usage:
    python scripts/purge_intent_memory.py [--memory-file PATH] [--apply]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SEGMENT_TYPE = "INTENT_ANALYSIS"
JSON_BACKENDS = ("basic", "faiss_cpu", "faiss_gpu")


def load_segments(path: Path) -> tuple[list[dict], bool]:
    """Read the memory file.

    Returns:
        The segment dicts and whether the file used the ``{"segments": [...]}`` wrapper.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return list(data.get("segments", [])), True
    return list(data), False


def count_intent(segments: list[dict]) -> int:
    """Count segments whose type is INTENT_ANALYSIS."""
    return sum(1 for s in segments if isinstance(s, dict) and s.get("type") == SEGMENT_TYPE)


def web_server_running(host: str = "127.0.0.1", port: int = 8000) -> bool:
    """Best-effort check: True when something accepts connections on the web UI port."""
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def write_backup(path: Path) -> Path:
    """Copy *path* to a timestamped sibling and verify the copy. Raises OSError on failure."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = path.with_name(f"{path.name}.{stamp}.bak")
    if backup.exists():
        raise OSError(f"backup already exists: {backup}")
    shutil.copy2(path, backup)
    if backup.stat().st_size != path.stat().st_size or backup.read_bytes() != path.read_bytes():
        raise OSError(f"backup verification failed: {backup}")
    return backup


def purge(path: Path, apply: bool) -> tuple[int, int, Path | None]:
    """Count (and with *apply*, remove) INTENT_ANALYSIS segments.

    Returns:
        (intent_count, total_segments, backup_path_or_None).
    """
    segments, wrapped = load_segments(path)
    found = count_intent(segments)
    if not apply or found == 0:
        return found, len(segments), None
    backup = write_backup(path)  # Reason: abort before any write if the backup cannot be made.
    kept = [s for s in segments if not (isinstance(s, dict) and s.get("type") == SEGMENT_TYPE)]
    payload = {"segments": kept} if wrapped else kept
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return found, len(segments), backup


def _default_memory_file() -> tuple[Path, str]:
    from core.config import load_config

    config = load_config()
    return Path(config.memory_manager.memory_file_path), config.memory_manager.backend


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--memory-file", type=Path, help="memory JSON (default: from config)")
    parser.add_argument("--apply", action="store_true", help="delete rows (default: dry run)")
    parser.add_argument("--port", type=int, default=8000, help="web UI port to check")
    args = parser.parse_args(argv)

    path, backend = args.memory_file, None
    if path is None:
        path, backend = _default_memory_file()
    if backend is not None and backend not in JSON_BACKENDS:
        print(f"Backend '{backend}' is not covered by this script; nothing done.")
        return 2
    if not path.is_file():
        print(f"Memory file not found: {path}")
        return 1
    if args.apply and web_server_running(port=args.port):
        print(f"Web server appears to be running on port {args.port}; stop it first.")
        return 3
    try:
        found, total, backup = purge(path, args.apply)
    except (OSError, ValueError) as exc:
        print(f"Aborted, memory file unchanged unless noted above: {exc}")
        return 1
    print(f"{found} of {total} segments are {SEGMENT_TYPE} ({path})")
    if not args.apply:
        print("Dry run: nothing changed. Re-run with --apply to delete them.")
    elif backup:
        print(f"Removed {found}. Backup: {backup}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
