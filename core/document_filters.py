"""File-name filters for document_search, split out to keep document_tools.py small."""

from pathlib import Path
from typing import Any


def coerce_file_filters(*values: Any) -> list[str]:
    """Normalize LLM file filter args (file_name, file_names, source_files, …)."""
    out: list[str] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            if value.strip():
                out.append(value.strip())
            continue
        if isinstance(value, (list, tuple, set)):
            for item in value:
                if isinstance(item, str) and item.strip():
                    out.append(item.strip())
    # Preserve order while de-duplicating (case-insensitive).
    seen: set = set()
    unique: list[str] = []
    for name in out:
        key = name.lower()
        if key not in seen:
            seen.add(key)
            unique.append(name)
    return unique


def file_matches_filter(file_path: str, filters: list[str]) -> bool:
    """True when *file_path* matches any basename or substring filter."""
    if not filters:
        return True
    path_lower = (file_path or "").lower()
    base_lower = Path(file_path).name.lower()
    for filt in filters:
        fl = filt.lower()
        if fl in (path_lower, base_lower):
            return True
        if fl in path_lower or base_lower.endswith(fl) or path_lower.endswith(fl):
            return True
    return False


def default_query_from_filename(file_name: str) -> str:
    """Build a searchable query when the LLM only passes a file name."""
    stem = Path(file_name).stem.replace("_", " ").replace("-", " ").strip()
    return stem or file_name
