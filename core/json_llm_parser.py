"""
Shared JSON extraction and repair for LLM structured outputs.

Used by the orchestrator ReAct loop and WCCA intent analysis to recover
valid JSON from qwen3-style responses (think blocks, markdown fences,
truncated objects, trailing commas, etc.).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional

Logger = Optional[logging.Logger]
Validator = Callable[[Dict[str, Any]], Dict[str, Any]]
FallbackBuilder = Callable[[str, str], Dict[str, Any]]


def strip_think_blocks(text: str) -> str:
    """Remove qwen3-style think blocks and stray tags."""
    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return re.sub(r"</?think>", "", text, flags=re.IGNORECASE).strip()


def balanced_json_objects(text: str) -> List[str]:
    """
    Scan for top-level balanced {...} substrings, respecting strings.

    Truncated responses get a best-effort completion.
    """
    objects: List[str] = []
    i = 0
    n = len(text)

    while i < n:
        if text[i] != "{":
            i += 1
            continue

        start = i
        depth = 0
        in_string = False
        escaped = False
        j = i
        while j < n:
            ch = text[j]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        objects.append(text[start : j + 1])
                        break
            j += 1

        if depth > 0:
            fragment = text[start:n].rstrip()
            fragment = re.sub(r",\s*$", "", fragment)
            if in_string:
                fragment += '"'
            objects.append(fragment + "}" * depth)
            break

        i = j + 1 if j < n else n

    return objects


def extract_json_candidates(response: str) -> List[str]:
    """Extract candidate JSON object strings from a raw response, most-likely first."""
    text = strip_think_blocks(response)
    if not text:
        return []

    candidates: List[str] = []

    if text.startswith("{"):
        candidates.append(text)

    for match in re.finditer(
        r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE
    ):
        candidates.append(match.group(1))

    candidates.extend(balanced_json_objects(text))

    seen = set()
    unique: List[str] = []
    for candidate in candidates:
        candidate = candidate.strip()
        if candidate and candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def repair_json(json_str: str) -> str:
    """Apply conservative repairs for common LLM JSON mistakes."""
    repaired = json_str
    repaired = repaired.replace(""", '"').replace(""", '"')
    repaired = repaired.replace("'", "'").replace("'", "'")
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    repaired = re.sub(r"(?<=[:\[,\s])True(?=\s*[,}\]])", "true", repaired)
    repaired = re.sub(r"(?<=[:\[,\s])False(?=\s*[,}\]])", "false", repaired)
    repaired = re.sub(r"(?<=[:\[,\s])None(?=\s*[,}\]])", "null", repaired)
    return repaired


def parse_json_object(
    response: str,
    validator: Validator,
    *,
    logger: Logger = None,
    fallback: FallbackBuilder,
) -> Dict[str, Any]:
    """
    Parse a JSON object from an LLM response with progressive recovery.

    Args:
        response: Raw LLM response text
        validator: Validates and normalizes a parsed dict; raises ValueError on failure
        logger: Optional logger for parse warnings
        fallback: Called with (response, parse_error) when parsing fails entirely

    Returns:
        Validated dict, or fallback result (may include _parse_failed)
    """
    last_error = "No JSON found in response"

    for candidate in extract_json_candidates(response):
        for attempt in (candidate, repair_json(candidate)):
            try:
                parsed = json.loads(attempt)
            except json.JSONDecodeError as exc:
                last_error = str(exc)
                continue

            if not isinstance(parsed, dict):
                last_error = f"Top-level JSON is {type(parsed).__name__}, expected object"
                continue

            try:
                return validator(parsed)
            except ValueError as exc:
                last_error = str(exc)
                break

    if logger:
        logger.warning("Failed to parse JSON LLM response: %s", last_error)
    return fallback(response, last_error)


def build_json_repair_prompt(
    raw_response: str,
    parse_error: str,
    *,
    required_keys: str,
) -> str:
    """
    Build a prompt asking the model to rewrite malformed output as valid JSON.

    Args:
        raw_response: The response that failed to parse
        parse_error: The parse error message
        required_keys: Human-readable list of expected keys for the schema

    Returns:
        Repair prompt
    """
    return f"""The following text was supposed to be a single valid JSON object, but it failed to parse ({parse_error}).

TEXT:
{raw_response}

Rewrite it as ONE valid JSON object. Preserve the intended content and keys ({required_keys}). Do not add commentary, markdown fences, or any text outside the JSON object.

Respond ONLY with the corrected JSON object."""
