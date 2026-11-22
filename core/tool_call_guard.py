"""Per-turn guard for model tool calls (C3a).

Pure logic, no I/O: the native tool loop asks a ``TurnGuard`` about each call the model
makes and acts on the returned ``Verdict``. The rules come from the "Tool-call rules" in
docs/roadmap/packages/stage-2-core.md:

1. Unknown tool name: rejected, counts as a schema failure.
2. Arguments must match the tool's normalised schema. More than ``max_schema_retries``
   failures in one bucket ends the turn with an apology.
3. A call identical (name + canonical JSON arguments) to earlier runs is refused once
   ``max_identical_calls`` runs have happened.
4. After ``max_steps`` model rounds the loop stops and answers from what it has.
5. A text reply holding a ``<tool_call>`` JSON block and no structured call is one schema
   failure, never shown to the owner.

Any ``stop`` latches: every later verdict in the turn is the same stop.
"""

from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Literal

APOLOGY = (
    "Sorry, I couldn't complete that: the tool calls I tried kept failing. "
    "Please rephrase or try again."
)
OUT_OF_STEPS = "Step limit reached. Answer now from the observations you already have."
# Reason: names the model invents must not each get a fresh retry budget, so unknown
# tools and text-written calls all share this one bucket.
UNKNOWN_BUCKET = "<unknown>"
SUPPORTED_KEYWORDS = frozenset(
    {"type", "description", "title", "default", "examples", "enum"}
    | {"properties", "required", "items", "additionalProperties"}
)
MAX_ERRORS, MAX_REPR, MAX_MESSAGE = 5, 80, 600

_TEXT_CALL_RE = re.compile(
    r"<\s*tool_call\s*>(.*?)(?:<\s*/\s*tool_call\s*>|$)", re.DOTALL | re.IGNORECASE
)


def _is_number(v: Any) -> bool:
    # Reason: bool is a subclass of int, and NaN/Infinity are not JSON numbers.
    # Ints are checked apart because math.isfinite overflows on huge ints.
    if isinstance(v, bool):
        return False
    return isinstance(v, int) or (isinstance(v, float) and math.isfinite(v))


_TYPE_CHECKS = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: _is_number(v) and (isinstance(v, int) or v.is_integer()),
    "number": _is_number,
    "boolean": lambda v: isinstance(v, bool),
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "null": lambda v: v is None,
}


class SchemaError(ValueError):
    """A tool's schema uses something this validator does not support."""


@dataclass(frozen=True)
class Verdict:
    """What the loop should do with one tool call.

    Attributes:
        action: ``run`` the tool, ``reject`` (send ``message`` back to the model), or
            ``stop`` the turn (``message`` is the text to end with).
        message: Error text for the model, or the closing text on ``stop``.
    """

    action: Literal["run", "reject", "stop"]
    message: str = ""


def _short(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= MAX_REPR else text[: MAX_REPR - 3] + "..."


def _check_schema(schema: Any, path: str) -> None:
    """Raise SchemaError unless ``schema`` uses only the supported subset."""
    if not isinstance(schema, dict):
        raise SchemaError(f"{path} is not an object")
    unsupported = sorted(set(schema) - SUPPORTED_KEYWORDS)
    if unsupported:
        raise SchemaError(f"{path} uses unsupported keywords {unsupported}")
    kind = schema.get("type")
    names = kind if isinstance(kind, list) else [] if kind is None else [kind]
    if not all(isinstance(n, str) and n in _TYPE_CHECKS for n in names):
        raise SchemaError(f"{path} has unknown type {_short(kind)}")
    if "enum" in schema and not isinstance(schema["enum"], list):
        raise SchemaError(f"{path}.enum is not a list")
    required = schema.get("required", [])
    if not isinstance(required, list) or not all(isinstance(r, str) for r in required):
        raise SchemaError(f"{path}.required is not a list of names")
    if not isinstance(schema.get("additionalProperties", False), bool):
        raise SchemaError(f"{path}.additionalProperties must be true or false")
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        raise SchemaError(f"{path}.properties is not an object")
    for key, sub in properties.items():
        _check_schema(sub, f"{path}.{key}")
    if "items" in schema:
        _check_schema(schema["items"], f"{path}.items")


def normalise_schema(raw: dict[str, Any]) -> dict[str, Any]:
    """Turn either tool schema shape into one closed JSON Schema object.

    Tools return either ``{"name", "description", "parameters": {...}}`` or a bare
    ``{"type": "object", "properties": {...}}``; ``{"function": {...}}`` wrappers are
    unwrapped too. Supported keywords: ``type`` (string, integer, number, boolean,
    object, array, null, or a list of these), ``enum``, ``properties``, ``required``,
    ``items``, ``additionalProperties`` (bool), plus the annotations ``description``,
    ``title``, ``default`` and ``examples``. Anything else is refused, never skipped.

    Args:
        raw: The schema as a tool or registry provides it.

    Returns:
        ``{"type": "object", "properties", "required", "additionalProperties": False}``.

    Raises:
        SchemaError: The schema is malformed or uses an unsupported keyword.
    """
    if not isinstance(raw, dict):
        raise SchemaError("schema is not an object")
    body = raw.get("function") if isinstance(raw.get("function"), dict) else raw
    body = body.get("parameters") if "parameters" in body else body
    if isinstance(body, dict) and body.get("type", "object") != "object":
        raise SchemaError("top-level type must be object")
    if body is raw:
        # Reason: the bare shape may carry the tool's own name next to its properties.
        body = {k: v for k, v in raw.items() if k != "name"}
    _check_schema(body, "schema")
    properties = copy.deepcopy(body.get("properties", {}))
    for key in body.get("required", []):
        # Reason: an undeclared required name stays required, accepted with any value.
        properties.setdefault(key, {})
    return {
        "type": "object",
        "properties": properties,
        "required": sorted(set(body.get("required", []))),
        "additionalProperties": False,
    }


def validate(value: Any, schema: dict[str, Any], path: str = "arguments") -> list[str]:
    """Check ``value`` against a schema already accepted by ``normalise_schema``.

    Args:
        value: The decoded value to check.
        schema: The (sub)schema to check it against.
        path: Location used in error text.

    Returns:
        Error strings; empty when the value is valid.
    """
    kind = schema.get("type")
    names = kind if isinstance(kind, list) else [] if kind is None else [kind]
    if names and not any(_TYPE_CHECKS[n](value) for n in names):
        return [f"{path} must be {' or '.join(names)}, got {_short(value)}"]
    if "enum" in schema:
        allowed = {_canon_text(v) for v in schema["enum"]}
        if _canon_text(value) not in allowed:
            return [f"{path} must be one of {_short(schema['enum'])}, got {_short(value)}"]

    errors: list[str] = []
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}.{key} is required")
        for key, item in value.items():
            if key in props:
                errors.extend(validate(item, props[key], f"{path}.{key}"))
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}.{_short(key)} is not a known parameter")
    elif isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            errors.extend(validate(item, schema["items"], f"{path}[{i}]"))
    return errors


def _canon(value: Any) -> Any:
    """Make integer-valued floats ints, recursively, so 1.0 and 1 compare equal."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {k: _canon(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_canon(v) for v in value]
    return value


def _canon_text(value: Any) -> str:
    # Reason: JSON text keeps true and 1 apart, which plain == does not.
    return json.dumps(_canon(value), sort_keys=True, separators=(",", ":"), default=str)


def canonical_call(name: str, arguments: dict[str, Any]) -> str:
    """Return a key equal for calls differing only in key order or 1 versus 1.0."""
    return f"{name}:{_canon_text(arguments)}"


def _bounded(prefix: str, errors: list[str]) -> str:
    shown = errors[:MAX_ERRORS]
    more = f" (+{len(errors) - MAX_ERRORS} more)" if len(errors) > MAX_ERRORS else ""
    text = prefix + "; ".join(shown)
    room = MAX_MESSAGE - len(more)
    # Reason: keep the "(+N more)" count visible even when the list itself is cut.
    return (text if len(text) <= room else text[: room - 3] + "...") + more


class TurnGuard:
    """Tracks one turn's tool calls and applies the five tool-call rules.

    Create one per user turn. Call ``note_round()`` once per model reply, before
    checking that reply's calls. Calls made in rounds 1 to ``max_steps`` may run; a call
    in round ``max_steps + 1`` gets ``stop`` (so ``max_steps=1`` allows one tool round).
    Once any verdict is ``stop``, every later verdict is that same stop.
    """

    def __init__(self, max_steps: int, max_identical_calls: int, max_schema_retries: int):
        """Store the limits (from ``WitsV3Config.conversation``).

        Args:
            max_steps: Model rounds whose tool calls may run.
            max_identical_calls: Runs allowed for one identical call.
            max_schema_retries: Retries allowed per bucket after a schema failure.
        """
        self.max_steps = max_steps
        self.max_identical_calls = max_identical_calls
        self.max_schema_retries = max_schema_retries
        self.rounds = 0
        self.schema_failures: dict[str, int] = {}
        self.call_counts: dict[str, int] = {}
        self._stopped: Verdict | None = None

    def note_round(self) -> None:
        """Record that the model has replied once more this turn."""
        self.rounds += 1

    def steps_left(self) -> int:
        """Return model rounds still allowed (never below zero)."""
        return max(0, self.max_steps - self.rounds)

    def check(self, name: Any, arguments: Any, schemas: dict[str, dict[str, Any]]) -> Verdict:
        """Decide whether one structured tool call may run. Never raises.

        Args:
            name: Tool name the model asked for.
            arguments: Decoded arguments (a JSON string is decoded here).
            schemas: Tool name to raw schema, in either shape.

        Returns:
            The verdict for this call.
        """
        try:
            return self._check(name, arguments, schemas)
        except Exception:  # Reason: a guard bug must end the turn, not crash the loop.
            return self._stop(APOLOGY)

    def text_call_verdict(self, content: str, has_structured_call: bool) -> Verdict | None:
        """Judge a reply that writes a tool call as ``<tool_call>`` text. Never raises.

        Args:
            content: The model's reply text.
            has_structured_call: True when the same reply also carried structured calls.

        Returns:
            None when the reply is not a text tool call; otherwise reject or stop. The
            text must not be shown to the owner either way.
        """
        try:
            if self._stopped:
                return self._stopped
            if has_structured_call or not isinstance(content, str):
                return None
            if not any(_json_object(m) for m in _TEXT_CALL_RE.findall(content)):
                return None
            if self.rounds > self.max_steps:
                return self._stop(OUT_OF_STEPS)
            return self._fail(
                UNKNOWN_BUCKET,
                "Tool calls written as <tool_call> text are not run. "
                "Use the structured tool-call format.",
            )
        except Exception:  # Reason: a guard bug must end the turn, not crash the loop.
            return self._stop(APOLOGY)

    def _check(self, name: Any, arguments: Any, schemas: dict[str, dict[str, Any]]) -> Verdict:
        if self._stopped:
            return self._stopped
        if self.rounds > self.max_steps:
            return self._stop(OUT_OF_STEPS)
        if not isinstance(name, str) or name not in schemas:
            known = ", ".join(sorted(schemas)) or "none"
            return self._fail(
                UNKNOWN_BUCKET, _bounded(f"Unknown tool {_short(name)}. Tools: ", [known])
            )

        try:
            schema = normalise_schema(schemas[name])
        except SchemaError as exc:
            return self._fail(name, f"Tool '{name}' has an unusable schema (fix the tool): {exc}")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except (ValueError, RecursionError):
                return self._fail(name, f"Arguments for '{name}' are not valid JSON.")
        if arguments is None:
            arguments = {}
        errors = validate(arguments, schema)
        if errors:
            return self._fail(name, _bounded(f"Invalid arguments for '{name}': ", errors))

        key = canonical_call(name, arguments)
        if self.call_counts.get(key, 0) >= self.max_identical_calls:
            return Verdict(
                "reject",
                f"'{name}' was already called with these exact arguments; its result is "
                "above. Use it, change the arguments, or answer.",
            )
        self.call_counts[key] = self.call_counts.get(key, 0) + 1
        return Verdict("run")

    def _fail(self, bucket: str, message: str) -> Verdict:
        # Reason: max_schema_retries counts retries, so the turn ends on the failure
        # after the last allowed retry, not on the first failure.
        self.schema_failures[bucket] = self.schema_failures.get(bucket, 0) + 1
        if self.schema_failures[bucket] > self.max_schema_retries:
            return self._stop(APOLOGY)
        return Verdict("reject", message)

    def _stop(self, message: str) -> Verdict:
        self._stopped = Verdict("stop", message)
        return self._stopped


def _json_object(body: str) -> bool:
    try:
        return isinstance(json.loads(body.strip()), dict)
    except (ValueError, RecursionError):
        return False
