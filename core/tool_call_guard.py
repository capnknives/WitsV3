"""Per-turn guard for model tool calls (C3a).

Pure logic, no I/O: the native tool loop asks a ``TurnGuard`` about each call the model
makes and acts on the returned ``Verdict``. The rules and their order come from the
"Tool-call rules" in docs/roadmap/packages/stage-2-core.md:

1. Unknown tool name: rejected, counts as a schema failure.
2. Arguments must match the tool's normalised schema. More than ``max_schema_retries``
   failures for one tool in a turn ends the turn with an apology.
3. A call identical (name + canonical JSON arguments) to earlier runs is refused once
   ``max_identical_calls`` runs have happened.
4. After ``max_steps`` model rounds the loop stops and answers from what it has.
5. A text reply holding a ``<tool_call>`` block and no structured call is one schema
   failure, never shown to the owner.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from typing import Any, Literal

APOLOGY = (
    "Sorry, I couldn't complete that: the tool calls I tried kept failing. "
    "Please rephrase or try again."
)
OUT_OF_STEPS = "Step limit reached. Answer now from the observations you already have."

_TEXT_CALL_RE = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.DOTALL)
# Reason: bool is a subclass of int in Python, so it must be excluded explicitly.
_TYPE_CHECKS = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "null": lambda v: v is None,
}


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


def normalise_schema(raw: dict[str, Any]) -> dict[str, Any]:
    """Turn either tool schema shape into one closed JSON Schema object.

    Tools return either ``{"name", "description", "parameters": {...}}`` or a bare
    ``{"type": "object", "properties": {...}}``; OpenAI-style ``{"function": {...}}``
    wrappers are unwrapped too.

    Args:
        raw: The schema as a tool or registry provides it.

    Returns:
        ``{"type": "object", "properties", "required", "additionalProperties": False}``.
    """
    body = raw
    if isinstance(body.get("function"), dict):
        body = body["function"]
    if isinstance(body.get("parameters"), dict):
        body = body["parameters"]
    properties = copy.deepcopy(body.get("properties") or {})
    # Reason: a required key the schema never declares could never be satisfied once
    # unknown keys are refused, so it is dropped rather than failing every call.
    required = sorted(k for k in body.get("required") or [] if k in properties)
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def validate(value: Any, schema: dict[str, Any], path: str = "arguments") -> list[str]:
    """Check ``value`` against a small JSON Schema subset.

    Supports ``type`` (single or list), ``enum``, ``required``, ``properties``,
    ``additionalProperties: false`` and array ``items``, recursively.

    Args:
        value: The decoded value to check.
        schema: The (sub)schema to check it against.
        path: Location used in error text.

    Returns:
        Error strings; empty when the value is valid.
    """
    expected = schema.get("type")
    if expected:
        names = expected if isinstance(expected, list) else [expected]
        known = [n for n in names if n in _TYPE_CHECKS]
        if known and not any(_TYPE_CHECKS[n](value) for n in known):
            return [f"{path} must be {' or '.join(known)}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        return [f"{path} must be one of {schema['enum']}, got {value!r}"]

    errors: list[str] = []
    if isinstance(value, dict):
        props = schema.get("properties") or {}
        for key in schema.get("required") or []:
            if key not in value:
                errors.append(f"{path}.{key} is required")
        for key, item in value.items():
            if key in props:
                errors.extend(validate(item, props[key], f"{path}.{key}"))
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}.{key} is not a known parameter")
    elif isinstance(value, list) and isinstance(schema.get("items"), dict):
        for i, item in enumerate(value):
            errors.extend(validate(item, schema["items"], f"{path}[{i}]"))
    return errors


def canonical_call(name: str, arguments: dict[str, Any]) -> str:
    """Return a key that is equal for calls differing only in argument order."""
    args = json.dumps(arguments, sort_keys=True, separators=(",", ":"), default=str)
    return f"{name}:{args}"


class TurnGuard:
    """Tracks one turn's tool calls and applies the five tool-call rules.

    Create one per user turn. Call ``note_round()`` after each model round and
    ``check()`` for every structured call before running it.
    """

    def __init__(self, max_steps: int, max_identical_calls: int, max_schema_retries: int):
        """Store the limits (from ``WitsV3Config.conversation``).

        Args:
            max_steps: Model rounds allowed in one turn.
            max_identical_calls: Runs allowed for one identical call.
            max_schema_retries: Retries allowed per tool after a schema failure.
        """
        self.max_steps = max_steps
        self.max_identical_calls = max_identical_calls
        self.max_schema_retries = max_schema_retries
        self.rounds = 0
        self.schema_failures: dict[str, int] = {}
        self.call_counts: dict[str, int] = {}
        self.last_verdict: Verdict | None = None

    def note_round(self) -> None:
        """Record that the model has been asked once more this turn."""
        self.rounds += 1

    def steps_left(self) -> int:
        """Return model rounds still allowed (never below zero)."""
        return max(0, self.max_steps - self.rounds)

    def check(self, name: str, arguments: Any, schemas: dict[str, dict[str, Any]]) -> Verdict:
        """Decide whether one structured tool call may run.

        Args:
            name: Tool name the model asked for.
            arguments: Decoded arguments (a JSON string is decoded here).
            schemas: Tool name to raw schema, in either shape.

        Returns:
            The verdict; also kept in ``last_verdict``.
        """
        if name not in schemas:
            known = ", ".join(sorted(schemas)) or "none"
            return self._fail(name, f"Unknown tool '{name}'. Available tools: {known}.")

        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except ValueError:
                return self._fail(name, f"Arguments for '{name}' are not valid JSON.")
        if arguments is None:
            arguments = {}
        errors = validate(arguments, normalise_schema(schemas[name]))
        if errors:
            return self._fail(name, f"Invalid arguments for '{name}': " + "; ".join(errors))

        key = canonical_call(name, arguments)
        if self.call_counts.get(key, 0) >= self.max_identical_calls:
            return self._done(
                Verdict(
                    "reject",
                    f"'{name}' was already called with these exact arguments; its result is "
                    "above. Use it, change the arguments, or answer.",
                )
            )

        if self.steps_left() <= 0:
            return self._done(Verdict("stop", OUT_OF_STEPS))

        self.call_counts[key] = self.call_counts.get(key, 0) + 1
        return self._done(Verdict("run"))

    def plain_text_call(self, content: str, has_structured_call: bool = False) -> bool:
        """Detect a tool call written as text instead of a structured call.

        When detected it counts as one schema failure for the named tool (or for
        ``<tool_call>`` when no name can be read), and ``last_verdict`` says whether to
        reject (send the message back) or stop the turn. Either way the text must not
        be shown to the owner.

        Args:
            content: The model's reply text.
            has_structured_call: True when the same reply also carried structured calls.

        Returns:
            True when the reply is a text-only tool call.
        """
        match = _TEXT_CALL_RE.search(content or "")
        if has_structured_call or not match:
            return False
        name = "<tool_call>"
        try:
            parsed = json.loads(match.group(1).strip())
            if isinstance(parsed, dict) and isinstance(parsed.get("name"), str):
                name = parsed["name"]
        except ValueError:
            pass
        self._fail(
            name,
            "Tool calls written as <tool_call> text are not run. "
            "Use the structured tool-call format.",
        )
        return True

    def _fail(self, name: str, message: str) -> Verdict:
        # Reason: max_schema_retries counts retries, so the turn ends on the failure
        # after the last allowed retry, not on the first failure.
        self.schema_failures[name] = self.schema_failures.get(name, 0) + 1
        if self.schema_failures[name] > self.max_schema_retries:
            return self._done(Verdict("stop", APOLOGY))
        return self._done(Verdict("reject", message))

    def _done(self, verdict: Verdict) -> Verdict:
        self.last_verdict = verdict
        return verdict
