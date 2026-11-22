"""Tests for core/tool_call_guard.py: one per tool-call rule plus review fixes."""

import pytest

from core.config import ConversationSettings
from core.tool_call_guard import (
    APOLOGY,
    MAX_MESSAGE,
    OUT_OF_STEPS,
    UNKNOWN_BUCKET,
    TurnGuard,
    canonical_call,
    normalise_schema,
)

# The two shapes tools use today: wrapped under "parameters" (e.g. math_tool) and bare
# top-level properties (e.g. conversation_history_tool).
PROPS = {
    "operation": {"type": "string", "enum": ["sum", "mean"]},
    "data": {"type": "array", "items": {"type": "number"}},
    "options": {"type": "object", "properties": {"round": {"type": "integer"}}},
}
WRAPPED = {
    "name": "math",
    "description": "Do maths",
    "parameters": {"type": "object", "properties": PROPS, "required": ["operation", "data"]},
}
BARE = {"type": "object", "properties": PROPS, "required": ["data", "operation"]}
SCHEMAS = {"math": WRAPPED, "other": BARE}
GOOD = {"operation": "sum", "data": [1, 2.5]}


def make_guard(steps=8, identical=2, retries=1):
    return TurnGuard(max_steps=steps, max_identical_calls=identical, max_schema_retries=retries)


def test_both_schema_shapes_normalise_to_same_object():
    wrapped, bare = normalise_schema(WRAPPED), normalise_schema(BARE)
    assert wrapped == bare
    assert wrapped["additionalProperties"] is False
    assert wrapped["required"] == ["data", "operation"]


def test_rule1_unknown_tool_rejected_and_counts_as_schema_failure():
    guard = make_guard(retries=1)
    first = guard.check("nope", {}, SCHEMAS)
    assert first.action == "reject" and "'nope'" in first.message and "math" in first.message
    assert guard.check("nope", {}, SCHEMAS).message == APOLOGY


def test_unknown_names_share_one_bucket_and_bad_names_do_not_raise():
    guard = make_guard(retries=2)
    assert guard.check("a", {}, SCHEMAS).action == "reject"
    assert guard.check(["b"], {}, SCHEMAS).action == "reject"
    assert guard.check(None, {}, SCHEMAS).action == "stop"
    assert guard.schema_failures == {UNKNOWN_BUCKET: 3}


def test_rule2_schema_errors_go_back_then_turn_ends_after_retries():
    guard = make_guard(retries=1)
    assert guard.check("math", GOOD, SCHEMAS).action == "run"
    bad = guard.check("math", {"operation": "sum", "data": ["x"]}, SCHEMAS)
    assert bad.action == "reject" and "arguments.data[0] must be number" in bad.message
    missing = guard.check("other", {"operation": "sum"}, SCHEMAS)  # own bucket
    assert missing.action == "reject" and "arguments.data is required" in missing.message
    assert guard.check("math", {**GOOD, "colour": "red"}, SCHEMAS).message == APOLOGY


@pytest.mark.parametrize(
    "args",
    [
        {"operation": "max", "data": []},
        {**GOOD, "options": {"round": "2"}},
        {**GOOD, "options": {"round": True}},
        {"operation": "sum", "data": [float("nan")]},
        {"operation": "sum", "data": [float("inf")]},
        [1, 2],
        7,
        "{not json",
        "[" * 100000,
    ],
)
def test_rule2_bad_arguments_rejected_without_raising(args):
    assert make_guard(retries=5).check("math", args, SCHEMAS).action == "reject"


def test_rule2_json_string_and_int_valued_float_accepted():
    guard = make_guard()
    assert guard.check("math", '{"operation": "mean", "data": [3]}', SCHEMAS).action == "run"
    assert guard.check("math", {**GOOD, "options": {"round": 2.0}}, SCHEMAS).action == "run"


def test_enum_is_type_strict():
    schemas = {"t": {"type": "object", "properties": {"v": {"enum": [1, "a"]}}}}
    guard = make_guard(retries=5)
    assert guard.check("t", {"v": True}, schemas).action == "reject"
    assert guard.check("t", {"v": "1"}, schemas).action == "reject"
    assert guard.check("t", {"v": 1.0}, schemas).action == "run"


@pytest.mark.parametrize(
    "prop",
    [
        {"type": "str"},
        {"type": "STRING"},
        {"type": "string", "pattern": "("},
        {"type": "integer", "minimum": "0"},
        {"type": "string", "maxLength": -1},
        {"not": {"type": "string"}},
        {"anyOf": [{"type": "string"}]},
        {"$ref": "#/x"},
        {"const": 1},
        "string",
        None,
    ],
)
def test_unsupported_schema_fails_closed(prop):
    schemas = {"t": {"type": "object", "properties": {"x": prop}}}
    verdict = make_guard().check("t", {"x": "a"}, schemas)
    assert verdict.action == "reject" and "unusable schema" in verdict.message


BOUNDED = {
    "b": {
        "type": "object",
        "properties": {
            "n": {"type": "number", "minimum": 1, "exclusiveMaximum": 10},
            "m": {"type": "integer", "exclusiveMinimum": 0, "maximum": 5},
            "s": {"type": "string", "minLength": 2, "maxLength": 4, "pattern": "^[a-z]+$"},
            "a": {"type": "array", "minItems": 1, "maxItems": 2, "title": "t", "examples": []},
            "p": {"type": "string", "pattern": "end$"},
        },
    }
}


@pytest.mark.parametrize(
    "args, ok",
    [
        ({"n": 1, "m": 5, "s": "ab", "a": [1, 2]}, True),
        ({"n": 0.5}, False),
        ({"n": 10}, False),
        ({"m": 0}, False),
        ({"m": 6}, False),
        ({"s": "a"}, False),
        ({"s": "abcde"}, False),
        ({"s": "AB"}, False),
        ({"a": []}, False),
        ({"a": [1, 2, 3]}, False),
        ({"p": "x" * 20_000 + "end"}, False),  # only the first 10,000 chars are matched
        ({"p": "the end"}, True),
    ],
)
def test_number_size_and_pattern_constraints(args, ok):
    verdict = make_guard(retries=5).check("b", args, BOUNDED)
    assert verdict.action == ("run" if ok else "reject"), verdict.message


def test_malformed_required_fails_closed_and_undeclared_required_is_kept():
    guard = make_guard(retries=5)
    broken = {"t": {"type": "object", "properties": {}, "required": True}}
    assert "unusable schema" in guard.check("t", {}, broken).message
    implicit = {"t": {"parameters": {"required": ["path"]}}}
    assert guard.check("t", {}, implicit).action == "reject"
    assert guard.check("t", {"path": 5}, implicit).action == "run"


def test_error_text_is_bounded():
    many = {f"key{i}" + "x" * 200: "v" * 500 for i in range(50)}
    verdict = make_guard().check("math", {**GOOD, **many}, SCHEMAS)
    assert verdict.action == "reject"
    assert len(verdict.message) <= MAX_MESSAGE and "(+45 more)" in verdict.message


def test_rule3_identical_call_not_run_after_limit():
    guard = make_guard(identical=2)
    assert guard.check("math", GOOD, SCHEMAS).action == "run"
    assert guard.check("math", GOOD, SCHEMAS).action == "run"
    third = guard.check("math", GOOD, SCHEMAS)
    assert third.action == "reject" and "already called" in third.message
    assert guard.check("math", {"operation": "mean", "data": [1]}, SCHEMAS).action == "run"


def test_argument_order_and_int_float_do_not_defeat_identical_check():
    guard = make_guard(identical=1)
    assert guard.check("math", {"operation": "sum", "data": [1]}, SCHEMAS).action == "run"
    assert guard.check("math", {"data": [1.0], "operation": "sum"}, SCHEMAS).action == "reject"
    assert canonical_call("t", {"a": {"b": [2.0]}}) == canonical_call("t", {"a": {"b": [2]}})
    assert canonical_call("t", {"a": True}) != canonical_call("t", {"a": 1})


@pytest.mark.parametrize("steps", [1, 8])
def test_rule4_calls_in_rounds_up_to_max_steps_run_then_stop(steps):
    guard = make_guard(steps=steps)
    for i in range(steps):
        guard.note_round()
        assert guard.check("math", {"operation": "sum", "data": [i]}, SCHEMAS).action == "run"
    assert guard.steps_left() == 0
    guard.note_round()
    fresh = {"operation": "mean", "data": [99]}  # valid and never seen: would otherwise run
    verdict = guard.check("math", fresh, SCHEMAS)
    assert verdict.action == "stop" and verdict.message == OUT_OF_STEPS


def test_step_cap_comes_before_unknown_and_identical_rules():
    guard = make_guard(steps=1, identical=1, retries=100)
    guard.note_round()
    assert guard.check("math", GOOD, SCHEMAS).action == "run"
    assert guard.check("math", GOOD, SCHEMAS).action == "reject"
    guard.note_round()
    assert guard.check("math", GOOD, SCHEMAS).action == "stop"
    assert all(guard.check(f"x{i}", {}, SCHEMAS).action == "stop" for i in range(200))


def test_stop_latches_for_the_rest_of_the_turn():
    guard = make_guard(retries=0)
    stop = guard.check("nope", {}, SCHEMAS)
    assert stop.action == "stop"
    assert guard.check("math", GOOD, SCHEMAS) == stop
    assert guard.text_call_verdict('<tool_call>{"name": "math"}</tool_call>', False) == stop


def test_rule5_text_tool_call_is_one_schema_failure():
    guard = make_guard(retries=1)
    text = 'Sure.\n< TOOL_CALL >{"name": "math", "arguments": {"data": [1]}}</tool_call >'
    assert guard.text_call_verdict("Plain answer, no calls.", False) is None
    assert guard.text_call_verdict(text, True) is None
    assert guard.text_call_verdict(text, False).action == "reject"
    assert guard.schema_failures == {UNKNOWN_BUCKET: 1}
    # Text calls and unknown tools share the same budget.
    assert guard.check("nope", {}, SCHEMAS).message == APOLOGY


def test_text_mentioning_tool_call_tag_is_not_a_call():
    guard = make_guard(retries=0)
    prose = "Models sometimes emit a <tool_call> block; I will not do that here."
    assert guard.text_call_verdict(prose, False) is None
    assert guard.text_call_verdict("<tool_call>" + "[" * 100000, False) is None
    assert guard.text_call_verdict(None, False) is None


def test_conversation_settings_defaults():
    conv = ConversationSettings()
    assert (conv.max_steps, conv.max_identical_calls, conv.max_schema_retries) == (8, 2, 1)
