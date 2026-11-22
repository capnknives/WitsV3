"""Tests for core/tool_call_guard.py: one per tool-call rule plus shape and order cases."""

from core.config import WitsV3Config
from core.tool_call_guard import APOLOGY, OUT_OF_STEPS, TurnGuard, normalise_schema

# The two shapes tools use today: wrapped under "parameters" (e.g. math_tool) and bare
# top-level properties (e.g. conversation_history_tool).
WRAPPED = {
    "name": "math_operations",
    "description": "Do maths",
    "parameters": {
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": ["sum", "mean"]},
            "data": {"type": "array", "items": {"type": "number"}},
            "options": {"type": "object", "properties": {"round": {"type": "integer"}}},
        },
        "required": ["operation", "data"],
    },
}
BARE = {
    "type": "object",
    "properties": {
        "operation": {"type": "string", "enum": ["sum", "mean"]},
        "data": {"type": "array", "items": {"type": "number"}},
        "options": {"type": "object", "properties": {"round": {"type": "integer"}}},
    },
    "required": ["data", "operation"],
}
SCHEMAS = {"math": WRAPPED, "other": BARE}
GOOD = {"operation": "sum", "data": [1, 2.5]}


def make_guard(steps=8, identical=2, retries=1):
    return TurnGuard(max_steps=steps, max_identical_calls=identical, max_schema_retries=retries)


def test_both_schema_shapes_normalise_to_same_object():
    wrapped, bare = normalise_schema(WRAPPED), normalise_schema(BARE)
    assert wrapped == bare
    assert wrapped["additionalProperties"] is False
    assert wrapped["type"] == "object"
    assert wrapped["required"] == ["data", "operation"]


def test_rule1_unknown_tool_rejected_and_counts_as_schema_failure():
    guard = make_guard(retries=1)
    first = guard.check("nope", {}, SCHEMAS)
    assert first.action == "reject"
    assert "Unknown tool 'nope'" in first.message and "math" in first.message
    second = guard.check("nope", {}, SCHEMAS)
    assert second.action == "stop" and second.message == APOLOGY


def test_rule2_schema_errors_go_back_then_turn_ends_after_retries():
    guard = make_guard(retries=1)
    assert guard.check("math", GOOD, SCHEMAS).action == "run"

    bad_type = guard.check("math", {"operation": "sum", "data": ["x"]}, SCHEMAS)
    assert bad_type.action == "reject"
    assert "arguments.data[0] must be number" in bad_type.message

    # A different tool has its own failure budget.
    missing = guard.check("other", {"operation": "sum"}, SCHEMAS)
    assert missing.action == "reject" and "arguments.data is required" in missing.message

    unknown_key = guard.check("math", {**GOOD, "colour": "red"}, SCHEMAS)
    assert unknown_key.action == "stop" and unknown_key.message == APOLOGY


def test_rule2_validator_covers_enum_nested_objects_bools_and_json_strings():
    guard = make_guard(retries=10)
    bad_enum = guard.check("math", {"operation": "max", "data": []}, SCHEMAS)
    assert "must be one of" in bad_enum.message
    nested = guard.check("math", {**GOOD, "options": {"round": "2"}}, SCHEMAS)
    assert "arguments.options.round must be integer" in nested.message
    as_bool = guard.check("math", {**GOOD, "options": {"round": True}}, SCHEMAS)
    assert as_bool.action == "reject"
    assert guard.check("math", "{not json", SCHEMAS).action == "reject"
    assert guard.check("math", '{"operation": "mean", "data": [3]}', SCHEMAS).action == "run"


def test_rule3_identical_call_not_run_after_limit():
    guard = make_guard(identical=2)
    assert guard.check("math", GOOD, SCHEMAS).action == "run"
    assert guard.check("math", GOOD, SCHEMAS).action == "run"
    third = guard.check("math", GOOD, SCHEMAS)
    assert third.action == "reject" and "already called" in third.message
    # Different arguments are a different call.
    assert guard.check("math", {"operation": "mean", "data": [1]}, SCHEMAS).action == "run"


def test_argument_order_does_not_defeat_identical_check():
    guard = make_guard(identical=1)
    assert guard.check("math", {"operation": "sum", "data": [1]}, SCHEMAS).action == "run"
    reordered = guard.check("math", {"data": [1], "operation": "sum"}, SCHEMAS)
    assert reordered.action == "reject"


def test_rule4_stop_after_max_steps():
    guard = make_guard(steps=2)
    assert guard.steps_left() == 2
    guard.note_round()
    assert guard.check("math", GOOD, SCHEMAS).action == "run"
    guard.note_round()
    assert guard.steps_left() == 0
    verdict = guard.check("math", {"operation": "mean", "data": [2]}, SCHEMAS)
    assert verdict.action == "stop" and verdict.message == OUT_OF_STEPS
    guard.note_round()
    assert guard.steps_left() == 0


def test_rule5_text_tool_call_is_one_schema_failure():
    guard = make_guard(retries=1)
    text = 'Sure.\n<tool_call>{"name": "math", "arguments": {"data": [1]}}</tool_call>'
    assert guard.plain_text_call("Plain answer, no calls.") is False
    assert guard.plain_text_call(text, has_structured_call=True) is False

    assert guard.plain_text_call(text) is True
    assert guard.last_verdict.action == "reject"
    assert guard.schema_failures == {"math": 1}

    # The text failure and a schema failure share the same per-tool budget.
    verdict = guard.check("math", {"data": [1]}, SCHEMAS)
    assert verdict.action == "stop" and verdict.message == APOLOGY

    unnamed = make_guard(retries=0)
    assert unnamed.plain_text_call("<tool_call>garbage") is True
    assert unnamed.last_verdict.action == "stop"


def test_conversation_settings_defaults():
    conv = WitsV3Config().conversation
    assert (conv.max_steps, conv.max_identical_calls, conv.max_schema_retries) == (8, 2, 1)
    loaded = WitsV3Config.from_yaml("config.yaml").conversation
    assert (loaded.max_steps, loaded.max_identical_calls, loaded.max_schema_retries) == (8, 2, 1)
