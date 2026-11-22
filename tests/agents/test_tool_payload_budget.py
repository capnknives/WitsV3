"""Always-tier payload budget and the sent-vs-guard schema views (WP C4d)."""

from __future__ import annotations

import json
import random

import pytest

from agents.native_tool_orchestrator import NativeToolOrchestrator
from agents.tool_tiers import build_tier_specs
from core.config import ToolTierSettings
from core.tool_call_guard import validate
from core.tool_context import CONTEXT_PARAMS
from core.tool_registry import ToolRegistry

MAX_PAYLOAD_CHARS = 8800
MAX_TOOL_CHARS = 800
EXCLUDE = NativeToolOrchestrator._ORCHESTRATOR_TOOL_EXCLUDE


@pytest.fixture(scope="module")
def registry():
    return ToolRegistry()


@pytest.fixture(scope="module")
def specs(registry):
    return build_tier_specs(registry, EXCLUDE, ToolTierSettings())


def _shape(schema: dict) -> dict:
    """Names, types, enums and required: what both views must agree on."""
    props = {
        n: (p.get("type"), p.get("enum"), _shape(p) if "properties" in p else None)
        for n, p in schema.get("properties", {}).items()
    }
    return {"properties": props, "required": sorted(schema.get("required", []))}


def test_always_payload_fits_the_budget(specs):
    assert len(specs.payload) == len(ToolTierSettings().always)
    assert len(json.dumps(specs.payload)) <= MAX_PAYLOAD_CHARS
    for tool in specs.payload:
        assert len(json.dumps(tool)) <= MAX_TOOL_CHARS, tool["function"]["name"]


def test_payload_is_independent_of_registry_order(registry, specs):
    class _Shuffled:
        def __init__(self, tools):
            items = list(tools.items())
            random.Random(7).shuffle(items)
            self.tools = dict(reversed(items))

    again = build_tier_specs(_Shuffled(registry.tools), EXCLUDE, ToolTierSettings())
    assert json.dumps(again.payload) == json.dumps(specs.payload)
    assert again.lookup == specs.lookup


def test_sent_and_guard_views_agree_and_sent_view_is_slim(specs):
    for tool in specs.payload:
        name, sent = tool["function"]["name"], tool["function"]["parameters"]
        guard = specs.schemas[name]
        assert _shape(sent) == _shape(guard), name
        assert guard["additionalProperties"] is False
        text = json.dumps(sent)
        for key in ('"title"', '"default"', '"examples"', '"additionalProperties"'):
            assert key not in text, (name, key)
        assert sent.get("required", None) != []
        assert not CONTEXT_PARAMS & set(sent["properties"]), name


def test_lookup_entries_carry_the_full_guard_schema(specs):
    assert specs.lookup
    for entry in specs.lookup:
        assert entry["parameters"] == specs.schemas[entry["name"]]


@pytest.mark.parametrize(
    "name, good, wrong_type, missing, unknown",
    [
        (
            "write_file",
            {"file_path": "a.txt", "content": "hi"},
            {"file_path": "a.txt", "content": 3},
            {"file_path": "a.txt"},
            {"file_path": "a.txt", "content": "hi", "overwrite": True},
        ),
        (
            "document_search",
            {"query": "garden"},
            {"query": "garden", "max_results": "five"},
            {"max_results": 3},
            {"query": "garden", "folder": "x"},
        ),
        (
            "calculator",
            {"expression": "1 + 2"},
            {"expression": 12},
            {},
            {"expression": "1 + 2", "precision": 2},
        ),
    ],
)
def test_guard_still_rejects_bad_calls(specs, name, good, wrong_type, missing, unknown):
    schema = specs.schemas[name]
    assert validate(good, schema) == []
    for bad in (wrong_type, missing, unknown):
        assert validate(bad, schema), (name, bad)


def test_context_params_supplied_are_rejected(specs):
    schema = specs.schemas["read_conversation_history"]
    for param in CONTEXT_PARAMS:
        assert validate({param: "x"}, schema), param
