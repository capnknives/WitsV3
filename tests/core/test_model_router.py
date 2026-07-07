"""
Tests for the smart model router (core/model_router.py).

The router classifies raw user text — never full prompt templates — and
picks the Ollama model per request: trivial chat to the small fast model,
code work to the coder model, everything else to the default.
"""

import pytest

from core.config import WitsV3Config
from core.model_router import ModelRouter


@pytest.fixture
def router():
    return ModelRouter(WitsV3Config())


class TestClassify:
    def test_greetings_are_trivial(self, router):
        for text in ["hi", "Hello!", "hey, how are you today?", "good morning", "thanks!"]:
            assert router.classify(text) == "trivial", text

    def test_short_small_talk_is_trivial(self, router):
        assert router.classify("what's your name?") == "trivial"

    def test_empty_text_is_trivial(self, router):
        assert router.classify("") == "trivial"
        assert router.classify(None) == "trivial"

    def test_code_keywords_route_to_code(self, router):
        for text in [
            "write a python function to sort a list",
            "why does this traceback happen?",
            "help me debug my script",
            "refactor this class",
            "I'm getting a syntax error",
        ]:
            assert router.classify(text) == "code", text

    def test_code_fence_routes_to_code(self, router):
        assert router.classify("what does this do?\n```\nx = 1\n```") == "code"

    def test_word_boundaries_no_substring_match(self, router):
        # "api" inside "rapid", "git" inside "digital" must not trigger code
        assert router.classify("things are moving at a rapid pace in digital marketing") != "code"

    def test_short_task_requests_are_not_trivial(self, router):
        # Short but tool-shaped: must not go to the small model
        for text in [
            "search the web for ollama news",
            "summarize my audit report",
            "create a file called notes.txt",
        ]:
            assert router.classify(text) != "trivial", text

    def test_long_prose_is_complex(self, router):
        text = (
            "I've been thinking about the way my weekly schedule works and I want you to "
            "look at the trade-offs between the different approaches we discussed "
            "yesterday and tell me which one makes the most sense going forward, "
            "considering everything we know."
        )
        assert router.classify(text) == "complex"


class TestRoute:
    def test_trivial_routes_to_trivial_model(self, router):
        assert router.route("hi there") == router.settings.trivial_model

    def test_code_routes_to_code_model(self, router):
        assert router.route("debug this python error") == router.settings.code_model

    def test_complex_routes_to_default(self, router):
        assert router.route("plan my week", default="qwen3:8b") == "qwen3:8b"

    def test_complex_without_default_uses_complex_model(self, router):
        long_text = "analyze the report " * 20
        assert router.route(long_text) == router.settings.complex_model

    def test_allow_trivial_false_keeps_default(self, router):
        # Orchestrator ReAct loop: trivial goals must not hit the small model
        assert router.route("hi", default="qwen3:8b", allow_trivial=False) == "qwen3:8b"

    def test_allow_trivial_false_still_routes_code(self, router):
        assert (
            router.route("fix my python script", default="qwen3:8b", allow_trivial=False)
            == router.settings.code_model
        )

    def test_disabled_routing_returns_default(self):
        config = WitsV3Config()
        config.model_routing.enabled = False
        router = ModelRouter(config)
        assert router.route("hi", default="qwen3:8b") == "qwen3:8b"
        assert router.route("debug my python code", default="qwen3:8b") == "qwen3:8b"

    def test_custom_models_from_config(self):
        config = WitsV3Config()
        config.model_routing.trivial_model = "tiny:1b"
        config.model_routing.code_model = "coder:13b"
        router = ModelRouter(config)
        assert router.route("hello") == "tiny:1b"
        assert router.route("write a regex for emails") == "coder:13b"
