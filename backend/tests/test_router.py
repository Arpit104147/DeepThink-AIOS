"""
Tests for TaskRouter.classify_task deterministic keyword fast-path.

The fast-path runs entirely before any LLM call, so these need no model and
are safe for CI. They pin the routing contract: a request with a strong
domain keyword must land in the right pipeline every time, regardless of the
(stochastic) router model.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from backend.orchestrator.router import TaskRouter  # noqa: E402


class TestChipDesignFastPath:
    PROMPTS = [
        "write a verilog binary counter",
        "design a carry-lookahead adder in systemverilog",
        "generate a 2nm GAAFET layout with gdstk",
        "simulate this spice netlist with ngspice",
        "implement a risc-v core (rv32i)",
    ]

    def test_all_route_to_chip_design(self):
        for p in self.PROMPTS:
            assert TaskRouter.classify_task(None, None, p) == "CHIP_DESIGN", p


class TestReasoningFastPath:
    def test_math_physics_proofs_route_to_reasoning(self):
        for p in ("prove that sqrt(2) is irrational",
                  "derive the lorentz transformations",
                  "give the exact integral of x*sin(x)"):
            assert TaskRouter.classify_task(None, None, p) == "REASONING", p

    def test_reasoning_keyword_with_code_word_does_not_hijack_coding(self):
        # "derive ... in python" mentions code, so the reasoning fast-path must
        # defer (it explicitly excludes prompts naming a programming language).
        got = TaskRouter.classify_task(None, None, "derive the formula then write it in python")
        assert got == "CODING", got


class TestCodingFastPath:
    def test_programming_prompts_route_to_coding(self):
        for p in ("write a python function for an lru cache",
                  "implement a binary search tree in c++",
                  "write a bash script to rotate logs",
                  "add a pytest unit test for this module"):
            assert TaskRouter.classify_task(None, None, p) == "CODING", p


class TestFallbackToModel:
    def test_no_keyword_and_no_model_defaults_to_simple(self):
        # With no keyword match and orchestrator=None (so _call_model would
        # raise), classify_task must swallow the error and default to SIMPLE.
        got = TaskRouter.classify_task(None, None, "what is the capital of japan")
        assert got == "SIMPLE"

    def test_model_classification_is_used_when_no_keyword(self):
        class FakeOrch:
            def _call_model(self, llm, prompt, max_tokens=64, temperature=0.1):
                return "This request is about general theory, so: REASONING"
            def _strip_thinking(self, raw):
                return raw
        got = TaskRouter.classify_task(FakeOrch(), object(), "tell me about entropy")
        assert got == "REASONING"

    def test_last_category_match_wins(self):
        # Guards the rfind-based parser: trailing category token must win.
        class FakeOrch:
            def _call_model(self, llm, prompt, max_tokens=64, temperature=0.1):
                return "It is not CODING nor REASONING. Final answer: SIMPLE"
            def _strip_thinking(self, raw):
                return raw
        assert TaskRouter.classify_task(FakeOrch(), object(), "ambiguous") == "SIMPLE"


class TestLooksNumericProblem:
    def test_detects_numeric_keywords(self):
        assert TaskRouter.looks_numeric_problem("calculate the derivative of x^2") is True

    def test_plain_prose_is_not_numeric(self):
        assert TaskRouter.looks_numeric_problem("tell me a story about a dog") is False
