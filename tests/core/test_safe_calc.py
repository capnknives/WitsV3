"""Unit tests for core.safe_calc: exact, bounded, eval-free arithmetic."""

import time

import pytest

from core.safe_calc import MAX_LENGTH, evaluate


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("2 + 3 * 4", "14"),
        ("(2 + 3) * 4", "20"),
        ("2 * (3 + (4 - 1)) / 3", "4"),
        ("10 - 4 - 3", "3"),
        ("2 ** 3 ** 2", "512"),
        ("-3 + 5", "2"),
        ("-(2 + 3) * 2", "-10"),
        ("--4", "4"),
        ("-2 ** 2", "-4"),
        ("7 // 2", "3"),
        ("1.5e3 + 1", "1501"),
        ("1_000 + 1", "1001"),
    ],
)
def test_precedence_brackets_unary(expr, expected):
    assert evaluate(expr) == expected


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("sqrt(16)", "4"),
        ("sqrt(2.25)", "1.5"),
        ("abs(-2.5)", "2.5"),
        ("round(2.5)", "3"),
        ("round(-2.5)", "-3"),
        ("round(2.567, 2)", "2.57"),
        ("round(1234.5, -2)", "1200"),
        ("round(sqrt(abs(-2)), 4)", "1.4142"),
        ("sqrt(round(15.6))", "4"),
        ("SQRT(9)", "3"),
    ],
)
def test_functions_and_nesting(expr, expected):
    assert evaluate(expr) == expected


def test_sqrt_non_square_is_float_precision():
    assert evaluate("sqrt(2)") == "1.4142135623730951"


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("1,250 * 48", "60000"),
        ("1,250,000.5 + 0.5", "1250001"),
        ("round(2.567, 2)", "2.57"),
        ("round(2.567,2)", "2.57"),
        ("round(2.567,100)", "2.567"),  # fraction before comma: an argument
        # Ambiguous by design: "1,234" reads as one grouped number, not round(1, 234).
        ("round(1,234)", "1234"),
    ],
)
def test_comma_handling(expr, expected):
    assert evaluate(expr) == expected


@pytest.mark.parametrize("expr", ["1234,567", "1,23", "(1, 2)"])
def test_bad_commas_are_errors(expr):
    assert evaluate(expr).startswith("Error")


@pytest.mark.parametrize(
    "expr, expected",
    [("240 * 15%", "36"), ("50%", "0.5"), ("200 + 12.5%", "200.125"), ("2^10", "1024")],
)
def test_percent_and_caret(expr, expected):
    assert evaluate(expr) == expected


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("1/3", "0.333333333333333"),
        ("2/3", "0.666666666666667"),
        ("10/4", "2.5"),
        ("1/8", "0.125"),
        ("0.1 + 0.2", "0.3"),
        ("1e-7 * 1", "0.0000001"),
    ],
)
def test_exact_and_rounded_output(expr, expected):
    assert evaluate(expr) == expected


def test_big_exact_integer_beyond_str_limit():
    assert evaluate("10 ** 5000") == "1" + "0" * 5000


@pytest.mark.parametrize(
    "expr",
    ["9**9**9", "2**99999999", "10**20000", "(10**9000) * (10**9000)", "1e99999 * 1", "2.5**1e9"],
)
def test_power_and_size_limits(expr):
    start = time.monotonic()
    assert evaluate(expr).startswith("Error")
    assert time.monotonic() - start < 1.0


@pytest.mark.parametrize(
    "expr, expected",
    [("1**999999999999", "1"), ("(-1)**999999999999", "-1"), ("4**0.5", "2"), ("2**-2", "0.25")],
)
def test_cheap_powers_still_work(expr, expected):
    assert evaluate(expr) == expected


def test_length_limit():
    assert evaluate("1+" * (MAX_LENGTH // 2) + "1").startswith("Error")


def test_depth_limit():
    # Brackets add no AST nodes, so nest operators and calls instead.
    assert evaluate("-" * 150 + "1").startswith("Error")
    assert evaluate("abs(-" * 70 + "1" + ")" * 70).startswith("Error")
    assert evaluate("abs(" * 30 + "1" + ")" * 30) == "1"


@pytest.mark.parametrize(
    "expr",
    [
        "x + 1",
        "pi",
        "(1).real",
        "math.sqrt(4)",
        "pow(2, 3)",
        "max(1, 2)",
        "sqrt(4, 2)",
        "round(1, ndigits=2)",
        "lambda: 1",
        "[1, 2][0]",
        "'1' + '2'",
        "1 < 2",
        "(y := 3)",
        "True + 1",
        "1j * 2",
        "__import__('os')",
        "import os",
        "sqrt(-4)",
        "(-8) ** 0.5",
        "1 / 0",
        "5 // 0",
        "0 ** -1",
        "round(2, 0.5)",
        "",
    ],
)
def test_rejected_inputs(expr):
    assert evaluate(expr).startswith("Error")
