"""Exact, bounded arithmetic for the calculator tool, without eval.

The expression is normalised (thousands commas, ``^``, ``%``), parsed with ``ast`` and
walked against an explicit whitelist. Numbers are ``fractions.Fraction`` so decimal input
stays exact; only sqrt of a non-square and non-integer powers fall back to float.
"""

import ast
import math
import re
from decimal import ROUND_HALF_UP, Decimal, localcontext
from fractions import Fraction

Number = Fraction | float

MAX_LENGTH = 500
# Reason: a 500-char expression can chain ~250 operators; real calculations are far
# shorter, and this keeps recursion well under Python's limit.
MAX_DEPTH = 120
MAX_DIGITS = 10_000
MAX_BITS = int(MAX_DIGITS * math.log2(10))
# Reason: 15 significant digits is what a float can carry reliably, so a rounded
# non-terminating result never claims more precision than a typical calculator shows.
SIG_DIGITS = 15
MAX_ROUND_PLACES = 100

# A thousands-grouped number: 1-3 digits, then one or more ",ddd" groups, not part of a
# longer digit run or a fractional part. "round(2.567, 2)" and "round(2.567,200)" keep
# their comma. Ambiguous: "round(1,234)" is read as the number 1234 (one argument),
# because "1,234" is far more often a grouped number than a two-argument call.
_GROUPED = re.compile(r"(?<![\d.])\d{1,3}(?:,\d{3})+(?![\d])")
_PERCENT = re.compile(r"(\d+(?:\.\d*)?|\.\d+)\s*%")
_EXPONENT = re.compile(r"[eE]([+-]?\d+)$")

_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Pow)
_FUNC_ARITY = {"sqrt": (1, 1), "abs": (1, 1), "round": (1, 2)}


class CalcError(ValueError):
    """Raised for any expression the calculator refuses or cannot compute."""


def evaluate(expression: str) -> str:
    """Evaluate an arithmetic expression exactly and return the result as text.

    Args:
        expression: Arithmetic such as ``"84,913 * 7,629 / 13.4"`` or ``"240 * 15%"``.

    Returns:
        The result as a plain decimal string, or a string starting with ``"Error"``.
    """
    try:
        src = normalise(expression)
        tree = ast.parse(src, mode="eval")
        _check_depth(tree.body, 1)
        return format_number(_eval(tree.body, src))
    except ZeroDivisionError:
        return "Error: division by zero"
    except CalcError as e:
        return f"Error: {e}"
    except (SyntaxError, ValueError, TypeError):
        return "Error: invalid expression"
    except (OverflowError, MemoryError, RecursionError):
        return "Error: result too large"


def normalise(expression: str) -> str:
    """Strip thousands commas and translate ``^`` and ``%`` into Python syntax.

    Args:
        expression: Raw user expression.

    Returns:
        The expression ready for ``ast.parse``.

    Raises:
        CalcError: If the expression is empty, too long or contains ``__``.
    """
    expr = str(expression or "").strip()
    if not expr:
        raise CalcError("No expression provided")
    if len(expr) > MAX_LENGTH:
        raise CalcError(f"expression longer than {MAX_LENGTH} characters")
    if "__" in expr:
        raise CalcError("invalid expression")
    expr = expr.replace("\n", " ").replace("\r", " ")
    expr = _GROUPED.sub(lambda m: m.group(0).replace(",", ""), expr)
    expr = expr.replace("^", "**")
    return _PERCENT.sub(r"(\1/100)", expr)


def _check_depth(node: ast.AST, depth: int) -> None:
    if depth > MAX_DEPTH:
        raise CalcError("expression too deeply nested")
    for child in ast.iter_child_nodes(node):
        _check_depth(child, depth + 1)


def _eval(node: ast.AST, src: str) -> Number:
    if isinstance(node, ast.Constant):
        return _literal(node, src)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        val = _eval(node.operand, src)
        return -val if isinstance(node.op, ast.USub) else val
    if isinstance(node, ast.BinOp) and isinstance(node.op, _BINOPS):
        left, right = _eval(node.left, src), _eval(node.right, src)
        if isinstance(node.op, ast.Pow):
            return _check_size(_power(left, right))
        if isinstance(node.op, ast.Add):
            return _check_size(left + right)
        if isinstance(node.op, ast.Sub):
            return _check_size(left - right)
        if isinstance(node.op, ast.Mult):
            return _check_size(left * right)
        if right == 0:
            raise ZeroDivisionError
        if isinstance(node.op, ast.Div):
            return _check_size(left / right)
        return _check_size(Fraction(math.floor(left / right)))
    if isinstance(node, ast.Call):
        return _call(node, src)
    if isinstance(node, ast.Name):
        raise CalcError(f"unknown name '{node.id}'")
    raise CalcError(f"unsupported element '{type(node).__name__}'")


def _literal(node: ast.Constant, src: str) -> Number:
    value = node.value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalcError("only numbers are allowed")
    if isinstance(value, int):
        return Fraction(value)
    # Reason: read the decimal from its source text so "0.1" is exactly 1/10, not the
    # nearest binary float.
    text = (ast.get_source_segment(src, node) or "").replace("_", "")
    exp = _EXPONENT.search(text)
    if exp and abs(int(exp.group(1))) > MAX_DIGITS:
        raise CalcError("number too large")
    return Fraction(text)


def _power(base: Number, exp: Number) -> Number:
    if isinstance(exp, Fraction) and exp.denominator == 1:
        e = exp.numerator
        if isinstance(base, Fraction):
            if base in (0, 1, -1):
                return base**e
            # Reason: estimate the result size from logarithms before computing, so a
            # tower like 9**9**9 is refused instantly instead of hanging the process.
            bits = max(math.log2(abs(base.numerator)), math.log2(base.denominator))
            if abs(e) * bits > MAX_BITS:
                raise CalcError(f"result would exceed {MAX_DIGITS} digits")
            return base**e
        return _finite(base**e)
    # Non-integer exponent: cannot be exact, so compute in float precision.
    b, x = float(base), float(exp)
    if b < 0:
        raise CalcError("negative number to a fractional power")
    return _finite(b**x)


def _call(node: ast.Call, src: str) -> Number:
    if not isinstance(node.func, ast.Name) or node.func.id.lower() not in _FUNC_ARITY:
        name = node.func.id if isinstance(node.func, ast.Name) else "expression"
        raise CalcError(f"function '{name}' is not allowed (use sqrt, round, abs)")
    name = node.func.id.lower()
    low, high = _FUNC_ARITY[name]
    if node.keywords or not low <= len(node.args) <= high:
        raise CalcError(f"wrong arguments for {name}()")
    args = [_eval(a, src) for a in node.args]
    if name == "abs":
        return abs(args[0])
    if name == "sqrt":
        return _sqrt(args[0])
    return _round(args[0], args[1] if len(args) > 1 else Fraction(0))


def _sqrt(x: Number) -> Number:
    if x < 0:
        raise CalcError("square root of a negative number")
    if isinstance(x, Fraction):
        num, den = math.isqrt(x.numerator), math.isqrt(x.denominator)
        if num * num == x.numerator and den * den == x.denominator:
            return Fraction(num, den)
    return _finite(math.sqrt(float(x)))


def _round(x: Number, places: Number) -> Number:
    if not (isinstance(places, Fraction) and places.denominator == 1):
        raise CalcError("round() places must be a whole number")
    if abs(places) > MAX_ROUND_PLACES:
        raise CalcError(f"round() places must be within {MAX_ROUND_PLACES}")
    # Reason: half-up (2.5 -> 3) matches what people expect from a calculator, unlike
    # Python's banker's rounding.
    exact = Decimal(repr(x)) if isinstance(x, float) else _to_decimal(x)
    quantum = Decimal(1).scaleb(-int(places))
    with localcontext() as ctx:
        ctx.prec = MAX_DIGITS + MAX_ROUND_PLACES + 10
        return Fraction(exact.quantize(quantum, rounding=ROUND_HALF_UP))


def _to_decimal(x: Fraction) -> Decimal:
    with localcontext() as ctx:
        ctx.prec = MAX_DIGITS + MAX_ROUND_PLACES + 10
        return Decimal(x.numerator) / Decimal(x.denominator)


def _finite(x: float | complex) -> float:
    if isinstance(x, complex) or not math.isfinite(x):
        raise CalcError("result too large or not a real number")
    return x


def _check_size(x: Number) -> Number:
    if isinstance(x, float):
        return _finite(x)
    if max(x.numerator.bit_length(), x.denominator.bit_length()) > MAX_BITS:
        raise CalcError(f"result would exceed {MAX_DIGITS} digits")
    return x


def format_number(x: Number) -> str:
    """Render a result: integers as digits, terminating decimals exactly, others rounded.

    Args:
        x: An exact ``Fraction`` or an inexact ``float``.

    Returns:
        A plain (non-scientific) decimal string without trailing zeros.
    """
    if isinstance(x, float):
        return _plain(Decimal(repr(_finite(x))))
    # Reason: Decimal avoids Python's 4300-digit int->str limit for big exact results.
    if x.denominator == 1:
        return format(Decimal(x.numerator), "f")
    den = x.denominator
    for p in (2, 5):
        while den % p == 0:
            den //= p
    if den == 1:
        return _plain(_to_decimal(x))
    with localcontext() as ctx:
        ctx.prec = SIG_DIGITS
        return _plain(Decimal(x.numerator) / Decimal(x.denominator))


def _plain(d: Decimal) -> str:
    text = format(d, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text
