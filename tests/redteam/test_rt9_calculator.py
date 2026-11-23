"""RT-9: CalculatorTool must give exact, safe answers (fixed by core/safe_calc.py)."""

import subprocess
import sys
from pathlib import Path

import pytest

from core.tool_registry import CalculatorTool

ROOT = Path(__file__).resolve().parents[2]


async def calc(expr: str) -> str:
    return await CalculatorTool().execute(expression=expr)


def calc_in_subprocess(expr: str, timeout: float = 2.0) -> str:
    """Run the tool in a child process so a runaway computation is killed, not left hanging."""
    code = (
        "import asyncio, sys\n"
        "from core.tool_registry import CalculatorTool\n"
        "print(asyncio.run(CalculatorTool().execute(expression=sys.argv[1])))\n"
    )
    try:
        out = subprocess.run(
            [sys.executable, "-c", code, expr],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    return out.stdout.strip()


# ---- former holes (fixed in RT-9b) -----------------------------------------------


@pytest.mark.asyncio
async def test_thousands_commas_division():
    res = await calc("84,913 * 7,629 / 13.4")
    assert "(" not in res and "," not in res
    assert round(float(res), 3) == 48343378.881


@pytest.mark.asyncio
async def test_thousands_commas_product():
    assert float(await calc("1,250 * 48")) == 60000


@pytest.mark.asyncio
async def test_decimal_artefact_add():
    assert await calc("0.1 + 0.2") == "0.3"


@pytest.mark.asyncio
async def test_decimal_artefact_multiply():
    assert await calc("1.1 * 3") == "3.3"


@pytest.mark.asyncio
async def test_large_exact_division():
    assert await calc("123456789012345678901234567890 / 10") == "12345678901234567890123456789"


def test_tower_power_refused_quickly():
    assert calc_in_subprocess("9**9**9").startswith("Error")


@pytest.mark.asyncio
async def test_percent_suffix():
    assert float(await calc("240 * 15%")) == 36


@pytest.mark.asyncio
async def test_round_function():
    assert float(await calc("round(48343378.8806, 3)")) == 48343378.881


@pytest.mark.asyncio
async def test_caret_is_power():
    assert await calc("2^10") == "1024"


# ---- pins ------------------------------------------------------------------------------


def test_big_power_refused_quickly():
    # The size estimate refuses this before computing anything.
    assert calc_in_subprocess("2**99999999").startswith("Error")


@pytest.mark.asyncio
async def test_simple_addition():
    assert await calc("2 + 2") == "4"


@pytest.mark.asyncio
async def test_sqrt():
    assert await calc("sqrt(16)") in ("4", "4.0")


@pytest.mark.asyncio
async def test_division_by_zero_is_error_string():
    assert (await calc("1 / 0")).startswith("Error")


@pytest.mark.asyncio
@pytest.mark.parametrize("expr", ["__import__('os')", "import os", "os.system"])
async def test_code_and_letters_refused(expr):
    assert (await calc(expr)).startswith("Error")
