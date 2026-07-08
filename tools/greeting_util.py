"""Small greeting helper — used as a live end-to-end test target for WITS's
self-repair capability (see tests/tools/test_greeting_util.py)."""


def format_greeting(name: str, excited: bool = False) -> str:
    """Return a greeting for name; add an exclamation mark if excited."""
    greeting = f"Hello, {name}"
    if excited:
        greeting += "!"
    return greeting
