from tools.greeting_util import format_greeting


def test_format_greeting_plain():
    assert format_greeting("Wits") == "Hello, Wits"


def test_format_greeting_excited():
    assert format_greeting("Wits", excited=True) == "Hello, Wits!"
