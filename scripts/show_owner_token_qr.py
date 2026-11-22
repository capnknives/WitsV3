"""Show the WITS owner token as a QR code in the console, for phone sign-in.

Run it on the PC (double-click ``show_owner_token_qr.bat``), scan the QR with the phone camera,
copy the text and paste it into the token box on the WITS page. The QR holds the bare token,
not a link, because the web page ignores owner tokens in URLs on any non-localhost address.

The token is never written anywhere: no file, log, clipboard or temp file.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

TOKEN_KEY = "WITSV3_WEB_TOKEN"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_token(env_path: Path) -> str:
    """Read the owner token from a ``.env`` file without loading the app or its config.

    Args:
        env_path: Path to the ``.env`` file.

    Returns:
        The token with surrounding whitespace and quotes removed (last assignment wins),
        or an empty string if the file is missing or the key is unset.
    """
    try:
        text = env_path.read_text(encoding="utf-8-sig")
    except OSError:
        return ""
    token = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        if key != TOKEN_KEY:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        token = value.strip()
    return token


def render_qr(token: str, encoding: str | None) -> str:
    """Draw the token as a console QR code that the given output encoding can display.

    Args:
        token: Text to encode (bare, not a URL).
        encoding: Encoding of the output stream; None is treated as ASCII.

    Returns:
        Multi-line text. Uses half-block characters when the encoding can show them,
        otherwise a plain ``#`` rendering of the same module matrix.
    """
    import qrcode

    qr = qrcode.QRCode(border=2)
    qr.add_data(token)
    buf = io.StringIO()
    qr.print_ascii(out=buf, tty=False, invert=True)
    art = buf.getvalue()
    try:
        art.encode(encoding or "ascii")
        return art
    except (UnicodeEncodeError, LookupError):
        qr.make()
        rows = ["".join("##" if cell else "  " for cell in row) for row in qr.get_matrix()]
        return "\n".join(rows) + "\n"


def clear_console() -> None:
    """Clear the console so the QR does not stay in the scrollback."""
    sys.stdout.write("\033[3J\033[2J\033[H")
    sys.stdout.flush()
    if os.name == "nt":
        os.system("cls")  # noqa: S605 - fixed literal command, no user input


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description="Show the WITS owner token as a QR code.")
    parser.add_argument("--text", action="store_true", help="also print the token as text")
    parser.add_argument(
        "--no-wait", action="store_true", help="do not wait for Enter or clear the screen"
    )
    return parser


def main(argv: list[str] | None = None, root: Path | None = None) -> int:
    """Entry point.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).
        root: Project root containing ``.env`` (defaults to the parent of ``scripts/``).

    Returns:
        0 on success, 1 if the token is missing.
    """
    args = build_parser().parse_args(argv)
    env_path = (root or PROJECT_ROOT) / ".env"
    token = parse_token(env_path)
    if not token:
        print(f"No {TOKEN_KEY} found in {env_path}. Nothing to show.")
        return 1

    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        try:
            reconfigure(encoding="utf-8")
        except (OSError, ValueError):
            pass

    print("WITS owner token as a QR code. Anyone who can see this screen can sign in as owner.")
    print()
    print(render_qr(token, getattr(sys.stdout, "encoding", None)))
    if args.text:
        print(f"Token: {token}")
        print()
    print("On your phone:")
    print("  1. Scan the QR code with the camera.")
    print("  2. Copy the text it shows.")
    print("  3. Open WITS at your tailnet address with /?owner=1 on the end and paste it")
    print("     into the token box.")
    print()

    if args.no_wait:
        return 0
    try:
        input("Press Enter to clear the screen and close")
    except EOFError:
        pass
    clear_console()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
