"""Slash command registry for the web chat composer.

Commands with ``dispatch: chat`` are sent as a normal chat message (e.g. owner
/process controls). Commands with ``dispatch: client`` run in the browser.
"""

from __future__ import annotations

from typing import Any

# roles: owner | guest — who may see the command in the picker
SLASH_COMMANDS: list[dict[str, Any]] = [
    {
        "id": "help",
        "command": "/help",
        "aliases": [],
        "description": "Show available slash commands",
        "roles": ["owner", "guest"],
        "dispatch": "client",
        "client_action": "help",
    },
    {
        "id": "new",
        "command": "/new",
        "aliases": ["/newchat"],
        "description": "Start a new chat session",
        "roles": ["owner", "guest"],
        "dispatch": "client",
        "client_action": "new_chat",
    },
    {
        "id": "export_verbose",
        "command": "/export verbose",
        "aliases": [],
        "description": "Export chat with tool/action/observation traces",
        "roles": ["owner", "guest"],
        "dispatch": "client",
        "client_action": "export_verbose",
    },
    {
        "id": "export",
        "command": "/export",
        "aliases": [],
        "description": "Export this chat to a text file in var/exports/",
        "roles": ["owner", "guest"],
        "dispatch": "client",
        "client_action": "export",
    },
    {
        "id": "chats",
        "command": "/chats",
        "aliases": [],
        "description": "Open the Chats side panel",
        "roles": ["owner", "guest"],
        "dispatch": "client",
        "client_action": "panel",
        "panel_tab": "chats",
    },
    {
        "id": "tools",
        "command": "/tools",
        "aliases": [],
        "description": "Open the Tools side panel",
        "roles": ["owner", "guest"],
        "dispatch": "client",
        "client_action": "panel",
        "panel_tab": "tools",
    },
    {
        "id": "memory",
        "command": "/memory",
        "aliases": [],
        "description": "Open the Memory side panel",
        "roles": ["owner"],
        "dispatch": "client",
        "client_action": "panel",
        "panel_tab": "memory",
    },
    {
        "id": "docs",
        "command": "/docs",
        "aliases": ["/documents"],
        "description": "Open the Documents side panel",
        "roles": ["owner"],
        "dispatch": "client",
        "client_action": "panel",
        "panel_tab": "docs",
    },
    {
        "id": "settings",
        "command": "/settings",
        "aliases": [],
        "description": "Open system settings",
        "roles": ["owner"],
        "dispatch": "client",
        "client_action": "navigate",
        "href": "/settings",
    },
    {
        "id": "personality",
        "command": "/personality",
        "aliases": [],
        "description": "Open the personality questionnaire",
        "roles": ["owner"],
        "dispatch": "client",
        "client_action": "navigate",
        "href": "/personality",
    },
    {
        "id": "mcp",
        "command": "/mcp",
        "aliases": [],
        "description": "Open MCP server manager",
        "roles": ["owner"],
        "dispatch": "client",
        "client_action": "navigate",
        "href": "/mcp",
    },
    {
        "id": "restart",
        "command": "/restart",
        "aliases": [],
        "description": "Restart the WITS web process (owner token required)",
        "roles": ["owner"],
        "dispatch": "chat",
        "dangerous": True,
    },
    {
        "id": "shutdown",
        "command": "/shutdown",
        "aliases": ["/stop", "/kill", "/quit"],
        "description": "Force-stop the WITS web process (owner token required)",
        "roles": ["owner"],
        "dispatch": "chat",
        "dangerous": True,
    },
]


def list_slash_commands(role: str | None) -> list[dict[str, Any]]:
    """Return slash commands visible to the caller."""
    normalized = role if role in ("owner", "guest") else "owner"
    visible: list[dict[str, Any]] = []
    for cmd in SLASH_COMMANDS:
        if normalized not in cmd.get("roles", []):
            continue
        visible.append(
            {
                "id": cmd["id"],
                "command": cmd["command"],
                "aliases": list(cmd.get("aliases") or []),
                "description": cmd["description"],
                "dispatch": cmd["dispatch"],
                "dangerous": bool(cmd.get("dangerous")),
                **{k: cmd[k] for k in ("client_action", "panel_tab", "href") if k in cmd},
            }
        )
    return visible
