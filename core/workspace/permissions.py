"""Scope to tool mapping for workspace clients (contract sections 3 and 5, design rule 2).

Shaped like ``core.guest_access.tools_for_role``: a fixed map, default deny. Scopes never imply
each other, and no scope resolves an approval (only the owner route does).
"""

from collections.abc import Iterable

from .store import WorkspaceError

SCOPE_TOOLS: dict[str, frozenset[str]] = {
    "read": frozenset({"get_project_state"}),
    "write": frozenset({"claim_task", "append_handoff", "link_artifact"}),
    "decide:request": frozenset({"request_decision"}),
}


def tools_for_scopes(scopes: Iterable[str]) -> frozenset[str]:
    """Return the union of tools the given scopes reach; unknown scopes reach nothing."""
    allowed: set[str] = set()
    for scope in scopes:
        allowed |= SCOPE_TOOLS.get(scope, frozenset())
    return frozenset(allowed)


def check(identity, tool: str, project: str) -> None:
    """Refuse a call the identity is not allowed to make.

    Args:
        identity: A ``ClientIdentity`` (anything with ``scopes`` and ``projects``).
        tool: The tool being called.
        project: The project argument of the call.

    Raises:
        WorkspaceError: ``forbidden_scope`` when no granted scope maps to the tool (unknown tools
            included); ``forbidden_project`` when the project is not in the client's list and the
            list is not ``*``.
    """
    if tool not in tools_for_scopes(identity.scopes):
        raise WorkspaceError("forbidden_scope", "this client's scopes do not allow that tool")
    if "*" not in identity.projects and project not in identity.projects:
        raise WorkspaceError("forbidden_project", "this client may not use that project")
