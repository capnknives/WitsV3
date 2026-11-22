"""request_folder_access: ask the owner for access to a folder outside the WITS folders.

The tool only queues a request (folder access contract, section 5). The owner answers in the
Inbox; the grant, if any, is created there. The tool never waits or polls.
"""

from typing import Any

from core.base_tool import BaseTool
from core.tool_context import current_tool_context


class FolderAccessRequestTool(BaseTool):
    """Queue an owner approval for read, read_write or index access to one folder."""

    def __init__(self):
        super().__init__(
            name="request_folder_access",
            description=(
                "Ask the owner for access to a folder outside the WITS folders. Use only after a "
                "file tool refused a path and said the owner can approve access. Returns at once; "
                "the owner decides in the Inbox."
            ),
        )

    async def execute(self, path: str, mode: str, reason: str, user_role: str = "guest") -> str:
        """Queue a folder access request for the owner.

        Args:
            path: Absolute folder path; a last part with a suffix is requested as its folder.
            mode: ``read``, ``read_write`` or ``index``.
            reason: Why access is needed; shown to the owner.
            user_role: Ignored; accepted only so the old path's injected argument does not fail.
                The role comes from the tool context alone.

        Returns:
            A plain sentence for the model: the approval id and what to tell the owner, or why
            the request was not made.
        """
        # Reason: only the owner may widen what WITS can reach. Fail closed: with no tool
        # context (an argument could be supplied by anyone) the call is refused.
        ctx = current_tool_context()
        if ctx is None or ctx.user_role != "owner":
            return "Error: only the owner can request folder access."
        # Reason: imported here so tool discovery works without aiosqlite when the workspace
        # is off.
        from core.config import load_config
        from core.workspace import folder_access

        db = folder_access.current_db()
        if db is None:
            return "Error: folder access requests need the workspace, which is not running."
        config = load_config()
        ttl = config.workspace.approval_ttl_minutes * 60
        result = await folder_access.request(db, path, mode, reason, ttl_seconds=ttl, config=config)
        status = result["status"]
        if status == "already_allowed":
            return f"WITS can already use {result['path']}; no request is needed."
        if status == "refused":
            return f"Error: request not made ({result['code']}): {result['message']}"
        return (
            f"Request {result['approval_id']} for {mode} access to {result['path']} is waiting "
            "for the owner. Tell the owner a folder access request is waiting in the Inbox. "
            "Do not wait or ask again; access starts only after the owner approves."
        )

    def get_schema(self) -> dict[str, Any]:
        """Get tool schema."""
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path of the folder"},
                "mode": {
                    "type": "string",
                    "enum": ["read", "read_write", "index"],
                    "description": "Access wanted: read, read_write or index",
                },
                "reason": {"type": "string", "description": "Why access is needed"},
            },
            "required": ["path", "mode", "reason"],
        }
