"""
MCP registry search for WitsV3.

Lets WITS (and the human, via the /mcp page) discover new MCP servers from the
official Model Context Protocol registry (registry.modelcontextprotocol.io),
and derive a runnable stdio command so a server can be installed with one click.

Design / safety:
  * SEARCH is read-only — it only queries the registry's public REST API.
  * Deriving a command does NOT run anything; it just builds the argv that
    `/api/mcp/servers/{name}/connect` would later execute (npx/uvx).
  * Actually installing (writing config) and connecting (running the command,
    which downloads+executes third-party code) stays a deliberate human action
    in the web UI — the agent-facing tool can search but not self-install.

The registry response schema (2025-09-29) looks like:
    {"servers": [
        {"server": {"name","description","repository":{...},"version",
                    "packages":[{"registryType","identifier","version",
                                 "runtimeHint","transport":{"type"},
                                 "runtimeArguments":[...],"packageArguments":[...],
                                 "environmentVariables":[...]}],
                    "remotes":[{"type","url"}]},
         "_meta": {"io.modelcontextprotocol.registry/official": {"isLatest": ...}}}
    ], "metadata": {"nextCursor": ...}}
"""

import logging
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)

DEFAULT_REGISTRY_URL = "https://registry.modelcontextprotocol.io"

# registryType -> the runtime that runs the package as a local stdio server.
_RUNTIME_FOR_REGISTRY = {
    "npm": "npx",
    "pypi": "uvx",
    "oci": "docker",
}

# Flags every one-shot docker-run stdio server needs: interactive stdin, and
# clean up the container on exit so repeated connects don't pile up.
_DOCKER_BASE_ARGS = ["run", "-i", "--rm"]


def _arg_values(args: Optional[List[Dict[str, Any]]]) -> List[str]:
    """Flatten a registry runtimeArguments/packageArguments list into argv parts.

    Only concrete values are emitted. Arguments that are pure user-supplied
    placeholders (no value and no default) are skipped so we never bake a
    literal `<PATH>` into the command — the user can edit the command after
    install if such an arg is needed.
    """
    out: List[str] = []
    for arg in args or []:
        if not isinstance(arg, dict):
            continue
        value = arg.get("value")
        if value in (None, ""):
            value = arg.get("default")
        if value in (None, ""):
            continue  # placeholder needing user input — skip
        if arg.get("type") == "named" and arg.get("name"):
            out.append(str(arg["name"]))
        out.append(str(value))
    return out


def _package_spec(registry_type: str, identifier: str, version: Optional[str]) -> str:
    """Build the versioned package token for the runtime (e.g. pkg@1.2.3)."""
    if registry_type == "oci":
        # Image ref may already carry a tag/digest (":" or "@sha256:" after the
        # last path segment) — only append one if the identifier doesn't.
        last_segment = identifier.rsplit("/", 1)[-1]
        if not version or ":" in last_segment or "@" in identifier:
            return identifier
        return f"{identifier}:{version}"
    if not version:
        return identifier
    if registry_type == "npm":
        return f"{identifier}@{version}"
    if registry_type == "pypi":
        return f"{identifier}=={version}"
    return identifier


def build_stdio_command(package: Dict[str, Any]) -> Optional[List[str]]:
    """Derive an argv list that runs `package` as a local stdio MCP server.

    Returns None for packages that can't be launched as a local stdio process
    here (non-stdio transport, or a registry type we don't auto-support).
    """
    transport = (package.get("transport") or {}).get("type", "stdio")
    if transport and transport != "stdio":
        return None

    registry_type = package.get("registryType", "")
    runtime = package.get("runtimeHint") or _RUNTIME_FOR_REGISTRY.get(registry_type)
    if not runtime:
        return None  # unknown registry type with no runtimeHint — needs manual setup

    identifier = package.get("identifier", "")
    if not identifier:
        return None

    runtime_args = _arg_values(package.get("runtimeArguments"))
    if runtime == "npx":
        # npx must be non-interactive or it prompts to install and hangs the server.
        if "-y" not in runtime_args and "--yes" not in runtime_args:
            runtime_args.insert(0, "-y")
    elif runtime == "docker":
        for flag in reversed(_DOCKER_BASE_ARGS):
            if flag not in runtime_args:
                runtime_args.insert(0, flag)
        # A plain subprocess inherits the host env automatically; a container
        # does not. `-e NAME` (no value) tells docker to forward that var from
        # the process env we already set (see MCPServer.env), so declared
        # env vars still reach the server without baking secrets into argv.
        for ev in _env_vars(package):
            if ev["name"] not in runtime_args:
                runtime_args += ["-e", ev["name"]]

    spec = _package_spec(registry_type, identifier, package.get("version"))
    package_args = _arg_values(package.get("packageArguments"))
    return [runtime, *runtime_args, spec, *package_args]


def _env_vars(package: Dict[str, Any]) -> List[Dict[str, Any]]:
    result = []
    for ev in package.get("environmentVariables") or []:
        if not isinstance(ev, dict) or not ev.get("name"):
            continue
        result.append(
            {
                "name": ev["name"],
                "description": ev.get("description", ""),
                "required": bool(ev.get("isRequired")),
                "secret": bool(ev.get("isSecret")),
                "default": ev.get("default"),
            }
        )
    return result


def normalize_server(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Turn one registry entry into WITS's flat, install-ready shape."""
    server = raw.get("server", raw)
    meta = (raw.get("_meta") or {}).get("io.modelcontextprotocol.registry/official", {})

    packages = []
    install: Optional[Dict[str, Any]] = None
    for pkg in server.get("packages") or []:
        command = build_stdio_command(pkg)
        entry = {
            "registry_type": pkg.get("registryType", ""),
            "identifier": pkg.get("identifier", ""),
            "version": pkg.get("version"),
            "transport": (pkg.get("transport") or {}).get("type", "stdio"),
            "command": command,
            "env_vars": _env_vars(pkg),
        }
        packages.append(entry)
        # First locally-runnable package becomes the one-click install target.
        if install is None and command is not None:
            install = entry

    repo = server.get("repository") or {}
    return {
        "name": server.get("name", ""),
        "description": server.get("description", ""),
        "version": server.get("version"),
        "repository": repo.get("url", ""),
        "is_latest": bool(meta.get("isLatest", True)),
        "packages": packages,
        "install": install,  # None => remote-only or needs manual setup
        # Every locally-runnable package option, so the UI can let the user
        # pick between e.g. an npm and a docker build instead of only ever
        # offering the first one silently chosen as `install`.
        "install_options": [p for p in packages if p["command"] is not None],
        "remotes": [r.get("url") for r in (server.get("remotes") or []) if r.get("url")],
    }


def _dedupe_latest(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Registries list every published version; keep one card per server name,
    preferring the entry flagged isLatest."""
    by_name: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        name = entry["name"]
        current = by_name.get(name)
        if current is None or (entry["is_latest"] and not current["is_latest"]):
            by_name[name] = entry
    return list(by_name.values())


# The registry's `search` matches the query as a literal phrase, so a
# multi-word query ("postgres database") finds nothing while "postgres" works.
# These common words are dropped when falling back to single keywords.
_STOPWORDS = {
    "the", "a", "an", "for", "and", "or", "to", "of", "with", "server", "servers",
    "mcp", "tool", "tools", "that", "can", "able", "let", "lets", "me", "my",
    "some", "way", "new", "install", "search", "find", "how", "do",
}


def _keywords(query: str) -> List[str]:
    """Significant lowercase words from the query (stopwords/short words removed)."""
    words = [w.lower() for w in "".join(c if c.isalnum() else " " for c in query).split() if len(w) >= 3]
    return [w for w in words if w not in _STOPWORDS] or words


def _candidate_queries(query: str) -> List[str]:
    """Ordered search terms to try: the full phrase first, then the most
    significant individual words (longest-first) as a fallback."""
    query = (query or "").strip()
    candidates = [query] if query else [""]
    for word in sorted(set(_keywords(query)), key=len, reverse=True):
        if word not in candidates:
            candidates.append(word)
    return candidates[:3]


def _relevance(entry: Dict[str, Any], keywords: List[str]) -> int:
    """How many query keywords appear in the server's name or description."""
    haystack = (entry.get("name", "") + " " + entry.get("description", "")).lower()
    return sum(1 for kw in keywords if kw in haystack)


async def _fetch_servers(session, url: str, term: str, limit: int) -> List[Dict[str, Any]]:
    params: Dict[str, Any] = {"limit": max(1, min(limit * 3, 100))}
    if term:
        params["search"] = term
    async with session.get(url, params=params) as resp:
        if resp.status != 200:
            raise RuntimeError(f"registry returned HTTP {resp.status}")
        data = await resp.json(content_type=None)
    return data.get("servers", data if isinstance(data, list) else [])


async def search_registry(
    query: str,
    limit: int = 10,
    registry_url: str = DEFAULT_REGISTRY_URL,
    timeout_seconds: float = 12.0,
) -> List[Dict[str, Any]]:
    """Search the MCP registry and return normalized, deduped server entries.

    Tries the full query first, then falls back to individual keywords if the
    phrase matches nothing. Raises RuntimeError on a non-200 response.
    """
    url = registry_url.rstrip("/") + "/v0/servers"
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)

    candidates = _candidate_queries(query)
    raw_servers: List[Dict[str, Any]] = []
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Try the full phrase first; if it hits, use only those (most relevant).
        raw_servers = await _fetch_servers(session, url, candidates[0], limit)
        if not raw_servers and len(candidates) > 1:
            # Phrase matched nothing — merge results across the keyword fallbacks
            # (dedupe by name happens below), so we don't miss the better term.
            for term in candidates[1:]:
                raw_servers.extend(await _fetch_servers(session, url, term, limit))

    normalized = [normalize_server(s) for s in raw_servers]
    deduped = _dedupe_latest(normalized)
    # Rank: installable first, then by how many query keywords the entry matches
    # (so "slack" beats an unrelated "message" server when merging fallbacks).
    keywords = _keywords(query)
    deduped.sort(key=lambda e: (e["install"] is None, -_relevance(e, keywords)))
    return deduped[:limit]
