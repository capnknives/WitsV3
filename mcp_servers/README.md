# MCP servers (on-demand)

WitsV3 does **not** vendor MCP server repos in git. Clone them locally when needed:

```bash
python scripts/clone_mcp_servers.py
```

That script reads `data/mcp_tools.json`, clones any GitHub `type: github` entries into
this directory, installs dependencies, and updates working directories in the config.

**Included by default in config** (after clone):

- `Ollama-mcp` — Ollama MCP bridge
- `servers` (modelcontextprotocol/servers) — sequential thinking + filesystem subdirs

**Removed from default config (July 2026):** `supabase-mcp` — no active Supabase project;
hundreds of vendored files with no runtime benefit. Re-add to `data/mcp_tools.json` and
run the clone script if you need it again.

Registry-based servers (Gmail, codebase-memory, etc.) use `npx`/`uvx` and do not need
a local clone.
