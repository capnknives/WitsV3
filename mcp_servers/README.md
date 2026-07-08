# MCP server vendoring

Git submodules under this directory (`Ollama-mcp`, `servers`) are optional local
copies of upstream MCP server repos. Fresh clones can populate them on demand:

```bash
python scripts/clone_mcp_servers.py
```

That script reads `data/mcp_tools.json`, clones any GitHub `type: github` entries
that are missing, installs dependencies, and updates working directories.

**Supabase MCP** was removed from the default config (July 2026): there is no
active Supabase project, and the server added hundreds of vendored files with
no runtime benefit. To use it again, add an entry back to `data/mcp_tools.json`
and run the clone script.
