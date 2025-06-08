# WitsV3 Data Directory

This directory contains runtime data files for WitsV3. Personal and runtime data files are excluded from git tracking.

## Structure

### Tracked Files (Safe to commit)

- `mcp_config.json` - MCP server configuration
- `mcp_tools.json` - MCP tool definitions
- `*.template` - Template files showing structure

### Ignored Files (Personal/Runtime data)

- `wits_memory.json` - Personal conversation memory (auto-generated)
- `neural_web.json` - Personal neural connections (auto-generated)
- `*_backup.json` - Backup files
- `faiss_index.*` - Vector embeddings index files

## Important Notes

⚠️ **Personal Data Protection**: Memory and neural web files contain personal conversation data and are automatically excluded from git tracking.

🔄 **Auto-Generation**: Missing data files will be automatically created when WitsV3 starts.

🗑️ **Cleanup**: To reset memory, delete `wits_memory.json` and `neural_web.json` - they will be recreated.

## File Sizes

- Memory files can grow to 10MB+ over time
- Neural web files typically 1-5MB
- These files should never be committed to prevent repository bloat
