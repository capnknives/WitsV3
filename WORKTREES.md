# Local checkouts (Windows)

Three linked git worktrees of the same GitHub repo. Keep roles separate so
personal use, Claude, and Cursor do not stomp each other.

| Folder | Branch (typical) | Role |
|--------|------------------|------|
| **`WitsV3`** | `fix/revive-2026-07` (or `main`) | **Your runtime** — run the Web UI / CLI, daily self-repair, watchdog, live testing |
| **`WitsV3-cursor`** | `cursor/work` (or `cursor/*`) | **Cursor agent** — AI edits and PRs land here first |
| **`WitsV3-claude`** | `claude/work` (or `claude/*`) | **Claude agent** — Claude Code / Claude sessions only |

Paths (this machine):

```
C:\Users\capta\source\repos\capnknives\WitsV3
C:\Users\capta\source\repos\capnknives\WitsV3-cursor
C:\Users\capta\source\repos\capnknives\WitsV3-claude
```

## Rules of the road

1. **Run WITS for yourself from `WitsV3` only** (`start_web_ui.bat` / `run_web.py`). Point Scheduled Tasks / watchdog at this folder, not an agent worktree.
2. **Open Cursor’s project folder on `WitsV3-cursor`** when asking the Cursor agent to edit code.
3. **Open Claude on `WitsV3-claude`** the same way.
4. Agents push feature branches → merge into `fix/revive-2026-07` → you pull that into `WitsV3` for personal use → promote to `main` when ready.
5. Do not commit `.env`, `data/`, or memory files. Agent worktrees may share the personal `.venv` via a directory junction; that is intentional.

> ⚠️ **Edits made in `WitsV3-cursor` are NOT live in your runtime.** The Web UI /
> CLI you actually run come from **`WitsV3`**. After finishing changes here
> (especially `web/` static assets, which have no build step and are served
> as-is), sync them into `WitsV3` or they won't show up:
>
> ```powershell
> # from WitsV3-cursor: commit, then fast-forward the runtime worktree
> git -C C:\Users\capta\source\repos\capnknives\WitsV3-cursor add -A
> git -C C:\Users\capta\source\repos\capnknives\WitsV3-cursor commit -m "..."
> git -C C:\Users\capta\source\repos\capnknives\WitsV3 merge --ff-only cursor/work
> ```
>
> Then hard-refresh the browser (static assets are cached).

## Recreate / list

From any checkout:

```powershell
cd C:\Users\capta\source\repos\capnknives\WitsV3
git worktree list

# Add Cursor worktree (if missing)
git branch cursor/work origin/fix/revive-2026-07   # or current tip
git worktree add ..\WitsV3-cursor cursor/work

# Add Claude worktree (if missing)
git worktree add ..\WitsV3-claude claude/work
```

Bootstrap a new agent worktree (once):

```powershell
$src = "...\WitsV3"
$dst = "...\WitsV3-cursor"   # or WitsV3-claude
Copy-Item "$src\.env" "$dst\.env"
cmd /c mklink /J "$dst\.venv" "$src\.venv"
& "$dst\.venv\Scripts\python.exe" "$dst\scripts\setup_local_data.py"
```

## Watchdog / wake timer

Register Scheduled Tasks against **`WitsV3`** (see
[`docs/watchdog-and-wake-timer.md`](docs/watchdog-and-wake-timer.md)). If an
older task still points at `WitsV3-claude`, unregister and re-register it.
