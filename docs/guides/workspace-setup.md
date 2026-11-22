---
title: "Workspace Setup Guide"
created: "2026-11-21"
last_updated: "2026-11-21"
status: "active"
---

# Workspace setup guide

## 1. What this sets up

Outside AI agents connect to WITS at `/mcp`, each with its own token.
They claim tasks, leave handoffs and ask you for decisions.
You answer on the Inbox page from your phone.
You can cut off any one agent without affecting the others.

## 2. Before you start

1. Open PowerShell and go to the project folder.
```powershell
cd <path-to-WitsV3>
```
2. Get the latest code.
```powershell
git pull
```
3. Install the packages into the project's virtual environment. Two of them are new: `aiosqlite` and `mcp`.
```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```
4. Always start WITS from the project folder. Paths like `var/data/workspace.db` are relative, so starting from another folder can point WITS at the wrong files.

## 3. Tailscale

Tailscale links your PC, phone and laptop into one private network. Tailscale has a free personal plan; check https://tailscale.com/pricing for its current limits.

1. Install Tailscale on the PC from https://tailscale.com/download.
2. Install Tailscale on your laptop from the same page.
3. Install the Tailscale app on your phone from the app store.
4. Sign in to the same account on all three.
5. On the PC, run these two commands. The first shows the short machine name and the 100.x address. The second shows the full tailnet name.
```powershell
& 'C:\Program Files\Tailscale\tailscale.exe' status
(& 'C:\Program Files\Tailscale\tailscale.exe' status --json | ConvertFrom-Json).Self.DNSName
```
6. Tailscale shows up as a VPN on the phone and the PC. It only carries traffic between your own devices, unless you turn on an exit node. Leave exit nodes off, and switch Tailscale off on a device whenever you do not want a VPN active there. The Inbox is only reachable from outside the house while it is on.
7. Write down all three: the full tailnet name (it looks like `<your-pc>.<tailnet>.ts.net`, without the dot at the end), the short machine name and the 100.x address.

## 4. Turn the workspace on

1. Open `.env` in the project folder. Set a long random owner token.
```
WITSV3_WEB_TOKEN=<owner token>
```
2. To make a random token, run this command and copy the output.
```powershell
.venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(32))"
```
3. Keep `web_ui.require_auth: true` in `config.yaml`. Do not change it.
4. Create `config.local.yaml` in the project folder. Git ignores this file, so your settings stay off GitHub.
```powershell
notepad config.local.yaml
```
5. Paste this block into `config.local.yaml` and save it. Use your own names in place of the placeholders.
```yaml
workspace:
  enabled: true
  allowed_hosts:
    - localhost
    - 127.0.0.1
    - <your-pc>.<tailnet>.ts.net
    - <your-pc>
    - <100.x address>
```
6. Understand `allowed_hosts`. Every name an agent puts in its URL must be on this list. An entry without a port allows any port.
7. Stop WITS if it is running. Press Ctrl+C in its window.
8. Start WITS from the project folder.
```powershell
.venv\Scripts\python.exe run_web.py
```

## 5. Check it is up

Run these three checks from the PC. Use the full tailnet name from step 3.

1. Check the Inbox page. Expected result: `200`.
```powershell
(Invoke-WebRequest -Uri http://<your-pc>.<tailnet>.ts.net:8000/inbox -UseBasicParsing).StatusCode
```
2. Call the MCP endpoint with no token. Expected result: `401`.
```powershell
try { Invoke-WebRequest -Method Post -Uri http://<your-pc>.<tailnet>.ts.net:8000/mcp/ -UseBasicParsing } catch { [int]$_.Exception.Response.StatusCode }
```
3. Call an owner route with no token. Expected result: `401`.
```powershell
try { Invoke-WebRequest -Uri http://localhost:8000/api/workspace/clients -UseBasicParsing } catch { [int]$_.Exception.Response.StatusCode }
```

If you get a different number, use this table.

| Code | What it means | What to do |
|------|---------------|------------|
| 401 | Token missing, wrong, revoked or expired | Check the token in the agent's config |
| 403 | The request had an Origin header, so it came from a browser | Browser tools cannot use this endpoint. Use an agent |
| 405 | The request was not a POST | Send a POST to `/mcp/` |
| 413 | The request body is over 256 KB | Send a smaller request |
| 421 | The host name is not in `allowed_hosts` | Add the name to `config.local.yaml` and restart WITS |
| 429 | Too many requests, or 10 failed tokens in a minute from one address | Wait one minute |
| 503 | WITS could not check the token | Check the WITS window for errors |

## 6. Create a client

Make one client for each agent. There is no page for this yet, so you use PowerShell.

1. Read the owner token into a variable. Nothing shows as you type, and the token stays out of shell history.
```powershell
$secure = Read-Host "Owner token" -AsSecureString
$bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
$ownerToken = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
[Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
```
2. Create the client. Change the name and the scopes for each agent.
```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/workspace/clients -Headers @{ Authorization = "Bearer $ownerToken" } -ContentType 'application/json' -Body (@{ name = 'claude-code-laptop'; scopes = @('read','write','decide:request'); projects = @('aquainspection') } | ConvertTo-Json)
```
3. Copy the `token` value from the answer now. WITS shows it only once.
4. Know the scope names. Use only what the agent needs.
   - `read` lets the agent read project state.
   - `write` lets the agent claim tasks, leave handoffs and link artifacts.
   - `decide:request` lets the agent ask you for decisions.
5. Know the other fields. `projects` is required. Use `@('*')` to allow every project. Scopes do not imply each other, so list each one you want.
6. List your clients. The list never shows tokens.
```powershell
Invoke-RestMethod -Uri http://localhost:8000/api/workspace/clients -Headers @{ Authorization = "Bearer $ownerToken" }
```
7. Revoke a client. Copy its `id` from the list first. Revoking stops its access and cancels its open decisions. A revoked name cannot be used again, so pick a new name for a new agent.
```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/workspace/clients/<client id>/revoke -Headers @{ Authorization = "Bearer $ownerToken" }
```

## 7. Connect an agent

1. Use this URL for the agent. Keep the trailing slash.
```
http://<your-pc>.<tailnet>.ts.net:8000/mcp/
```
2. Send the token in a header named `Authorization` with the value `Bearer <client token>`. Never put the token in the URL.
3. Here is an example for Claude Code.
```powershell
claude mcp add --transport http wits-workspace http://<your-pc>.<tailnet>.ts.net:8000/mcp/ --header "Authorization: Bearer <client token>"
```
4. If the flags differ in your version, run `claude mcp add --help` and follow that instead.
5. The token now sits in that agent's config file in plain text. Keep that file private.
6. The agent gets five tools.
   - `get_project_state` reads a project's tasks, handoffs, decisions and artifacts.
   - `claim_task` claims an existing task or creates and claims a new one.
   - `append_handoff` leaves a note, handoff, blocked or done message.
   - `request_decision` asks you a question with one to three options.
   - `link_artifact` attaches a link, file or commit to a project.

## 8. The Inbox on your phone

1. Turn Tailscale on in the phone app.
2. Open `http://<your-pc>.<tailnet>.ts.net:8000/?owner=1` in your phone browser. A token box appears. Enter the owner token once. Use the tailnet address, not the home Wi-Fi address, because the browser remembers the token for each address. A sign-in link with the token in it does not work here: WITS only accepts those on the PC itself.
   To avoid typing the token, double-click `show_owner_token_qr.bat` in the project folder on the PC, scan the QR code with the phone camera, copy the text and paste it into the box. Press Enter on the PC afterwards to clear the window.
3. Open `http://<your-pc>.<tailnet>.ts.net:8000/inbox`.
4. Add the Inbox to your home screen. On iPhone, use Share, then Add to Home Screen. On Android, use the browser menu, then Add to home screen.
5. Read each card. It shows who asked, the project, the title and text. If the text looked like an injection attempt, the card shows a warning.
6. Tap one button per choice. Approving records your choice. It does not run anything. The agent reads your answer and acts on its own.

## 9. Phone push (optional)

1. Install the ntfy app on your phone.
2. Make a long random topic name. Anyone who knows the name can read the topic, so treat it like a password.
```powershell
.venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(24))"
```
3. Add the topic to `.env`.
```
WITSV3_NTFY_TOPIC=<topic>
```
4. Set the push provider in `config.local.yaml`. Keep the rest of the `workspace:` block.
```yaml
workspace:
  enabled: true
  allowed_hosts:
    - localhost
    - 127.0.0.1
    - <your-pc>.<tailnet>.ts.net
    - <your-pc>
    - <100.x address>
  push:
    provider: ntfy
```
5. Restart WITS.
6. In the ntfy app, subscribe to your topic.
7. Know what leaves the PC. WITS sends the agent's name and the words "decision waiting" to the ntfy server. It sends no title, no text and no project name.

## 10. Limits worth knowing

- A claim lease lasts 5 to 240 minutes. The default is 30.
- A decision expires after 24 hours by default.
- Each client can have 5 pending decisions at once and create 20 decisions in 24 hours.
- Each client has a request rate limit. The default is 60 requests a minute.
- Title is 200 characters at most. Body is 8,000. A handoff is 16,000.

## 11. If something goes wrong

| Problem | Likely cause | Fix |
|---------|--------------|-----|
| WITS will not start after you turn the workspace on | `WITSV3_WEB_TOKEN` is missing, or `web_ui.require_auth` is false | Set the token in `.env` and keep `require_auth: true` |
| Agent gets 421 | Its host name is not in `allowed_hosts` | Add the name to `config.local.yaml` and restart WITS |
| Agent gets 401 | Token is wrong, revoked or expired | Check the token, or create a new client |
| Inbox says you are not signed in | You have not signed in at this address | Sign in on the main page at the same address first |
| No push to your phone | Provider, topic or app subscription is wrong | Check `provider: ntfy`, `WITSV3_NTFY_TOPIC`, and the topic in the app |

## 12. Backups and Windows notes

- Your data is in `var/data/workspace.db`.
- Do not copy that file alone while WITS runs. Stop WITS first, then copy `workspace.db` with its `-wal` and `-shm` files if they exist.
- Keep `var/data` on a local disk, not in a synced folder such as OneDrive.
- When you edit project files, WITS restarts on its own. That resets rate limits and drops agents for a few seconds.
