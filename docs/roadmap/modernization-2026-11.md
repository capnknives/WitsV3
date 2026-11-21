---
title: "WitsV3 Modernization Roadmap"
created: "2026-11-17"
last_updated: "2026-11-21"
status: "active"
---

# WitsV3 Modernization Roadmap

**Created:** November 17, 2026
**Status:** Planned — no phase started
**Reviewed against:** `main` @ `e321f7b` (July 9, 2026)
**Supersedes for forward work:** [`suggested-features-2026-07.md`](suggested-features-2026-07.md) (kept as the record of the July revival)

> **Order of work:** [`unified-plan-2026-11.md`](unified-plan-2026-11.md) sets
> the stages and gates (a workspace slice ships before the rest of Phase 1).
> Item numbers and checkboxes below remain the tracker.

This is the **canonical forward roadmap**. It takes WitsV3 from a local
chat-and-tools assistant to a personal assistant that can run daily affairs,
delegate work to other agent harnesses, and be used by outside agents such as
Claude.

## Product stance change

The July roadmap described WitsV3 as a "personal LAN orchestration stack" and
parked voice, smart home and cloud routing. This roadmap changes that:

- WitsV3 becomes a **personal assistant with agent interfaces on both sides**.
- Local-first stays the default. Cloud models and outside agents are opt-in,
  per task, behind approval.
- Voice and Home Assistant move from "parked" to Phase 5.
- Still out of scope: fine-tuning, microVM sandboxes, Windows ODR.

## Three gaps

| Gap | Today | Target |
|-----|-------|--------|
| **Daily affairs** | No calendar, email, tasks, reminders or notifications. WITS only acts when typed to. | Personal data layer, connectors, a scheduler that starts work on its own, phone notifications |
| **Harness out** | One LLM backend (`OllamaInterface`); Claude reachable only via the one-question `ask_claude` escalation | WITS hands whole jobs to Claude Code, Codex, Gemini CLI and others as supervised workers |
| **Harness in** | No agent-facing interface; one shared `WITSV3_WEB_TOKEN` gates every `/api/*` route | MCP server first, then A2A, with per-client identity and scopes |

## Problems found in review

| Problem | Evidence |
|---------|----------|
| Ollama is the only real LLM backend | `core/llm_interface.py` has one concrete class |
| MCP client is hand-rolled, stdio only | `StdioMCPClient` in `core/mcp_adapter.py`; no HTTP transport |
| Nothing faces other agents | No MCP server, A2A endpoint or OpenAI-compatible API |
| One shared owner secret | No scopes, no per-client revocation |
| WITS only reacts | Scheduler runs self-repair and a health check; other background tasks are no-ops; `/api/tasks` is read-only |
| Storage is JSON files | `var/sessions/*.json`, `wits_memory.json` — unsafe with concurrent agent writers |
| Default tool calling is prompt-parsed JSON | `orchestrator.tool_calling_mode: json_react`; native mode is a pilot |
| Research code beside the hot path | Neural web, knowledge graph, cross-domain learning, meta-reasoning (~6,000 lines) |
| Unused dependencies | `langchain*`, `pandas`, `scikit-learn`, `typer`, `websockets`, `requests` have no imports |
| Entry-point sprawl | Four `run*.py`, `setup.py` + `pyproject.toml`, three Cursor rules copies |

## Target architecture

```
   You (web UI, phone, voice)            Outside agents (Claude, Claude Code, Cursor, ...)
            |                                   |                        |
        Web API                          MCP server (new)        A2A endpoint (new)
            |                                   |                        |
  +--------------------------------- WITS core ---------------------------------+
  |  Gateway (new): identity per client, scopes, approval queue, audit log     |
  |  Orchestrator        Scheduler + triggers (new)      Personal data + memory |
  +------------------------------------------------------------------------------+
            |                                   |                        |
      Model router                         MCP client             Harness runner (new)
   (Ollama, cloud models)           (calendar, mail, files)   (Claude Code, Codex, Gemini CLI)
```

Design rules:

1. **One tool registry, three faces.** A tool in `core/tool_registry.py` is
   callable by the orchestrator, listed by the MCP server, and available to
   harness workers. No tool is written twice.
2. **The gateway decides, not the caller.** The guest allowlist becomes the
   general permission model for every client.
3. **Risky actions wait for the owner.** Reuse the approval pattern from
   `core/escalation.py`.
4. **Standards over custom code.** Official MCP and A2A Python SDKs.

## Phases

Each phase ends at a gate the owner can test by hand. Sizes are rough,
part-time estimates.

### Phase 1 — Foundations (2–3 weeks)

- [ ] 1.1 Remove unused dependencies from `requirements.txt` / `pyproject.toml`
- [ ] 1.2 Move neural web, knowledge graph, cross-domain learning and meta-reasoning into a `research/` package outside the core
- [ ] 1.3 One `wits` command (`wits web`, `wits cli`, `wits mcp`) replacing the `run*.py` files
- [ ] 1.4 SQLite for sessions, memory records and the approval queue (FAISS stays for vectors; JSON becomes export only)
- [ ] 1.5 Anthropic and OpenAI-compatible providers beside `OllamaInterface`
- [ ] 1.6 Native tool calling as default; `json_react` kept as fallback
- [ ] 1.7 Replace `StdioMCPClient` with the official MCP SDK client (adds HTTP transport)
- [ ] 1.8 Gateway: client identities, scopes, revocable tokens, general approval queue
- [ ] 1.9 Data tiers: mark data classes local-only so the router never sends them to a cloud model

**Gate:** full test suite green with native tool calls; chat behaves as before.

### Phase 2 — Harness in (1–2 weeks)

- [ ] 2.1 MCP server over stdio and HTTP (`/mcp`), mounted in the FastAPI app, targeting MCP spec 2026-07-28
- [ ] 2.2 First tool set: `memory_search`, `memory_add`, `document_search`, `ask_wits`, `list_approvals`
- [ ] 2.3 Resources (documents, core memory) and prompts (`config/playbooks/*.yaml`)
- [ ] 2.4 OpenAI-compatible `/v1/chat/completions`
- [ ] 2.5 "Connected agents" page in `/settings`: create, scope, revoke, per-client audit
- [ ] 2.6 Inbound agent text passes `core/injection_guard.py`; per-client rate limits

**Gate:** Claude Code lists and calls WITS tools; revoking its token cuts access.

### Phase 3 — Daily affairs (3–4 weeks)

- [ ] 3.1 Tasks and reminders: tables plus `add_task`, `list_tasks`, `complete_task`, `snooze`
- [ ] 3.2 Scheduler with time, event and follow-up triggers; editable tasks page
- [ ] 3.3 Phone notifications (Web Push in the PWA, or ntfy)
- [ ] 3.4 Calendar connector via MCP (read free; create/move with approval)
- [ ] 3.5 Mail connector via MCP (read and draft free; send with approval)
- [ ] 3.6 Daily briefing job
- [ ] 3.7 Quick capture: PWA share target, inbox table, triage job
- [ ] 3.8 Watched folders for document RAG

**Gate:** a week of briefings reaches the phone without touching the PC; a reminder fires on time.

### Phase 4 — Harness out (2–3 weeks)

- [ ] 4.1 Agent Client Protocol (ACP) client
- [ ] 4.2 Harness runner: job records, one git worktree per job, time and cost limits
- [ ] 4.3 Relay agent permission requests to the approval queue; stream progress over SSE
- [ ] 4.4 Results pass the verified-edit rule (`core/safe_code_editor.py`) before any merge is proposed
- [ ] 4.5 Claude Agent SDK worker that receives WITS tools
- [ ] 4.6 Delegation policy: named agent, multi-file work outside this repo, or repeated local failure (`core/model_reliability.py`)
- [ ] 4.7 A2A endpoint and agent card on top of the job records

**Gate:** a delegated coding job ends in a tested diff approved from the phone.

### Phase 5 — Reach (ongoing)

- [ ] 5.1 Voice: local Whisper speech-to-text, Piper text-to-speech
- [ ] 5.2 Bills and renewals, people, family lists
- [ ] 5.3 Home Assistant via MCP
- [ ] 5.4 OAuth so hosted Claude can connect over a public HTTPS tunnel
- [ ] 5.5 Front-end restructure (`web/static/app.js` is ~1,200 lines of plain JavaScript)

## Approval policy

| Action | Default |
|--------|---------|
| Read memory, documents, tasks, agenda | Allowed within scopes; logged |
| Add tasks, notes, memory | Allowed; logged; undoable |
| Write files inside a job worktree | Allowed |
| Write files elsewhere; change WITS code | Approval, then verified-edit pipeline |
| Send mail or messages; create events that invite others | Approval every time |
| Run shell or Python | Docker sandbox only |
| Spend money | Never automated |

Also: keep remote access on Tailscale (never forward port 8000), move connector
tokens to the OS keyring, and extend offline mode into a "pause all agents" switch.

## Open decisions (owner)

- [ ] Which Google accounts WITS manages first
- [ ] Whether any personal data may go to a cloud model (assumed: local by default, cloud per task with approval)
- [ ] Hosted Claude access vs. local network only (assumed: local network first)
- [ ] Keep neural web in this repo or split it out (assumed: `research/` package)

## References

- MCP specification 2026-07-28: <https://modelcontextprotocol.io/specification/2026-07-28>
- A2A protocol: <https://a2a-protocol.org/latest/>
- Agent Client Protocol: <https://agentclientprotocol.com/overview/agents>
- Claude Agent SDK (Python): <https://github.com/anthropics/claude-agent-sdk-python>

Confirm protocol versions when Phase 2 starts; they move quickly.
