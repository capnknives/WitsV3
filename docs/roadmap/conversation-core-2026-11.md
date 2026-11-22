# Conversation core redesign (November 2026)

**Created:** November 22, 2026
**Status:** direction agreed by the owner; packages listed in
[`unified-plan-2026-11.md`](unified-plan-2026-11.md) Stage 2, briefs not yet written.
**Reviewed against:** `main` @ `86c04d4`

This file records why Stage 2 changed and what the new conversation core is. It
draws on three inputs from November 22: a line-by-line map of the current turn
path, our own web research, and an outside research report. Where the inputs
disagree, the code map wins, because the other two could not see the code.

## The problem

A plain sentence that mentioned "software" waited about 45 seconds and was then
handed to the coding agent. The causes, all confirmed in code:

- Messages the rule list does not catch go to a separate intent call
  (`agents/wcca_intent_mixin.py`) before any answer is produced.
- Specialist choice then runs on whole-word keyword lists in
  `_select_specialized_agent` (`agents/wits_control_center_agent.py`). A second,
  different set of lists lives in `agents/routing_classifier.py`.
- Every model call on the default path goes to `/api/generate` with the
  conversation flattened into one prompt string. Greeting and fallback replies
  get no history at all.
- Tools are described to the model by name and one line. Parameter schemas are
  dropped.
- Nothing streams token by token. Each answer arrives as one block.
- Memory is searched only on the orchestrator path. Intent output is stored as a
  memory record.
- The coding agent always asks for `model_routing.code_model`, whatever the
  router setting.

## Target design

One loop. The model reads the conversation and decides for itself whether to
reply, call a tool, or load a skill.

1. **Front door.** Plain code handles only exact commands (slash commands, owner
   shortcuts). Everything else goes straight to the loop. No intent call, no
   keyword routing.
2. **Stable prefix.** System prompt, then skill catalog, then a short fixed tool
   list, then earlier turns, then the new message. Nothing before the new
   message changes between turns, so the server can reuse its cache. Time,
   retrieved memory and other changing text go at the end.
3. **One resident model.** One chat model stays loaded, called through
   `/api/chat` with a real message list, native tool schemas, a fixed `num_ctx`
   and thinking off by default. Embeddings run on the CPU so they never evict
   it.
4. **Checks in code.** Every tool call is validated against its schema; errors
   go back to the model once; repeated identical calls and step counts are
   capped. Writes pass a gate that can raise a decision in the existing
   approval queue.
5. **Escalation by rule, not by guess.** Same model with thinking on first.
   Then the heavy model, on an explicit command or after two failed tool
   rounds, given a short written brief and file paths rather than the whole
   conversation. The heavy step is a provider slot: the local heavy model, or a
   free cloud model for turns that carry nothing private.
6. **Specialists become skills plus tools.** The instructions of the coding,
   self-repair and writing agents move into skill files loaded on demand. The
   verified-edit pipeline (`core/safe_code_editor.py`) stays as code and is
   offered to the loop as a tool. It is not replaced by a prompt.
7. **Memory writes are deliberate.** Only stated facts, preferences and
   decisions, with their source. Never routing output, raw reasoning, tool
   traces, or text that came from an untrusted page in the same turn without
   the owner's say-so.
8. **What the owner sees.** The reply streams as it is written. Tool work shows
   as one collapsed activity line. There is a stop button.

Not adopted: a team of role agents (retrieval, scoring, summary and so on), an
embedding or classifier router in front of the loop, and parallel subagents on
one GPU. A bounded subagent is allowed later for read-heavy exploration,
independent review, or reading untrusted content with read-only tools.

## What already exists

- `agents/native_tool_orchestrator.py` is a native tool-calling loop on
  `/api/chat`. It is opt-in (`orchestrator.tool_calling_mode: ollama_native`)
  and has no tests. Stage 2 finishes and promotes it instead of writing a new
  loop.
- Both entry points reach the agent through one call:
  `system.control_center.run(...)` at `web/server.py` (chat endpoint) and in
  `run.py`. A new front door can sit behind a setting at that seam and be
  compared with the old path.
- `llm_interface.generate_chat` already sends a message list and `tools`.
- The browser already renders `thinking`, `action`, `observation`, `result` and
  `error` events.

## Swapping models without losing the place

The conversation text is never lost on a swap. What is lost is the server's
processed copy of it, which has to be rebuilt by re-reading. Three answers, in
order of use:

1. Brief the heavy model with a summary and file paths (default).
2. Send heavy work to a cloud provider when the turn is not private.
3. Save and restore the processed copy on disk with `llama-server`, one file
   per model per conversation, deleted when the conversation is closed. To be
   tested on the PC before it is relied on.

## Evidence and its limits

Confirmed by our own reading of primary sources:

- Ollama's `/api/chat` supports `tools`, `think` and `keep_alive`; both
  installed model families list tool support.
- Cache reuse depends on an unchanged prompt prefix.
- A 180-configuration study found multi-agent setups lost 39 to 70% on
  sequential tasks and helped only on work that splits cleanly.
- Deferring tool definitions behind a search step cut context and raised
  accuracy in the vendor's own evaluation.
- The current MCP revision removed stream resumability and moved tasks to an
  extension, so approvals must be polled.

Taken from the outside report and **not yet checked by us**. Verify before any
brief depends on them:

- Thinking off cuts simple-turn wait by about four times.
- A cache-reuse fault affecting Gemma 4 in llama.cpp.
- Tool calls from Qwen 3.5 sometimes arriving as plain text.
- A harness study showing completion checks in code lifting small-model task
  success.
- The event count and version of the AG-UI protocol. Sources disagree, so
  Stage 2 borrows its event names and does not take the dependency.

Known error in the outside report's premises: it assumed 12 to 24 GB of GPU
memory. This PC has 8 GB, and Stage 0 found the heavy model fastest on
`llama-server`. The resident model is therefore chosen by measurement in
package C2, not taken from the report.

No published tool-calling scores exist for the exact installed models. Stage 0
item 0.6 remains the source of truth.
