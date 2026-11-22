---
title: "Stage 2 Briefs — Conversation Core"
created: "2026-11-22"
last_updated: "2026-11-22"
status: "active"
---

# Stage 2 briefs: conversation core

Rules and the review ladder are in [README.md](README.md). The design is in
[`../conversation-core-2026-11.md`](../conversation-core-2026-11.md). Briefs are
added here one package ahead of the build. Reviewed against `main` @ `7be3d54`.

## C1 contract (shared by C1a and C1b)

C1 changes no behaviour. It adds a record of each turn and a fixed set of test
messages, so every later package can be compared with the old path.

**Turn record.** One JSON object per turn, appended as one line to
`<logs_dir>/turn_metrics/YYYY-MM-DD.jsonl` (`core.runtime_paths.logs_dir()`).
Fields, all required, `null` where unknown:

| Field | Type | Meaning |
|-------|------|---------|
| `ts` | int | Unix seconds, UTC, at turn start |
| `turn_id` | str | `trn_` plus `secrets.token_hex(8)` |
| `session_id` | str or null | As passed to `control_center.run` |
| `entry` | str | `web`, `cli` or `golden` |
| `path` | str | `legacy` for now. C3 adds `native` |
| `input_chars` | int | Length of the message. The text itself is never stored |
| `first_event_ms` | float or null | Turn start to the first `StreamData` of any type |
| `first_result_ms` | float or null | Turn start to the first `result` or `error` event |
| `first_token_ms` | float or null | Turn start to the first model token. Set only by a streaming model call; otherwise null |
| `wall_ms` | float | Turn start to the end of the stream |
| `llm_calls` | list | One entry per model call, in order (below) |
| `tool_calls` | list | `{name, ok, ms}` per tool execution, in order |
| `outcome` | str | `ok`, `error` or `cancelled` |

Each `llm_calls` entry: `{endpoint, model, wall_ms, load_ms, prompt_ms,
prompt_tokens, eval_ms, eval_tokens}`. `endpoint` is `generate`, `chat` or
`generate_stream`. The last five come from Ollama's `load_duration`,
`prompt_eval_duration`, `prompt_eval_count`, `eval_duration` and `eval_count`
(nanoseconds converted to milliseconds), `null` when the response lacks them.

**Privacy.** No message text, reply text, prompt, tool argument or tool result
is written. Names, counts and times only.

**Seam.** `core/turn_metrics.py` exposes exactly:

```python
@dataclass
class TurnMetrics: ...                          # fields above; to_dict()
def start_turn(entry: str, session_id: str | None, input_chars: int, path: str = "legacy") -> TurnMetrics
def current_turn() -> TurnMetrics | None        # ContextVar, default None
def record_llm_call(endpoint: str, model: str, wall_ms: float, response: dict | None) -> None
def record_first_token() -> None                # keeps the first call only
def record_tool(name: str, ok: bool, ms: float) -> None
def note_event(event_type: str) -> None         # sets first_event_ms / first_result_ms once
async def finish_turn(turn: TurnMetrics, outcome: str) -> None   # appends the line, clears the ContextVar
```

Every `record_*` and `note_event` does nothing when `current_turn()` is `None`.
Nothing in this module may raise into the caller: failures are logged at debug
level and dropped. The file append runs through `asyncio.to_thread`.

**Config.** `turn_metrics.enabled` (default `true`) and
`turn_metrics.retention_days` (default `30`; older files are deleted when a new
day's file is first opened). With `enabled: false`, `start_turn` still returns
an object and `finish_turn` writes nothing.

**Golden case.** One YAML mapping per case in `tests/golden/turns.yaml`:

```yaml
- id: owner-software-statement          # unique, kebab-case
  message: "Its richard i created the software you are wearing"
  tags: [must-not-route, chat]
  expect:
    tools: none                         # none | any | optional | [names] (every name listed must be called)
    forbid_tools: []                    # names that must not be called
    forbid_routes: [coding]             # coding | self_repair | book | orchestrator
    reply: required                     # required | optional
```

Without a `tools` key the default is `any` when `tools_any_of` is given, `optional`
(any tool or none) when only `forbid_tools` is given, and `none` otherwise; an explicit
`tools:` value always wins.

A case passes when every `expect` key holds. The check is code over the
`StreamData` events of the turn. No model judges a reply.

## WP C1a: Turn metrics

```
Tier: Sonnet, medium
Branch: claude/wp-c1a-turn-metrics, from origin/main
Read: this file, "C1 contract"
      core/llm_interface.py lines 28-40, 184-320 (LLMResponse; generate_text,
        generate_chat, stream_text of the Ollama class)
      core/tool_registry.py lines 120-180 (execute_tool)
      web/server.py lines 385-450 (the chat stream loop only)
      run.py lines 505-525 (the CLI loop only)
      core/runtime_paths.py line 78 (logs_dir)
      core/config.py lines 334-360 (WorkspaceSettings, as the pattern for a settings block)
      core/guest_audit.py lines 55-85 (the existing JSONL append, as the pattern)
Do not read: docs/implementation/, docs/archive/, docs/roadmap/revival-2026-07.md,
      the rest of web/server.py, agents/
Contract: this file, "C1 contract"
Build:
  - core/turn_metrics.py as specified. New file.
  - core/config.py: TurnMetricsSettings and a `turn_metrics` field on WitsV3Config.
    config.yaml: the two keys with their defaults.
  - core/llm_interface.py: in the three Ollama methods, time the call and pass the
    parsed final response (or the final stream chunk) to record_llm_call;
    stream_text calls record_first_token on its first non-empty chunk. Return
    values and yielded text stay byte-for-byte the same.
  - core/tool_registry.py execute_tool: time the call, record_tool(name, ok, ms).
    `ok` is false when it raises; the exception is re-raised unchanged.
  - web/server.py and run.py: start_turn before the `async for` over
    control_center.run, note_event(stream_data.type) per event, finish_turn in a
    `finally` (outcome `cancelled` on asyncio.CancelledError or GeneratorExit,
    `error` if any error event was seen, else `ok`). Six lines or fewer per file.
Acceptance (tests/core/test_turn_metrics.py, no Ollama, tmp_path for the log dir):
  - a turn with two fake model calls and one tool writes exactly one line with
    every contract field; nanoseconds become milliseconds
  - no text from the message, prompt or tool arguments appears in the line
  - record_* outside a turn is a no-op; a write failure does not raise
  - enabled: false writes nothing
  - two turns run concurrently with asyncio.gather keep their calls apart
  - retention deletes a file older than retention_days and keeps a newer one
  - the Ollama methods return the same values as before with a mocked transport
    (reuse mock_async_client from tests/conftest.py), with and without the
    duration fields in the response
  - the existing suite still passes: pytest tests/ -q --no-cov
Run: pytest tests/core/test_turn_metrics.py tests/ -q --no-cov;
     ruff check agents core tools web tests; black --check agents core tools web tests
Out of scope: changing any prompt, route, model or reply; token streaming; a UI
     for the numbers; core/smoke_metrics.py and core/metrics.py (leave both alone);
     the llama-server backend
Report: files changed, test tail, open questions. 20 lines at most.
```

## WP C1b: Golden test set

```
Tier: Sonnet, medium
Branch: claude/wp-c1b-golden-set, from origin/main
Read: this file, "C1 contract" (golden case)
      tests/agents/test_coding_route_intent.py (the owner sentence and its neighbours)
      scripts/smoke_scenarios.yaml (first 60 lines, for message style only)
      tests/integration/test_smoke_scenarios_mock.py lines 20-56 (DummyLLM and the wcca fixture)
      core/schemas.py lines 13-45 (StreamData)
      tools/ : file names only, for real tool names (use the name each tool registers)
Do not read: docs/implementation/, docs/archive/, docs/roadmap/revival-2026-07.md,
      scripts/smoke_harness.py, agents/
Contract: this file, "C1 contract" (golden case)
Build:
  - tests/golden/turns.yaml: 30 cases. At least: 8 plain chat or statement
    (tools: none, forbid_routes: [coding, self_repair, book]), including the
    owner sentence and four more that contain a code noun without asking for
    code; 6 that need one named tool (file read, file list, web search, memory
    search, date or time, document search); 4 real coding requests; 3 that name
    a file to repair; 3 writing requests; 3 slash or exact commands; 3 follow-ups
    that only make sense with history (give `history:` as a list of role/content).
  - core/golden.py: load_cases(path) with validation (unknown key, duplicate id
    or bad enum raises ValueError naming the case); evaluate(case, events) ->
    GoldenResult(passed, failures: list[str]). Tool names are read from
    tool_call and action events' metadata; routes from the agent name on events.
    Where an event carries neither, that expectation fails with "not observable".
  - scripts/golden_run.py: runs the set through control_center.run against the
    live system, `--runs N` (default 1), `--only id,id`, `--tag`, `--out`
    (default docs/benchmarks/golden.md). Appends a table: case, pass count,
    median first_result_ms, tools seen. Exit code 1 if any case fails every run.
    Starts each turn with entry "golden" when core.turn_metrics is importable;
    works without it.
Acceptance (tests/core/test_golden.py, no Ollama):
  - turns.yaml loads; ids unique; 30 cases; every tag and enum valid
  - evaluate passes and fails correctly for each expect key, from hand-built events
  - the owner sentence case fails when a coding-route event is present
  - scripts/golden_run.py --help exits 0 without Ollama running
Run: pytest tests/core/test_golden.py -q --no-cov;
     ruff check core/golden.py scripts/golden_run.py tests/core/test_golden.py;
     black --check core/golden.py scripts/golden_run.py tests/core/test_golden.py
Out of scope: running the set against a model (a PC job, on hold); changing any
     agent; judging reply quality
Report: files changed, test tail, open questions (list any expectation that is
     "not observable" on today's events). 20 lines at most.
```

The baseline run of the golden set on the old path is a PC job. It waits for
the hold on PC runs to lift, or for the owner to allow a short run.

## C2 and C3 contract

Reviewed against `main` @ `7be3d54`. Both packages edit `core/llm_interface.py`
after C1a, so C1a merges first. Facts the briefs rest on:

- `generate_chat` (`core/llm_interface.py:216`) sends `model`, `messages`,
  `stream: false`, `options.temperature` and `tools`. It does not send
  `num_ctx`, `keep_alive` or `think`, and it does not stream.
- `get_embedding` (`:372`) posts to `/api/embeddings` with no options, so the
  embedding model competes for the GPU.
- `NativeToolOrchestrator` (`agents/native_tool_orchestrator.py`, 205 lines, no
  tests) puts the goal and the document list into its system message
  (`:102-108`), so the prompt prefix changes every turn. It decodes tool
  arguments without checking them (`:145-149`; a bad decode becomes `{}`), and
  its only bound is `agents.max_iterations` (15).
- Tools disagree on schema shape: the native loop reads
  `get_schema()["parameters"]` (`:49-51`), `BaseTool.get_enhanced_schema`
  (`core/base_tool.py:57`) reads top-level `properties`.
- Intent output is no longer written to memory (guard at
  `agents/wits_control_center_agent.py:229-243`). Old rows carry segment type
  `INTENT_ANALYSIS`.
- `WitsControlCenterAgent.run` (`:111`) has no slash or owner command handling.
  Its only fixed rule is the remember/recall check at `:135-157`. The intent
  call is at `:229`.

**Settings added** (all read through `WitsV3Config`):

| Key | Default | Package |
|-----|---------|---------|
| `ollama_settings.keep_alive` | `"30m"` | C2a |
| `ollama_settings.think` | `false` | C2a |
| `ollama_settings.embedding_on_cpu` | `true` | C2a |
| `conversation.front_door` | `legacy` (`legacy` or `native`) | C3c |
| `conversation.max_steps` | `8` | C3a |
| `conversation.max_identical_calls` | `2` | C3a |
| `conversation.max_schema_retries` | `1` | C3a |

**Message order** (`core/prompt_prefix.py`, C2b), fixed for every turn:

1. One system message: the static system prompt, then the skill catalog. No
   date, no goal, no document list, no retrieved memory.
2. Earlier turns, oldest first, unchanged from how they were first sent.
3. The new user message. Changing text (time, retrieved memory, document list)
   is appended to this message under a `Context for this turn:` heading.

The tool list is sent in a fixed order (sorted by name) so it does not disturb
the cache either.

**Tool-call rules** (`core/tool_call_guard.py`, C3a), applied in this order to
each call the model makes:

1. Unknown tool name: error text back to the model. Counts as a schema failure.
2. Arguments are checked against the tool's schema after normalising both
   shapes above to one JSON Schema object. On failure the error text goes back
   to the model. After `max_schema_retries` failures for the same tool in one
   turn, the turn ends with a plain apology and an `error` event.
3. A call identical (name and canonical JSON arguments) to one already made in
   this turn is not run again once `max_identical_calls` is reached; the model
   is told so.
4. After `max_steps` model rounds the loop stops and answers from what it has.
5. A reply whose text contains a `<tool_call>` block and no structured call is
   treated as one schema failure (rule 2), never shown to the owner.

**Out of C3 on purpose.** Guests stay on the legacy path whatever the setting
says, until C4 gives the native loop the guest tool limits. Text from
`workspace.db` still does not enter any prompt (workspace contracts, section 8,
point 1, is unchanged). The old intent call, keyword lists and JSON-repair loop
are not deleted before the Stage 2 gate.

## WP C2a: Chat serving settings

```
Tier: Sonnet, medium
Branch: claude/wp-c2a-serving, from origin/main (after C1a is merged)
Read: this file, "C2 and C3 contract"
      core/llm_interface.py lines 85-115, 184-265, 372-400
      core/config.py lines 80-100 (OllamaSettings) ; config.yaml lines 1-12, 107-128
Do not read: docs/implementation/, docs/archive/, docs/roadmap/revival-2026-07.md, agents/
Build:
  - the three ollama_settings keys, in core/config.py and config.yaml
  - generate_chat: new optional args num_ctx, think, keep_alive; each falls back
    to its setting; payload gets options.num_ctx and keep_alive always
  - `think` is sent only to models that list the "thinking" capability. Read it
    once per model from POST /api/show (field `capabilities`), cache it in the
    interface, and treat any failure as "no thinking" (the key is then left out)
  - generate_text and stream_text: add keep_alive only. Nothing else changes there
  - get_embedding: when embedding_on_cpu is true, send options {"num_gpu": 0}
    and the same keep_alive
Acceptance (tests/core/test_llm_serving.py, mocked transport, no Ollama):
  - chat payload carries num_ctx and keep_alive from settings and from arguments
  - think is present for a model whose /api/show lists thinking, absent otherwise,
    and /api/show is called once per model
  - embedding payload has num_gpu 0 by default and no options when the setting is off
  - return values of all four methods are unchanged
  - pytest tests/ -q --no-cov still passes
Run: pytest tests/core/test_llm_serving.py tests/ -q --no-cov;
     ruff check agents core tools web tests; black --check agents core tools web tests
Out of scope: streaming chat (C7), choosing the model, the provider interface (2.4)
Report: files changed, test tail, open questions. 20 lines at most.
```

## WP C2b: Stable prompt prefix

```
Tier: Sonnet, medium
Branch: claude/wp-c2b-prefix, from origin/main
Read: this file, "C2 and C3 contract" (message order)
      core/llm_interface.py lines 20-40 (LLMMessage)
      agents/native_tool_orchestrator.py lines 95-115 (what is built today; do not edit)
Do not read: docs/implementation/, docs/archive/, docs/roadmap/revival-2026-07.md
Build:
  - core/prompt_prefix.py, new, pure functions, no I/O:
    build_messages(system_prompt, skill_catalog, history, user_message,
                   turn_context: dict[str, str] | None) -> list[LLMMessage]
    order_tools(tools: list[dict]) -> list[dict]        # sorted by function name
    prefix_fingerprint(messages, tools) -> str          # sha256 of everything before the last message
Acceptance (tests/core/test_prompt_prefix.py):
  - across three turns of one conversation with different turn_context, the
    fingerprint of turn N equals the fingerprint over the same span in turn N+1
  - turn_context never appears before the last message; with None, the last
    message is the user text unchanged
  - order_tools is stable under input shuffling
Run: pytest tests/core/test_prompt_prefix.py -q --no-cov; ruff and black on the two files
Out of scope: wiring it into any agent (C3b), the skill catalog's content (C5)
Report: files changed, test tail, open questions. 20 lines at most.
```

## WP C2c: Remove stored intent rows

```
Tier: Sonnet, medium
Branch: claude/wp-c2c-intent-rows, from origin/main
Read: core/memory_manager.py (class and method names first; then only the
      methods that list and delete segments), core/memory_embedding.py lines 1-30
Do not read: docs/implementation/, docs/archive/, docs/roadmap/revival-2026-07.md, agents/
Build:
  - scripts/purge_intent_memory.py: counts segments of type INTENT_ANALYSIS in the
    configured backend; `--apply` deletes them after writing a timestamped backup
    of the memory file next to it; without --apply it only prints the count.
    Refuses to run while the web server holds the file if that can be detected.
  - a test that a new turn through the control center stores no INTENT_ANALYSIS
    segment (locks in the existing guard at wits_control_center_agent.py:229-243)
Acceptance: tests/core/test_purge_intent_memory.py on a temp memory file: dry run
      changes nothing; --apply removes only INTENT_ANALYSIS rows and leaves a backup
Run: pytest tests/core/test_purge_intent_memory.py -q --no-cov; ruff and black on the new files
Out of scope: any other memory clean-up; the SQLite move (2.8)
Report: files changed, test tail, open questions. 20 lines at most.
```

Running the purge with `--apply` on the PC is an owner decision and a PC job.

**C2m, the resident model, is a measurement, not a build.** On the PC, for each
candidate from package 0.6: the golden set three times with `think` off and
once with it on, reading first-result time and pass counts from the turn
records. The model with the best pass count that keeps a simple turn under two
seconds is named in the unified plan. It waits for the hold on PC runs.

## WP C3a: Tool-call guard

```
Tier: Opus, medium. Review: high
Branch: claude/wp-c3a-tool-guard, from origin/main
Read: this file, "C2 and C3 contract" (tool-call rules)
      core/base_tool.py lines 40-110 ; core/tool_registry.py lines 205-225
      agents/orchestrator_preflight.py lines 40-60, 170-185 (existing repeat counter; do not edit)
      two tools of your choice under tools/ for the two schema shapes
Do not read: docs/implementation/, docs/archive/, docs/roadmap/revival-2026-07.md
Build:
  - core/tool_call_guard.py, new, no I/O:
    normalise_schema(raw: dict) -> dict                  # one JSON Schema object, additionalProperties false
    class TurnGuard(max_steps, max_identical_calls, max_schema_retries)
      .check(name, arguments, schemas) -> Verdict(run | reject | stop, message)
      .note_round() ; .steps_left() ; .plain_text_call(content) -> bool
  - validation with the `jsonschema` package if it is already a dependency;
    otherwise a small validator for type, required, enum and unknown keys.
    Do not add a dependency without saying so in the report
  - conversation settings block (three keys above) in core/config.py and config.yaml
Acceptance (tests/core/test_tool_call_guard.py): one test per rule 1 to 5, plus
      both schema shapes normalise to the same object, plus argument order does
      not defeat the identical-call check
Run: pytest tests/core/test_tool_call_guard.py -q --no-cov; ruff and black on the CI scope
Out of scope: calling tools, any agent file, the write gate (C4 onwards)
Report: files changed, test tail, open questions. 20 lines at most.
```

## WP C3b: Finish the native loop

```
Tier: Opus, medium. Review: independent, medium
Branch: claude/wp-c3b-native-loop, from origin/main (after C2a, C2b and C3a are merged)
Read: this file, "C2 and C3 contract"
      agents/native_tool_orchestrator.py (whole file, 205 lines)
      agents/base_orchestrator_agent.py lines 50-70, 354-370, 495-505
      core/prompt_prefix.py, core/tool_call_guard.py, core/turn_metrics.py (signatures)
Do not read: docs/implementation/, docs/archive/, docs/roadmap/revival-2026-07.md,
      agents/llm_driven_orchestrator.py, agents/wcca_intent_mixin.py
Build:
  - messages come from build_messages; goal, document list and retrieved memory
    move into turn_context; tools go through normalise_schema and order_tools
  - every model round passes through TurnGuard; rejected calls are answered with
    a role "tool" message carrying the verdict text; `stop` ends the loop
  - a failed argument decode is a schema failure, not `{}`
  - events: one `tool_call` event per executed tool with metadata
    {"tool_name": name, "ok": bool}; one `result` at the end. No raw model
    text is emitted as `thinking`
  - if the split is needed to stay under 500 lines, new helpers go in
    agents/native_loop_support.py
Acceptance (tests/agents/test_native_tool_orchestrator.py, scripted fake LLM):
  - plain answer: one model call, no tool, one result
  - one tool then answer; two tools in one round
  - bad arguments once then corrected; bad twice ends with an error event
  - identical call repeated is not executed a third time
  - max_steps reached: loop stops with an answer
  - plain-text <tool_call> is never shown and counts as a schema failure
  - prefix_fingerprint is equal across two consecutive turns
Run: pytest tests/agents/test_native_tool_orchestrator.py tests/ -q --no-cov; ruff and black on the CI scope
Out of scope: the front door (C3c), streaming tokens (C7), skills (C5), guests
Report: files changed, test tail, open questions. 20 lines at most.
```

## WP C3c: The front door setting

```
Tier: Opus, medium. Review: independent, medium
Branch: claude/wp-c3c-front-door, from origin/main (after C3b is merged)
Read: this file, "C2 and C3 contract"
      agents/wits_control_center_agent.py lines 105-250
      run.py lines 205-245, 355-375 ; core/turn_metrics.py (start_turn's path argument)
      tests/golden/turns.yaml (first 40 lines) ; core/golden.py (evaluate)
Do not read: docs/implementation/, docs/archive/, docs/roadmap/revival-2026-07.md
Build:
  - `conversation.front_door`. With `native`, WitsControlCenterAgent.run hands
    every owner turn to the native loop straight after the remember/recall
    check (line 157): no clarification merge, no USER_INPUT store, no creator
    recognition, no intent call, no specialist selection
  - guests always take the legacy path
  - run.py builds the native orchestrator whenever front_door is native,
    whatever tool_calling_mode says; `legacy` behaves exactly as today
  - the turn record's `path` is `native` or `legacy`
Acceptance (tests/agents/test_front_door.py, fake LLM that fails the test if
      the intent prompt is ever requested):
  - native: the owner "software" sentence gets one model call, no tool, a reply,
    and core.golden.evaluate passes its case
  - native: a guest turn takes the legacy path
  - legacy: the existing routing tests pass unchanged
  - the remember/recall rule still returns early on both settings
Run: pytest tests/agents/test_front_door.py tests/ -q --no-cov; ruff and black on the CI scope
Out of scope: deleting the old path, changing the default, slash commands (they
      are handled in the browser today), the UI
Report: files changed, test tail, open questions. 20 lines at most.
```

## WP C1c: Golden runs in a sandbox

Found on November 22: the golden set was run against the live runtime, and the
case "list the files in the docs folder" made the old path ingest the owner's
configured ingest root. Golden runs must not touch real data.

```
Tier: Sonnet, medium
Branch: claude/wp-c1c-golden-sandbox, from origin/main
Read: scripts/golden_run.py (whole file) ; core/runtime_paths.py lines 30-110
      core/config.py: the settings classes for document_rag, filesystem roots
        and memory_manager (find them by name; read only those)
      core/turn_metrics.py (signatures)
Do not read: docs/implementation/, docs/archive/, docs/roadmap/revival-2026-07.md, agents/
Build:
  - scripts/golden_run.py gains --sandbox DIR (required unless --live is given
    with --i-understand-this-touches-real-data). In sandbox mode, before the
    system is built: every runtime data, log, session, memory and workspace
    path points inside DIR; filesystem read roots and document ingest roots
    are set to DIR/files only (seeded with three small text files and a docs/
    folder); auto-ingest is off; self-repair restart is off; the project root
    for verified edits is a throwaway copy of tests/golden/sandbox_project/
    placed in DIR (a tiny package with parser.py, config_loader.py and
    tests/test_dates.py so the repair cases have something to act on)
  - the runner refuses to start if any resolved path is outside DIR
  - the runner calls turn_metrics.note_event for each event, so the turn record
    carries first_event_ms and first_result_ms
  - a `--timeout-seconds` per case (default 180); a case that exceeds it is
    cancelled, counted as failed, and the run continues
  - the results table is rewritten after every case, not only at the end
Acceptance (tests/core/test_golden_sandbox.py, no Ollama):
  - with --sandbox, the built config has no path outside DIR (walk every path
    and root setting) and the owner's configured roots are gone
  - without --sandbox or the two live flags, the script exits 2 before building
    anything
  - a fake control center that never finishes is cancelled at the timeout and
    the next case still runs; the table on disk has both rows
Run: pytest tests/core/test_golden_sandbox.py tests/core/test_golden.py -q --no-cov;
     ruff and black on the CI scope plus scripts/golden_run.py
Out of scope: changing any agent or tool; the cases themselves
Report: files changed, test tail, open questions. 20 lines at most.
```

## C3d and C4 contract

Reviewed against `main` @ `56fb2be`, after the first live runs of the native
loop (25 of 30 golden cases). Facts the briefs rest on:

- The loop reads history through `to_llm_format`, which returns role and
  content only (`core/schemas.py`, about line 447). The context block a user
  message was sent with is not stored, so on the next turn that message differs
  from what the model saw and everything after the system message is re-read.
- The user message is added to the conversation before the run, with no
  metadata (`web/server.py:277`); sessions persist message metadata
  (`core/session_store.py:32-48`).
- Tools have no per-call context. The old path injected `conversation_history`
  and `user_role` as arguments (`agents/orchestrator_tool_helpers.py:180-247`);
  the native loop injects nothing, so `read_conversation_history` and
  `analyze_conversation` get no conversation, and tools that read `user_role`
  fall back to their default.
- `read_file` and `list_directory` declare `user_role` as a schema property, so
  a model could set its own role.
- All 30 registered tools are offered to the model on every round. That
  includes `restart_app`, `network_control`, `guest_set_age_band` and
  `ask_claude`.

**As-sent history (C3d).** When the loop sends a user message with a context
block, it stores that block on the message: `metadata["sent_context"]` (the
exact text appended after the user's words). When building history it rebuilds
each earlier user message as content plus its stored block, so earlier turns
are byte-identical to what was sent. Messages without the key are used as they
are.

**History budget (C3d).** The block trim from C3b stays, with a second
trigger: an estimated token count (characters divided by 4, plus the tool
payload) above 60% of `ollama_settings.num_ctx` cuts back to
`history_turns` in the same block-wise way. No summarising call in this
package.

**Tool context (C4a).** `core/tool_context.py`:

```python
@dataclass(frozen=True)
class ToolContext:
    session_id: str | None
    user_role: str
    conversation_history: Any | None      # ConversationHistory
    turn_id: str | None
def current_tool_context() -> ToolContext | None      # ContextVar, default None
def use_tool_context(ctx: ToolContext) -> ContextManager[None]
CONTEXT_PARAMS = frozenset({"user_role", "conversation_history", "session_id"})
```

The native loop sets it for the turn. Tools read the context first and fall
back to their argument only when no context is set (the old path keeps
working). On the native path the names in `CONTEXT_PARAMS` are removed from
every schema shown to the model, and a call that supplies one is a schema
failure. The model never chooses its own role.

**Tool tiers (C4b).** `conversation.tools`:

| Tier | Meaning | Default members |
|------|---------|-----------------|
| `always` | Sent with full schema every round | `read_file`, `list_directory`, `write_file`, `datetime`, `calculator`, `web_search`, `document_search`, `search_archival_memory`, `read_conversation_history`, `knowledge_log_add_fact`, `find_tools`, `use_tool` |
| `lookup` | Found with `find_tools`, run with `use_tool` | every other registered tool not listed below |
| `owner_approval` | Refused on the native path until the write-gate cards exist (C7) | `restart_app`, `network_control`, `guest_set_age_band`, `ask_claude` |

- `find_tools(query)` returns up to five matches from the `lookup` tier: name,
  description and the normalised schema. Matching is plain word overlap on
  name and description; no embedding call.
- `use_tool(name, arguments)` runs a `lookup` tool. The guard validates
  `arguments` against that tool's real schema and applies every rule in the
  tool-call rules above under the real tool name. The tool list sent to the
  model never changes during a conversation, so the cache is kept.
- An `owner_approval` tool called by either route gets a `reject` verdict that
  says owner approval is needed and is not available yet.
- A `tool_call` event always carries the real tool name.
- `always` descriptions (C4d): one short sentence, at most one "Not for ..." and one example; parameter descriptions are phrases (the 40-character floor is withdrawn).
- The sent schema drops `title`, `default`, `examples`, `additionalProperties` and empty `required` (the guard keeps the full schema); payload at most 8,800 characters, 800 per tool.

## WP C3d: As-sent history

```
Tier: Opus, medium
Branch: claude/wp-c3d-as-sent-history, from origin/main
Read: this file, "C3d and C4 contract"
      agents/native_tool_orchestrator.py (whole) ; agents/native_loop_support.py (whole)
      core/prompt_prefix.py (whole) ; core/schemas.py lines 400-460
Do not read: docs/implementation/, docs/archive/, docs/roadmap/revival-2026-07.md
Build:
  - store sent_context on the current user message when a context block is sent
    (the current message is the last user message in conversation_history; if
    there is none, skip silently)
  - build history from conversation_history.messages, not to_llm_format, and
    rebuild earlier user messages with their stored block
  - the token-estimate trigger in history_window
  - turn the strict xfail in tests/agents/test_native_tool_orchestrator.py into
    a passing test: the fingerprint over the shared span is equal across three
    consecutive turns with different context each turn
Acceptance: that test; a session saved and loaded through core/session_store
      keeps sent_context and gives the same fingerprint; a history over the
      budget is cut on a user message and the cut holds for the next turn;
      messages without sent_context pass through unchanged;
      pytest tests/ -q --no-cov passes
Run: pytest tests/agents/test_native_tool_orchestrator.py tests/agents/test_native_loop_support.py tests/ -q --no-cov; ruff and black on the CI scope
Out of scope: summarising old turns, storing tool messages in history, the old path
Report: files changed, test tail, open questions. 20 lines at most.
```

## WP C4a: Tool context

```
Tier: Opus, medium. Review: independent, medium
Branch: claude/wp-c4a-tool-context, from the C3d branch (stacked; merge C3d first)
Read: this file, "C3d and C4 contract"
      agents/orchestrator_tool_helpers.py lines 180-247 (what the old path injects; do not edit)
      tools/conversation_history_tool.py (whole) ; tools/file_tools.py lines 15-45, 150-180
      tools/web_search_tool.py lines 70-90 ; tools/guest_audit_tool.py lines 20-30, 75-80, 150-155
      tools/guest_profile_tool.py lines 28-38 ; tools/knowledge_log_tool.py lines 28-36, 58-66
      agents/native_tool_orchestrator.py, agents/native_loop_support.py (build_tool_specs)
      core/tool_call_guard.py (normalise_schema, TurnGuard.check signatures)
Do not read: docs/implementation/, docs/archive/, docs/roadmap/revival-2026-07.md
Build:
  - core/tool_context.py as in the contract
  - the native loop sets the context for the whole turn (session, role from the
    turn's kwargs with "owner" only when the caller said so, history, turn id)
  - each tool listed above reads the context first, its argument second
  - build_tool_specs strips CONTEXT_PARAMS from the schemas it shows and gives
    the guard; a call that supplies one is rejected
Acceptance (tests/core/test_tool_context.py, tests/agents/test_native_tool_orchestrator.py):
  - on the native path read_conversation_history returns the turn's conversation
    with no argument from the model
  - a model call passing user_role="owner" to read_file is rejected by the guard
  - with no context set, every tool behaves exactly as before (old-path tests unchanged)
  - two concurrent turns keep their contexts apart
  - pytest tests/ -q --no-cov passes
Run: pytest tests/core/test_tool_context.py tests/agents/ tests/tools/ -q --no-cov; ruff and black on the CI scope
Out of scope: tiers and find_tools (C4b), guests on the native path, the old path's injection code
Report: files changed, test tail, open questions. 20 lines at most.
```

## WP C4b: Tool tiers and lookup

```
Tier: Opus, medium. Review: independent, medium
Branch: claude/wp-c4b-tool-tiers, from the C4a branch (stacked; merge C4a first)
Read: this file, "C3d and C4 contract" and "Tool-call rules"
      agents/native_tool_orchestrator.py, agents/native_loop_support.py
      core/tool_call_guard.py (whole) ; core/config.py ConversationSettings
      the get_schema and description of each tool in the `always` tier
      tests/golden/turns.yaml
Do not read: docs/implementation/, docs/archive/, docs/roadmap/revival-2026-07.md
Build:
  - conversation.tools settings with the three tiers and the defaults above; a
    registered tool named in no tier is `lookup`; a name in a tier that is not
    registered is logged once and ignored
  - tools/tool_lookup.py: find_tools and use_tool as in the contract
  - the loop sends only the `always` tier; use_tool is unwrapped before the
    guard so every rule and every event uses the real tool name
  - owner_approval tools are rejected by both routes
  - rewritten descriptions and examples for the `always` tier
  - golden cases: add two (a lookup-tier tool reached through find_tools then
    use_tool; an owner_approval tool asked for and refused)
Acceptance (tests/agents/test_tool_tiers.py):
  - the payload sent to the model is identical across rounds and turns, and
    holds exactly the `always` tier
  - use_tool with bad arguments counts a schema failure against the real tool;
    use_tool naming an `always` or unknown tool is rejected
  - restart_app is rejected through a direct call and through use_tool, with no execution
  - find_tools never returns an owner_approval tool's schema
  - events carry the real tool name
  - pytest tests/ -q --no-cov passes
Run: pytest tests/agents/test_tool_tiers.py tests/ -q --no-cov; ruff and black on the CI scope
Out of scope: approval cards (C7), skills (C5), guests
Report: files changed, test tail, open questions. 20 lines at most.
```
