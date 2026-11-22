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
    tools: none                         # none | any | [names] (every name listed must be called)
    forbid_tools: []                    # names that must not be called
    forbid_routes: [coding]             # coding | self_repair | book | orchestrator
    reply: required                     # required | optional
```

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
