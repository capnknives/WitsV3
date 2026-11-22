# Golden set results

Run with `scripts/golden_run.py --sandbox <dir>`. Checks are code over the
events of each turn; no model judges a reply. Times are milliseconds to the
first result.

## Baseline: old path, `qwen3:8b`, November 22, 2026

`main` @ `565d864`, `conversation.front_door: legacy`, `num_ctx` 8192, thinking
off, model fully on the GPU, embeddings on the CPU. One run per case, in a
sandbox folder. 20 of 30 passed.

| case | pass | first_result_ms | tools seen |
|---|---|---|---|
| owner-software-statement | 1/1 | 1081 | - |
| chat-greeting | 1/1 | 362 | - |
| chat-app-opinion | 1/1 | 3144 | - |
| chat-class-schedule | 1/1 | 1481 | - |
| chat-python-pet | 1/1 | 1377 | - |
| chat-api-curiosity | 1/1 | 3290 | - |
| chat-feeling-tired | 1/1 | 1570 | - |
| chat-thanks | 1/1 | 461 | - |
| tool-file-read | 1/1 | 43 | read_file |
| tool-file-list | 1/1 | 8183 | list_directory |
| tool-web-search | 1/1 | 11317 | web_search |
| tool-memory-search | 1/1 | 35339 | search_archival_memory |
| tool-datetime | 1/1 | 5338 | datetime |
| tool-document-search | 1/1 | 29854 | document_search, ingest_documents |
| code-rename-script | 0/1 | 3791 | - |
| code-small-website | 0/1 | 29269 | - |
| code-date-function | 0/1 | 19430 | - |
| code-backup-script | 0/1 | 3677 | - |
| repair-parser-bug | 0/1 | 4127 | - |
| repair-config-loader | 0/1 | 4282 | - |
| repair-test-failure | 0/1 | - | - |
| write-short-poem | 1/1 | 2692 | - |
| write-email-draft | 0/1 | 2408 | - |
| write-story-opening | 1/1 | 4123 | - |
| command-save-conversation | 1/1 | 89 | read_conversation_history, write_file |
| command-help | 1/1 | 1206 | - |
| command-exact-math | 0/1 | 233 | calculator |
| followup-shorter | 1/1 | 3235 | - |
| followup-yes-do-it | 0/1 | 10173 | document_search |
| followup-what-about-tomorrow | 1/1 | 13276 | calculator, datetime |

Reading the failures:

- **Coding and repair (7).** The cases expect a tool call. The old coding and
  self-repair agents do their work without reporting it as tool calls, so the
  check cannot see it. These are a limit of the old path's events, not proof
  the work failed. They become meaningful on the new loop, where every tool
  call is reported. `repair-test-failure` produced no result at all.
- **`command-exact-math`.** The case names a tool that does not exist
  (`math_operations`); the real tool is `calculator`, and it was called. The
  case is wrong and will be corrected.
- **`write-email-draft`** and **`followup-yes-do-it`** are real misses on the
  old path: the first broke a must-not-route rule, the second ran a document
  search for "yes please do that".

Slow spots on the old path, for comparison with the new loop: memory search
35 s, document search 30 s (it also started indexing), file listing 8 s, a
date question 5 s.

Before `num_ctx` was fixed (PR #51), the first case took 42.8 s cold and 25.6 s
warm, and a greeting 11.1 s.

## New loop: `conversation.front_door: native`, `qwen3:8b`, November 22, 2026

`main` @ `56fb2be`, same model, settings and sandbox as the baseline. One run
per case. 25 of 30 passed, against 20 on the old path. The whole set ran in
about a minute.

| case | pass | first_result_ms | tools seen | old path ms |
|---|---|---|---|---|
| owner-software-statement | 1/1 | 748 | - | 1081 |
| chat-greeting | 1/1 | 671 | - | 362 |
| chat-app-opinion | 1/1 | 958 | - | 3144 |
| chat-class-schedule | 1/1 | 882 | - | 1481 |
| chat-python-pet | 1/1 | 743 | - | 1377 |
| chat-api-curiosity | 1/1 | 517 | - | 3290 |
| chat-feeling-tired | 1/1 | 3185 | - | 1570 |
| chat-thanks | 1/1 | 530 | - | 461 |
| tool-file-read | 1/1 | 1030 | read_file | 43 |
| tool-file-list | 1/1 | 1017 | list_directory | 8183 |
| tool-web-search | 1/1 | 5067 | web_search | 11317 |
| tool-memory-search | 1/1 | 1118 | search_archival_memory | 35339 |
| tool-datetime | 1/1 | 1081 | datetime | 5338 |
| tool-document-search | 0/1 | 673 | - | 29854 |
| code-rename-script | 1/1 | 3406 | python_execute | 3791 (fail) |
| code-small-website | 0/1 | 10061 | - | 29269 (fail) |
| code-date-function | 1/1 | 5149 | write_file | 19430 (fail) |
| code-backup-script | 0/1 | 3987 | - | 3677 (fail) |
| repair-parser-bug | 1/1 | 4658 | apply_code_fix | 4127 (fail) |
| repair-config-loader | 1/1 | 1613 | diagnose_log_errors | 4282 (fail) |
| repair-test-failure | 1/1 | 2414 | run_test_suite | none (fail) |
| write-short-poem | 1/1 | 1345 | - | 2692 |
| write-email-draft | 1/1 | 1819 | - | 2408 (fail) |
| write-story-opening | 1/1 | 1582 | - | 4123 |
| command-save-conversation | 0/1 | 686 | - | 89 |
| command-help | 1/1 | 2180 | - | 1206 |
| command-exact-math | 1/1 | 876 | calculator | 233 (case was wrong) |
| followup-shorter | 1/1 | 373 | - | 3235 |
| followup-yes-do-it | 0/1 | 8456 | web_search | 10173 (fail) |
| followup-what-about-tomorrow | 1/1 | 1049 | datetime | 13276 |

The five misses, all on the model's choice of action with today's thin
instructions, and the packages that address them:

- `tool-document-search`: answered without searching (C4, tool list and descriptions).
- `code-small-website`, `code-backup-script`: wrote the code in the reply
  instead of saving a file. The cases expect a tool call; whether a reply with
  code should count is a case decision for C5.
- `command-save-conversation`: did not save. The tool needs the session, which
  the new loop does not hand to tools yet (C4).
- `followup-yes-do-it`: ran a web search for "yes please do that" on both
  paths. Needs the earlier turns read properly (C3d, as-sent history).

A pass here means the expected kind of action happened. It does not judge the
quality of the reply or of the code written. One run per case is a first
reading, not a pass rate; the gate uses three runs.

An earlier native run the same day scored 16 of 30: nine turns asked the model
router for models that are not installed and failed at once (fixed in PR #60).
