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
