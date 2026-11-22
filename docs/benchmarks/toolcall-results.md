## Tool-calling eval 2026-11-21 18:45:28

Host: DESKTOP-BOQHPAH (Windows 11) | Server: http://localhost:11434 | 3 run(s) per case | pass rates use all attempts as the denominator, so server errors and timeouts count as failures (also shown in Errors). Strict fails a call that adds a schema-valid optional argument; lenient passes it

Run with `scripts/toolcall_eval.py` from PR #19 (36 cases). Small models only: the two large models were not run because the PC shut itself off during an earlier attempt and long runs are on hold until it is cleaned.

| Model | Backend | Think | single | hard | none | parallel | chain | recovery | Overall (strict) | Overall (lenient) | Median latency (s) | Errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma4:e4b | ollama | on | 92% | 72% | 67% | 75% | 75% | 75% | 79% | 81% | 2.24 | 0 |
| gemma4:e4b | ollama | off | 83% | 83% | 83% | 75% | 75% | 0% | 72% | 72% | 0.25 | 0 |
| qwen3:8b | ollama | on | 83% | 83% | 83% | 50% | 0% | 75% | 69% | 83% | 3.49 | 3 |
| qwen3:8b | ollama | off | 75% | 100% | 100% | 25% | 25% | 100% | 75% | 92% | 0.74 | 0 |
| qwen3.5:9b | ollama | on | 92% | 67% | 100% | 75% | 75% | 75% | 83% | 92% | 3.27 | 3 |
| qwen3.5:9b | ollama | off | 92% | 67% | 100% | 75% | 100% | 50% | 83% | 83% | 1.15 | 3 |

Failed cases:

- gemma4:e4b [ollama, think=on]: c01 (step 2: extra optional argument) x3, e02 (step 2: no tool call) x3, h02 (no tool call) x2, h03 (wrong arguments) x3, n02 (unexpected tool call) x3, n05 (unexpected tool call) x3, p04 (wrong arguments) x3, s08 (wrong arguments) x3
- gemma4:e4b [ollama, think=off]: c01 (step 2: wrong arguments) x3, e01 (step 2: no tool call) x3, e02 (step 2: no tool call) x3, e03 (step 2: no tool call) x3, e04 (step 2: no tool call) x3, h03 (no tool call) x3, n02 (unexpected tool call) x3, p04 (wrong arguments) x3, s08 (wrong arguments) x3, s09 (no tool call) x3
- qwen3:8b [ollama, think=on]: c01 (extra optional argument) x3, c02 (wrong tool) x3, c03 (wrong tool) x3, e04 (step 2: wrong arguments) x3, h05 (wrong arguments) x3, n02 (unexpected tool call) x3, p01 (extra optional argument) x3, p03 (extra optional argument) x3, s02 (extra optional argument) x3, s06 (extra optional argument) x3
- qwen3:8b [ollama, think=off]: c01 (step 2: wrong tool) x3, c03 (step 2: extra optional argument) x3, c04 (extra optional argument) x3, p01 (extra optional argument) x3, p03 (extra optional argument) x3, p04 (wrong arguments) x3, s02 (extra optional argument) x3, s06 (extra optional argument) x3, s08 (no tool call) x3
- qwen3.5:9b [ollama, think=on]: c04 (extra optional argument) x3, e04 (step 2: wrong arguments) x3, h04 (wrong arguments) x3, p01 (extra optional argument) x3, s06 (extra optional argument) x3
- qwen3.5:9b [ollama, think=off]: e01 (step 2: no tool call) x3, e02 (step 2: no tool call) x3, h04 (wrong arguments) x3, p04 (wrong arguments) x3, s08 (wrong arguments) x3

Notes from the lead (not produced by the script):

- Scores are provisional. Cases `p04`, `s08` and `n02` failed for most or all models, which usually means the case is ambiguous or too strict. They need review before these numbers are used to pick a model.
- Every failure repeated on all three runs (temperature 0), so this is effectively 36 cases per model.
- The report does not list which cases ended in a server error (the "Errors" column). That is a gap in the script to fix.

