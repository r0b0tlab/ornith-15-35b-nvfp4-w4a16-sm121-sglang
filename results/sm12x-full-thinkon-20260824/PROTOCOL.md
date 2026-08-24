# Protocol audit — SM12X-LLM-BENCH `full` think-on (claim row)

Verdict: **this is the native-thinking claim row.**
`run_status=complete`, `invalid_for_publish=false`, `partial=false`, `--only` unset,
`infra_errors_total=0`. All 15 profile lanes `PASS`.

Thinking was **on** (`enable_thinking` / `thinking` true). Served window
`max_model_len=262144` (model `max_position_embeddings`). Client resumed
after an interrupt that had already finished official BFCL; remaining
profile lanes ran in the same process. `--resume` is not `--only`.

Independent recount of stored quality rows matches the lane summaries.

## What was measured

- Client: `sm12x-llm-bench` 0.2.0, profile `full`, `--resume`
- Model: `Ornith-1.5-35B-A3B` NVFP4-W4A16 + EAGLE K=1
- Window: 262144 · KV pool 3,242,950 tokens · quality `max_tokens` 8192
- Client concurrency 2
- Report written `2026-08-24T13:21:28Z`

ID contract sha `8941db3ef707e1a9dd72b1dcea344993ea0f0ff2faea06fca241cd9ef09d44c6`

| Lane | Pin / n | Recount |
|---|---|---|
| GSM8K | sha `3730d312…` n=1319 | 1197/1319, 0 trunc, 0 unparseable |
| MMLU-Pro | sha `a229ed37…` n=1000 | 812/1000, 35 length |
| GPQA Diamond | sha `a8472c5a…` n=198 | 146/198, 38 length |
| QA ARC-Easy | even-spread 400 | 377/400 |
| IFEval | 200 | 173/200 **lightweight constraint scorer, not official IFEval** |
| HumanEval | 164 | 151/164 pass@1 |
| BFCL-MT | official `multi_turn_base` 200 | 136/200 |
| BFCL-AST | official micro 68+108+38 | 214/600 |
| NIAH | advertised 262144; single 25/50/90 + multi-key 33/66 | 5/5 found |
| MTP | `/metrics` accept_len after a decode probe | 1.925 |
| longctx | 25/50/90 of 262144−8192 | 3/3 completed |

GSM8K scoring is last-bolded. Quality JSON has item ids, gold/pred letters
or numbers, timing, HTTP — no stems.

## NIAH / longctx actual usage

Haystack constructor uses 4 chars/token, so **actual** `prompt_tokens` are
below the nominal depth. Label the cell by usage, not the constructor:

| cell | nominal depth | actual prompt tokens | result |
|---|---:|---:|---|
| single 25% | 65408 | 38098 | found |
| single 50% | 130816 | 76153 | found |
| single 90% | 235468 | 137042 | found |
| multi 33% | 86338 | 50354 | 4/4 keys |
| multi 66% | 172677 | 100588 | 4/4 keys |

longctx decode: 25% produced 122658 tokens then `stop`; 50% stopped at 924;
90% hit `length` at 33588 (measured truncation).

## Not protocol failures

- BFCL 68% / AST 35.7% and remaining misses are **model** scores.
- Isolated `finish_reason=length` with tokens produced is scored wrong, not infra.
- IFEval is the lightweight checker, labeled.

## Relation to 2026-08-23 think-off

That suite forced thinking off on every lane and used a 32768 serve cap.
It remains a diagnostic only (`results/sm12x-full-20260823/`).
