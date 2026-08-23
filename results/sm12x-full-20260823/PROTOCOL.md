# Protocol audit — SM12X-LLM-BENCH `full` on Ornith-1.5-35B-A3B NVFP4-W4A16

**This row forced thinking off on every lane.** That is a protocol error for a
native-quality claim. Keep the numbers as a **think-off diagnostic** only.
Do not treat them as the model's native (template-default thinking-on) suite.

Harness execution of that think-off profile was otherwise clean:
`run_status=complete`, `partial=false`, `--only` unset, `infra_errors_total=0`.
All 15 profile lanes `PASS`. Independent quality recount matches the summaries.

## What was measured

- Client: `sm12x-llm-bench` 0.2.0, profile `full`
- Model served: `Ornith-1.5-35B-A3B` (NVFP4-W4A16 candidate B + EAGLE K=1)
- Window: `max_model_len=32768` (memory-safe; not the 262144 think-on card)
- Think-off: `chat_template_kwargs={enable_thinking:false, thinking:false}`
- Client concurrency 4
- Report written `2026-08-23T06:14:57Z`

ID contract sha `8941db3ef707e1a9dd72b1dcea344993ea0f0ff2faea06fca241cd9ef09d44c6`

| Lane | Pin / n | Recount |
|---|---|---|
| GSM8K | sha `3730d312…` n=1319 full test | 1170/1319, 0 trunc, 0 unparseable, all `finish=stop` |
| MMLU-Pro | sha `a229ed37…` n=1000 | 755/1000, 1 length, 2 unparseable |
| GPQA Diamond | sha `a8472c5a…` n=198 full | 124/198, 1 length |
| QA ARC-Easy | even-spread 400 | 376/400 |
| IFEval | 200 | 190/200 **lightweight constraint scorer, not official IFEval** |
| HumanEval | 164 | 146/164 pass@1, subprocess grader + 10s timeout |
| BFCL-MT | official `multi_turn_base` 200 | 96/200, 0 inference-transport errors |
| BFCL-AST | official multiple+parallel+parallel_multiple | 203/600 micro (see recording note) |
| NIAH | 25/50/90 of (32768−512) = 8064/16128/29030 | 3/3 found |
| MTP | `/metrics` `sglang:spec_accept_length` after a decode probe | 1.975 |
| longctx | 6144/12288/22118 | 3/3 completed |

GSM8K scoring is last-bolded / last-number (house rule). Item records store gold/pred only — no stem text.

## Not protocol failures

- BFCL 48% / AST 33.8% and GPQA 62.6% are **model** scores.
- Isolated `finish_reason=length` rows with tokens produced are scored wrong, not infra.
- HumanEval `unparseable` count is failed unit tests, not missing HTTP.

## Recording / harness notes (disclosed, not run fails)

1. **BFCL-AST accuracy field in the live `report.json` was 0.51** — the first `*score*.json` the then-running client found (`parallel` only). Official category files are 67/200, 102/200, 34/200. Published number here is the micro-average **203/600 = 33.8%**. Client later learned to micro-average; this process did not pick that up.
2. **IFEval is a lightweight constraint checker**, labeled in-row. Not the official instruction-following eval.
3. **Client `telemetry` sampled the bench host GPU** (idle). Do not treat `report.telemetry` as serve-node utilization. Serve-node `nvidia-smi` during BFCL was ~94–96% / ~70°C / ~44 W.
4. **Throughput prefill rows used 8192 prompt tokens** (actual usage), not a 14k constructor. One prefill rep is prefix-cache hot (~15.8k tok/s) — median ~4.0k tok/s.
5. This NIAH is **32k-window**, think-off, MTP on. It is not the historical 262k think-on NIAH row.

## Safety

GPQA plaintext was cache-only and is not in this bundle. Quality JSON has item ids, letters/numbers, timing, and HTTP — no stems or model text.
