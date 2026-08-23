# SM12X-LLM-BENCH `full` — Ornith-1.5-35B-A3B NVFP4-W4A16

Date (report close): 2026-08-23T06:14:57Z
Client: sm12x-llm-bench 0.2.0 · profile `full` · think-off · C=4
Serve: SGLang, EAGLE K=1, ctx 32768, FP8 KV, triton attn, marlin MoE
Hardware: NVIDIA DGX Spark GB10 / SM12.1, single node
Verdict: **complete**, 15/15 PASS, infra_errors=0, valid for publish

Protocol audit: [`PROTOCOL.md`](PROTOCOL.md)

## Quality

| Lane | n | correct | accuracy | Wilson 95% |
|---|---:|---:|---:|---|
| GSM8K (full test, last-bolded) | 1319 | 1170 | **88.70%** | 86.88–90.30 |
| MMLU-Pro | 1000 | 755 | **75.50%** | 72.74–78.06 |
| GPQA Diamond | 198 | 124 | **62.63%** | 55.71–69.06 |
| ARC-Easy (even-spread) | 400 | 376 | **94.00%** | 91.23–95.94 |
| IFEval (lightweight) | 200 | 190 | **95.00%** | 91.04–97.26 |
| HumanEval pass@1 | 164 | 146 | **89.02%** | — |

## Official tool-calling

| Lane | official score | note |
|---|---|---|
| BFCL V4 multi_turn_base | **96/200 = 48.0%** | 0 transport errors |
| BFCL V4 AST micro | **203/600 = 33.8%** | multiple 67/200, parallel 102/200, parallel_multiple 34/200 |

AST misses are model/tool-state, not harness/HTTP.

## Systems

| Lane | result |
|---|---|
| latency C1 | ttft p50 83.9 ms · ITL p50 14.4 ms · 66.1 tok/s |
| concurrency C1–C8 | PASS, 0 failed |
| throughput decode | p50 **62.6 tok/s** (2048-out reps) |
| throughput prefill | p50 **~4.0k tok/s** at 8192 prompt tokens |
| NIAH 25/50/90 of 32k−512 | **3/3 PASS** (8064 / 16128 / 29030) |
| MTP accept_len | **1.975** |
| longctx 25/50/90 | 3/3 completed |

## Files

- `METRICS.json` — reduced lane summaries
- `QUALITY-SCORES.json` — 3281 quality rows, no stems/responses
- `PROTOCOL.md` — independent recount + disclosures
