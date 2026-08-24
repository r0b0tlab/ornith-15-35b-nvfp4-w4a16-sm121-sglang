# SM12X-LLM-BENCH `full` think-on — Ornith-1.5-35B-A3B NVFP4-W4A16

**Claim row.** Native thinking on. ctx 262144. EAGLE K=1.

Date (report close): 2026-08-24T13:21:28Z
Client: sm12x-llm-bench 0.2.0 · profile `full` · `--resume` after official BFCL
Serve: SGLang, EAGLE K=1, ctx 262144, FP8 KV, triton attn, marlin MoE
Hardware: NVIDIA DGX Spark GB10 / SM12.1, single node
Verdict: **complete**, 15/15 PASS, infra_errors=0, valid for publish

Protocol audit: [`PROTOCOL.md`](PROTOCOL.md)

## Quality

| Lane | n | correct | accuracy | Wilson 95% |
|---|---:|---:|---:|---|
| GSM8K (full test, last-bolded) | 1319 | 1197 | **90.75%** | 89.07–92.20 |
| MMLU-Pro | 1000 | 812 | **81.20%** | 78.66–83.50 |
| GPQA Diamond | 198 | 146 | **73.74%** | 67.20–79.37 |
| ARC-Easy (even-spread) | 400 | 377 | **94.25%** | 91.52–96.14 |
| IFEval (lightweight) | 200 | 173 | **86.50%** | 81.07–90.55 |
| HumanEval pass@1 | 164 | 151 | **92.07%** | 86.91–95.31 |

## Official tool-calling

| Lane | official score | note |
|---|---|---|
| BFCL V4 multi_turn_base | **136/200 = 68.0%** | 0 transport errors |
| BFCL V4 AST micro | **214/600 = 35.7%** | multiple 68/200, parallel 108/200, parallel_multiple 38/200 |

## Systems

| Lane | result |
|---|---|
| latency C1 | ttft p50 78.8 ms · ITL p50 13.1 ms · 74.9 tok/s |
| concurrency C1–C8 | PASS, 0 failed |
| throughput decode | p50 **67.5 tok/s** |
| throughput prefill | p50 **~15.3k tok/s** (includes cache-hot reps) |
| NIAH advertised 262144 | **5/5 PASS** — single 25/50/90 + 4-key 33/66 |
| MTP accept_len | **1.925** |
| longctx 25/50/90 | 3/3 (90% measured `length`) |

NIAH actual prompt tokens are below the 4-char constructor (90% cell = 137042). See PROTOCOL.md.

## Files

- `METRICS.json` — reduced lane summaries
- `QUALITY-SCORES.json` — 3281 quality rows, no stems/responses
- `PROTOCOL.md` — independent recount + disclosures
