# Ornith 1.5 35B-A3B — matched Aider Polyglot runs

Audited artifacts from two matched 225-task Aider Polyglot benchmarks on one
NVIDIA GeForce RTX 5090 (32 GB). Both runs used the same Aider and Polyglot
revisions, whole-file edit format, two attempts, non-thinking sampler, 262,144
token context, and 12 client workers.

| Model | First attempt | Best of two | API/context errors | Malformed |
|---|---:|---:|---:|---:|
| r0b0t W4A16 | 18/225 (8.0%) | 57/225 (25.3%) | 0 | 2 |
| Official stock NVFP4 | 34/225 (15.1%) | 80/225 (35.6%) | 0 | 1 |

## Runs

- `r0b0t-w4a16/`: [`r0b0tlab/Ornith-1.5-35B-A3B-NVFP4-W4A16`](https://huggingface.co/r0b0tlab/Ornith-1.5-35B-A3B-NVFP4-W4A16), revision `dcbc0a25d5b3ce634c2f5d988a81ba598ca7adcc`.
- `official-stock-nvfp4/`: [`ornith-ai/Ornith-1.5-35B-A3B-NVFP4`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-NVFP4), revision `0f0b1b59b879ccde1353e6ebd0fb10c204d4c544`.

Each directory contains the complete benchmark log, Aider stats, final audit,
locked protocol and model settings, launcher/audit scripts, telemetry, and a
compressed archive of every per-task workspace and result record. Verify files
with the per-run `SHA256SUMS` manifest.

These valid runs supersede the invalid 2026-08-23 result, which used excessive
concurrency and insufficient effective context and is intentionally removed.

