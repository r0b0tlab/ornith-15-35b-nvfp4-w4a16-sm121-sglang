# Ornith 1.5 35B-A3B r0b0t — full Aider Polyglot run

Raw artifacts from the completed 225-task Aider Polyglot benchmark on the
RTX 5090 profile for
[`r0b0tlab/Ornith-1.5-35B-A3B-NVFP4-W4A16`](https://huggingface.co/r0b0tlab/Ornith-1.5-35B-A3B-NVFP4-W4A16).

- Run: `ornith15-35b-a3b-nvfp4-aider-polyglot-20260823`
- Model revision: `dcbc0a25d5b3ce634c2f5d988a81ba598ca7adcc`
- Hardware: NVIDIA GeForce RTX 5090 (32 GB)
- Serving backend: SGLang
- Aider: `0.86.3.dev`
- Date: 2026-08-23
- Workload: 225 Exercism Polyglot tasks, 32 Aider workers, two tries

## Recorded result

| Metric | Value |
|---|---:|
| Pass rate, first attempt | 8.9% (20/225) |
| Pass rate, best of two attempts | 23.6% (53/225) |
| Well-formed cases | 97.8% |
| Error outputs | 29 |
| Malformed responses | 6 |
| Exhausted context windows | 23 |
| Test timeouts | 3 |
| Prompt tokens | 2,957,355 |
| Completion tokens | 481,983 |
| Mean seconds per case | 151.9 |

## Contents

- `logs/`: complete benchmark stdout/stderr, stats, and model label.
- `protocol/`: exact Aider model settings used for the run.
- `scripts/`: the exact launcher and telemetry harness.
- `telemetry/`: GPU and SGLang live-metrics samples for the full run.
- `full-artifacts.tar.gz`: complete `full/` output tree, including every task
  workspace, `.aider.chat.history.md`, `.aider.results.json`, generated edits,
  and test artifacts.
- `SHA256SUMS`: checksums for every uploaded artifact.

The separate upstream Exercism fixture checkout is not duplicated here; the
complete per-task workspaces are preserved in `full-artifacts.tar.gz`.

The exercise content is sourced from the Exercism language tracks and is used
under their respective open-source licenses. See the upstream Aider Polyglot
benchmark for fixture provenance.
