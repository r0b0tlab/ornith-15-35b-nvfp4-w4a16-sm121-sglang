# Ornith-1.5-35B-A3B NVFP4-W4A16 (SGLang, SM12X)

NVFP4 **W4A16 weight-only** quantization of
[`ornith-ai/Ornith-1.5-35B-A3B`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B)
(MIT, revision `e4dfb35a93d4b6822a811a7676f3488514abe7e2`), produced with
NVIDIA Model Optimizer (dev `913f5e224`) using the official `w4_nvfp4`
recipe preset, with the bundled MTP head kept in BF16 and served via SGLang
EAGLE/nextn speculative decoding.

This repo is the **reproduction package**: click-run serve for SM12X
(SM 12.0 + SM 12.1), the SM12X-LLM-BENCH claim-row evidence, the older
200-question harness, checkpoint audit tooling, and the quantization recipes.

- Checkpoint: [huggingface.co/r0b0tlab/Ornith-1.5-35B-A3B-NVFP4-W4A16](https://huggingface.co/r0b0tlab/Ornith-1.5-35B-A3B-NVFP4-W4A16)
- Containers: `ghcr.io/r0b0tlab/ornith-15-35b-nvfp4-w4a16-sm121-sglang:latest` (GB10/SM121) and `:rtx5090` (x86 SM120)
- Claim-row bench: [SM12X-SOCOM/SM12X-LLM-BENCH](https://github.com/SM12X-SOCOM/SM12X-LLM-BENCH) `full` · native thinking on

Independent community quantization by r0b0tlab. Not affiliated with or
endorsed by Ornith AI or NVIDIA.

### Hardware

| Part | CC | Memory | Compose / profile | Default ctx | Status |
|---|---|---|---|---:|---|
| DGX Spark / GB10 | SM 12.1 | 121 GB unified | `docker-compose.yml` | **262144** | **validated** (claim row) |
| RTX PRO 6000 | SM 12.0 | ~96 GB | `docker-compose.rtx-pro-6000.yml` | 131072 | sized, not physically signed off |
| GeForce RTX 5090 / 50-class | SM 12.0 | 32 GB | `docker-compose.rtx5090.yml` | 32768 | sized, not physically signed off |

`./scripts/click-run.sh` picks the row from `nvidia-smi` compute capability + memory.

---

## Click-run

```bash
git clone https://github.com/r0b0tlab/ornith-15-35b-nvfp4-w4a16-sm121-sglang.git
cd ornith-15-35b-nvfp4-w4a16-sm121-sglang
chmod +x scripts/click-run.sh scripts/bench-sm12x.sh
./scripts/click-run.sh
# waits for /health, then:
./scripts/bench-sm12x.sh    # SM12X-LLM-BENCH full, native think-on
```

Override detection with `./scripts/click-run.sh --profile spark|pro6000|rtx5090`.
Override context with `CONTEXT_LENGTH=32768 ./scripts/click-run.sh`.

Smoke (native thinking — do **not** force think-off for a quality claim):

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Ornith-1.5-35B-A3B",
    "messages": [{"role": "user", "content": "What is 19*23? Answer with just the number."}],
    "temperature": 0, "max_tokens": 256
  }'
```

If you omit `MODEL_PATH` / a local `models/` tree, the entrypoint pulls
`r0b0tlab/Ornith-1.5-35B-A3B-NVFP4-W4A16` from Hugging Face at boot.

---

## Results

### 200-question suite (historical, greedy, thinking disabled)

| Family | n | BF16 baseline | This model (base AR) | This model + MTP |
|---|---|---|---|---|
| GSM8K (flex extraction) | 80 | 76.25% (61) | 76.25% (61) | **78.75% (63)** |
| HumanEval pass@1 | 40 | 92.50% (37) | 92.50% (37) | **95.00% (38)** |
| IFEval (strict subset) | 40 | 90.00% (36) | 90.00% (36) | 90.00% (36) |
| Agentic coding | 20 | 70.00% (14) | 75.00% (15) | **85.00% (17)** |

**No measurable quality loss** — every family at or above the BF16 baseline;
deltas are within sample noise (±5–11 pp at n = 80/40/20).

### Speculative decoding (BF16 MTP head, K=1 draft)

- Mean accept length **1.737**, mean accept rate **0.737** over 964 decode batches
- Decode throughput (single request, CUDA graphs on): **63–77 tok/s**,
  vs 38.6 tok/s measured for an experts-only NVFP4 variant on the same box
- CUDA graphs: decode `full` + prefill `breakable` (SGLang per-phase
  defaults) — both verified captured at boot

### Serving performance (think-on, 262,144-token context, DGX Spark GB10)

MTP sweep (EAGLE steps, draft = K+1; 15 reps × ~1,950 tokens, temp 0,
thinking enabled, 262K context):

| K | decode tok/s (e2e) | mean accept len | mean accept rate |
|---|---|---|---|
| K=1 | **80.78** | 1.993 | 0.993 |
| K=2 | 61.39 | 2.083 | 0.541 |
| K=3 | 50.44 | 2.091 | 0.363 |

K=1 wins: the reject-rate penalty (0.99 → 0.36) dominates the longer accept
length. Verified: finish=stop on all reps (reply headroom, no truncation).

Concurrency ladder (r0b0bench lane, K=1, 512-token outputs, 3 reps each):

| concurrency | aggregate out tok/s | completed | failed |
|---|---|---|---|
| C1 | 82.18 | 4 | 0 |
| C2 | 132.72 | 8 | 0 |
| C4 | 194.39 | 16 | 0 |
| C8 | 269.68 | 32 | 0 |
| C16 | 368.57 | 64 | 0 |
| C32 | 481.91 | 128 | 0 |
| C64 | 513.02 | 256 | 0 |

Max-context NIAH (r0b0bench lane, 25/50/90% of 262,144):

| depth (tokens) | with MTP K=1 | base-AR (no spec) |
|---|---|---|
| 65,472 | PASS | — |
| 130,944 | PASS | — |
| 235,699 | **FAIL** (degenerate "!" output, deterministic) | **PASS** (needle retrieved) |

**MTP at ~235k constructor tokens is a confirmed r0b0bench-lane limitation**
(degenerate `"!"` stream). The SM12X claim-row NIAH (below) passed 5/5 with
MTP on; its 90% cell was **137k actual prompt tokens**, not 235k. Do not
treat those as the same depth. For workloads that really sit at ≥230k with
MTP, serve base-AR (`MTP=0`).

Canary lane (r0b0bench protocol): all cases pass with the fixed JSON probe
question and `--tool-call-parser qwen3_coder`; the `structured` case shows
run-to-run nondeterminism on this model (the model sometimes responds to the
"output ONLY this exact JSON" instruction with a prompt-injection hedge and
then complies). Raw: `results/canary-parser-fix.json`.

Raw artifacts: `results/bench-report-thinkon.json`,
`results/niah-90pct-baseAR.json`, `results/niah-90pct-mtp-repro.json`.

### Checkpoint audit

- Scale companions present for every quantized module (30,970/30,970)
- 785 MTP tensors byte-identical (SHA-256) to the source checkpoint
- NVFP4 reconstruction cosine vs BF16: experts 0.9956–0.9973; dense/GDN
  projections 0.967–0.986 (plain max calibration on outlier-rich
  projections; no end-to-end effect on this suite — flagged as residual
  out-of-distribution risk)

Raw per-question rows for all four runs: [`eval/data/`](eval/data/).
Scoring methodology and the scorer bug we found and fixed: see
[Methodology notes](#methodology-notes).

### SM12X-LLM-BENCH `full` think-on (2026-08-24) — claim row

Native thinking on, ctx 262144, EAGLE K=1. 15/15 PASS, infra 0.
Audit: [`results/sm12x-full-thinkon-20260824/PROTOCOL.md`](results/sm12x-full-thinkon-20260824/PROTOCOL.md).

| Lane | n | result |
|---|---:|---|
| GSM8K (full test, last-bolded) | 1319 | **90.75%** (1197) · Wilson 89.1–92.2 |
| MMLU-Pro | 1000 | **81.20%** (812) |
| GPQA Diamond | 198 | **73.74%** (146) |
| ARC-Easy | 400 | **94.25%** (377) |
| IFEval (lightweight scorer) | 200 | **86.50%** (173) |
| HumanEval pass@1 | 164 | **92.07%** (151) |
| BFCL V4 MT official | 200 | **68.0%** (136) |
| BFCL V4 AST official micro | 600 | **35.7%** (214) — 68 / 108 / 38 by category |
| NIAH advertised 262144 | 5 | **5/5 PASS** (single 25/50/90 + multi-key 33/66) |
| latency / decode / MTP | — | 74.9 tok/s C1 · 67.5 tok/s decode · accept 1.925 |

NIAH cells are labeled by actual prompt tokens in the protocol note (constructor overshoots).
IFEval here is a lightweight constraint checker, not the official scorer.

### SM12X-LLM-BENCH `full` think-off (2026-08-23) — diagnostic only

**Protocol error:** thinking was forced off for the entire run. These numbers
are the think-off profile, **not** native thinking-on quality/BFCL.
Do not use them as the claim row.

15/15 lanes completed, infra 0. Audit:
[`results/sm12x-full-20260823/PROTOCOL.md`](results/sm12x-full-20260823/PROTOCOL.md).

| Lane | n | result |
|---|---:|---|
| GSM8K (full test, last-bolded) | 1319 | **88.70%** (1170) · Wilson 86.9–90.3 |
| MMLU-Pro | 1000 | **75.50%** (755) |
| GPQA Diamond | 198 | **62.63%** (124) |
| ARC-Easy | 400 | **94.00%** (376) |
| IFEval (lightweight scorer) | 200 | **95.00%** (190) |
| HumanEval pass@1 | 164 | **89.02%** (146) |
| BFCL V4 MT official | 200 | **48.0%** (96) |
| BFCL V4 AST official micro | 600 | **33.8%** (203) — 67 / 102 / 34 by category |
| NIAH 25/50/90 of 32k | 3 | **3/3 PASS** |
| latency / decode / MTP | — | 66.1 tok/s C1 · 62.6 tok/s 2048-out · accept 1.975 |

This NIAH is the 32768-window think-off ladder, not the 262144 think-on row above.
IFEval here is a lightweight constraint checker, not the official scorer.

---

## Manual compose (if you skip click-run)

Requirements: NVIDIA GPU with Blackwell-class FP4 support, Docker with
nvidia-container-toolkit, ~25 GB disk for the checkpoint.

```bash
hf download r0b0tlab/Ornith-1.5-35B-A3B-NVFP4-W4A16 \
  --local-dir ./models/ornith-15-35b-a3b-nvfp4-w4a16-B
docker compose up -d          # Spark/GB10 claim defaults (262144)
# or: docker compose -f docker-compose.rtx5090.yml up -d
# or: docker compose -f docker-compose.rtx-pro-6000.yml up -d
curl http://127.0.0.1:8000/health
```

Without compose:

```bash
docker run --rm -it --gpus all --ipc host --shm-size 32g -p 8000:8000 \
  -v $PWD/models/ornith-15-35b-a3b-nvfp4-w4a16-B:/models/ckpt:ro \
  -e MODEL_PATH=/models/ckpt \
  ghcr.io/r0b0tlab/ornith-15-35b-nvfp4-w4a16-sm121-sglang:latest
```

If you omit `MODEL_PATH`, the entrypoint defaults to pulling
`r0b0tlab/Ornith-1.5-35B-A3B-NVFP4-W4A16` from Hugging Face at boot
(`HF_HOME` is a named volume, so the download persists across restarts).

The GB10 image bakes the **pre-warmed FlashInfer JIT caches** (CUTLASS FP4 GEMM
sm121 modules + autotune tables + Triton cache) from the validated runs —
first boot does zero JIT compilation. On a cold system, first-use JIT with
the model resident can OOM (see [Known runtime notes](#known-runtime-notes)).

### RTX 5090 profile (Blackwell SM 12.0, 32 GB, x86-64)

A second, x86 image and tuned profile target a single RTX 5090:

```bash
hf download r0b0tlab/Ornith-1.5-35B-A3B-NVFP4-W4A16 \
  --local-dir ./models/ornith-15-35b-a3b-nvfp4-w4a16-B

docker compose -f docker-compose.rtx5090.yml up -d
# or bare:
docker run --rm --gpus all --ipc host -p 8000:8000 \
  -v $PWD/models/ornith-15-35b-a3b-nvfp4-w4a16-B:/models/ckpt:ro \
  -e MODEL_PATH=/models/ckpt \
  ghcr.io/r0b0tlab/ornith-15-35b-nvfp4-w4a16-sm121-sglang:rtx5090
```

Differences vs the GB10 profile ([`serve-profiles/rtx5090/serve.sh`](serve-profiles/rtx5090/serve.sh)):

| Knob | GB10 (121 GB unified) | RTX 5090 (32 GB) | Why |
|---|---|---|---|
| `--mem-fraction-static` | **0.75** | **0.92** | Spark claim serve; 32 GB discrete packs tighter |
| `--max-running-requests` | default | **4** | mamba state cache binds 32 GB |
| `--max-mamba-cache-size` | default | **24** | ~1.4 GB state → 4 × 32K contexts |
| context / KV | **262144** | 32768 × 4 reqs | Spark claim NIAH; 5090 stays 32k |
| MTP (EAGLE K=1) | on | on; `MTP=0` to disable | draft ~3.6 GB |

Memory budget ≈ 21.1 GB weights + 3.6 GB draft + 1.1 GB graphs + 1.4 GB
mamba + 2.5 GB KV ≈ 29.7 GB. First boot on a cold cache JIT-compiles the
sm_120 CUTLASS FP4 kernels (~10–20 min; runs in host RAM on a discrete-GPU
box — safe, unlike unified-memory parts). `docker-compose.rtx5090.yml`
mounts a persistent volume for the caches so subsequent boots are warm.
If capture OOMs, lower `MAX_RUNNING_REQUESTS`/`CONTEXT_LENGTH` first.

The `rtx5090` tag is built by CI ([workflow](.github/workflows/build-rtx5090-image.yml))
from [container/Dockerfile.rtx5090](container/Dockerfile.rtx5090), which
pip-installs the same pinned stack (sglang `5a7b26c63`, torch 2.13.0+cu130,
triton 3.7.1, flashinfer-python 0.6.17 + cubins) on x86 Ubuntu 24.04.
CI pushes require the package's **Manage Actions access → repository with
write** grant (repo settings → packages); until that is granted, the tag is
built and pushed cross-platform (arm64 builder + qemu) and CI remains a
verification path only.
**Note:** validated end-to-end on GB10/SM121; the 5090 profile is sized from
the measured memory ledger and the vendor's SM 12.0 support envelope, but was
not run on physical 5090 hardware before publication — report issues if
first-boot JIT behaves differently on sm_120 discrete parts.

### Serving without the container

SGLang `0.5.6.post3.dev9218+g5a7b26c63`, FlashInfer 0.6.17,
torch 2.13.0+cu130, Python 3.12:

```bash
python -m sglang.launch_server \
  --model-path r0b0tlab/Ornith-1.5-35B-A3B-NVFP4-W4A16 \
  --served-model-name Ornith-1.5-35B-A3B \
  --trust-remote-code \
  --attention-backend triton \
  --moe-runner-backend marlin \
  --context-length 32768 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.80 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 1 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 2 \
  --port 8000
```

Thinking-off protocol: `chat_template_kwargs: {"enable_thinking": false}`.

### Reproduce the evaluation

**Claim row** is SM12X-LLM-BENCH `full` with thinking on — `./scripts/bench-sm12x.sh`.

The 200-question harness below is the **historical think-off subset** (n=80/40/20).
It is not the native-quality claim.

```bash
cd eval
python run_quality_set.py --base-url http://127.0.0.1:8000 --run-id my-post
python rescore.py my-post
```

BF16-baseline and quantized rows from our runs are in `eval/data/` —
`rescore.py pre-bf16 post-nvfp4-B post-nvfp4-B-mtp` reproduces the table
above from the raw rows.

---

## Quantization recipe

| Item | Value |
|---|---|
| Tool | NVIDIA Model Optimizer, dev tree @ `913f5e224` |
| Format | NVFP4 W4A16 weight-only: E2M1 weights, block-16 E4M3 scales, scalar FP32 per-tensor scale |
| Scope | All `Linear` targets (routed + shared experts, dense attention, GDN linear-attention projections); lm_head/embeddings/routers/conv1d/MTP stay BF16 |
| Recipe | `w4_nvfp4` preset (max) + `default_disabled_quantizers` + `kv_fp8_cast` — [quantization/w4a16-nvfp4-std.yaml](quantization/w4a16-nvfp4-std.yaml) |
| Calibration | `cnn_nemotron_v2_mix`, 1024 samples × 1024 tokens, batch 16 |
| KV cache | FP8-E4M3 constant-amax cast (NVFP4 KV unavailable on aarch64) |
| MTP | 785 BF16 tensors re-attached verbatim (SHA-256 audited) — [quantization/reattach-mtp.py](quantization/reattach-mtp.py) |

Expert-only alternative (`nvfp4_four_over_six` MSE on routed experts) is
included for reference: [quantization/w4a16-expert-4o6.yaml](quantization/w4a16-expert-4o6.yaml).
It scored GSM8K 70.00% vs this recipe's 76.25% and decoded at 38.6 tok/s vs
71 tok/s on our hardware — the all-linear recipe wins on both axes here.

## Known runtime notes (all root-caused during bring-up)

1. **Warm the FlashInfer JIT cache before first boot on a cold system.**
   First use of the dense FP4 GEMMs JIT-compiles ~18 CUTLASS kernels;
   with the 23 GB model resident + autotune this OOM-killed the machine
   (kernel oom-killer, ninja exit 137). The container ships the cache
   pre-warmed; bare-metal users should complete the build with nothing
   else resident (e.g. `taskset -c 0-7 ninja -j2 -C <flashinfer cache>/fp4_gemm_cutlass_sm120`).
2. **Pin `--moe-runner-backend marlin`** — `auto` can resolve to
   `flashinfer_trtllm`, which raises `NotImplementedError` for NVFP4 MoE
   during CUDA-graph capture in this SGLang version.
3. **Use `--speculative-algorithm EAGLE` for the MTP head.**
   `FROZEN_KV_MTP` is not implemented for qwen3_5 MTP (no context hooks);
   EAGLE auto-remaps the draft architecture to `Qwen3_5ForCausalLMMTP` and
   loads the bundled BF16 head from the same checkpoint. It also requires
   `--speculative-num-steps`/`--speculative-eagle-topk` to be set
   explicitly (1/1 for the K=1 MTP profile).
4. **Never execute model-generated code inline.** Our first harness ran
   HumanEval submissions via `exec()` in-process; a submission containing
   an infinite loop hung the grader. The harness here runs them in an
   isolated subprocess with a 10 s timeout.

## Methodology notes

- **Scorer correctness matters more than it looks.** The initial in-run
  GSM8K scorer extracted the *first* number in the response; this model
  answers in markdown with intermediate results, so correct answers failed
  and the first campaign readings showed an artificial "collapse"
  (5/80 pre-quantization). The fixed scorer (last bolded number, then last
  number, with $/comma/trailing-dot normalization) is
  [`eval/answer_extract.py`](eval/answer_extract.py); all rows were rescored
  with it. Check a scorer against known-correct answers before trusting
  relative-loss numbers.
- Eval protocol: greedy (temperature 0), thinking disabled via chat
  template kwargs, 1024 max tokens (1536 for code families), one request
  at a time, radix cache on.
- The GSM8K family is a fixed 80-question subset with reference answers;
  HumanEval-style problems carry entry-point + tests; IFEval items use the
  four most robust instruction types.

## Repository layout

```
├── README.md
├── scripts/click-run.sh              # detect SM120/121, pull ckpt, compose up
├── scripts/bench-sm12x.sh            # SM12X-LLM-BENCH full, native think-on
├── docker-compose.yml                # GB10/SM121 claim defaults (262144)
├── docker-compose.rtx5090.yml        # RTX 50 / SM120 32 GB
├── docker-compose.rtx-pro-6000.yml   # RTX PRO 6000 / SM120 ~96 GB
├── container/                        # GB10 Dockerfile + serve.sh
│   └── Dockerfile.rtx5090            # x86 SM120 image
├── serve-profiles/
│   ├── rtx5090/serve.sh
│   └── rtx-pro-6000/serve.sh
├── eval/                             # historical 200-q think-off harness
├── results/sm12x-full-thinkon-20260824/   # claim-row evidence
├── quantization/
└── audits/
```

## Credits and attribution

- **Base model:** [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B)
  by Ornith AI — MIT license.
- **Quantization:** [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer)
  (dev @ `913f5e224`) — `w4_nvfp4`, `default_disabled_quantizers`,
  `kv_fp8_cast` recipe units.
- **Calibration data:** [abisee/cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail)
  and [nvidia/Nemotron-Post-Training-Dataset-v2](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2).
- **Inference engine:** [SGLang](https://github.com/sgl-project/sglang)
  `0.5.6.post3.dev9218+g5a7b26c63`, with [FlashInfer](https://github.com/flashinfer-ai/flashinfer)
  0.6.17 CUTLASS FP4 GEMMs and Marlin W4A16 MoE kernels; hosted on
  [PyTorch](https://pytorch.org) 2.13.0+cu130.
- **Infrastructure:** NVIDIA DGX Spark (GB10 / SM 12.1) validated;
  SM 12.0 (RTX 50 / PRO 6000) profiles are sized from that ledger.

## License

MIT — matching the base model. Calibration datasets remain under their own
licenses.
