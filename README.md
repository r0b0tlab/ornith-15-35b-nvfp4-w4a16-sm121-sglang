# Ornith-1.5-35B-A3B NVFP4-W4A16 (SGLang, DGX Spark GB10/SM121)

NVFP4 **W4A16 weight-only** quantization of
[`ornith-ai/Ornith-1.5-35B-A3B`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B)
(MIT, revision `e4dfb35a93d4b6822a811a7676f3488514abe7e2`), produced with
NVIDIA Model Optimizer (dev `913f5e224`) using the official `w4_nvfp4`
recipe preset, with the bundled MTP head kept in BF16 and served via SGLang
EAGLE/nextn speculative decoding.

This repo contains the **reproduction package**: the ready-to-run container,
the 200-question evaluation harness with the exact question set, scoring
code, raw eval rows from all four runs, checkpoint audit tooling, and the
quantization recipes.

- Checkpoint: [huggingface.co/r0b0tlab/Ornith-1.5-35B-A3B-NVFP4-W4A16](https://huggingface.co/r0b0tlab/Ornith-1.5-35B-A3B-NVFP4-W4A16)
- Container: `ghcr.io/r0b0tlab/ornith-15-35b-nvfp4-w4a16-sm121-sglang:latest`
- Hardware validated: NVIDIA DGX Spark (GB10, SM 12.1, 121 GB unified memory), aarch64

Independent community quantization by r0b0tlab. Not affiliated with or
endorsed by Ornith AI or NVIDIA.

---

## Results

### 200-question suite (greedy, thinking disabled, fixed scorer)

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

---

## Quick start (container)

Requirements: NVIDIA GPU with Blackwell-class FP4 support, Docker with
nvidia-container-toolkit, ~25 GB disk for the checkpoint.

```bash
# 1) Get the checkpoint (either from HF, or use your own local copy)
hf download r0b0tlab/Ornith-1.5-35B-A3B-NVFP4-W4A16 \
  --local-dir ./models/ornith-15-35b-a3b-nvfp4-w4a16-B

# 2) Launch
docker compose up -d

# 3) Wait for health (first boot ~4-5 min: load 152 s + CUDA-graph capture)
curl http://127.0.0.1:8000/health

# 4) Smoke test
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Ornith-1.5-35B-A3B",
    "messages": [{"role": "user", "content": "What is 19*23? Answer with just the number."}],
    "temperature": 0, "max_tokens": 32,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
# → "437"
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

The image bakes the **pre-warmed FlashInfer JIT caches** (CUTLASS FP4 GEMM
sm121 modules + autotune tables + Triton cache) from the validated runs —
first boot does zero JIT compilation. On a cold system, first-use JIT with
the model resident can OOM (see [Known runtime notes](#known-runtime-notes)).

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

```bash
cd eval
python run_quality_set.py --base-url http://127.0.0.1:8000 --run-id my-post
python rescore.py my-post          # fixed-scorer summary
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
├── docker-compose.yml        # one-command launch
├── container/                # Dockerfile, serve.sh entrypoint, build script
├── eval/
│   ├── run_quality_set.py    # run the 200-question suite (resumable)
│   ├── rescore.py            # fixed-scorer summaries from raw rows
│   ├── answer_extract.py     # GSM8K answer extraction (the fixed scorer)
│   └── data/                 # quality-200.jsonl + raw rows for 4 runs
├── quantization/
│   ├── w4a16-nvfp4-std.yaml  # the shipped recipe (candidate B)
│   ├── w4a16-expert-4o6.yaml # experts-only 4/6 alternative (candidate A)
│   ├── quant-cand-custom.py  # CPU-first quantize driver
│   └── reattach-mtp.py       # BF16 MTP re-attachment
└── audits/
    ├── audit_checkpoint.py   # scale pairing / MTP hashes / key closure
    ├── cosine_probe.py       # NVFP4 dequant cosine vs BF16 source
    └── results/              # audit JSON outputs
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
- **Infrastructure:** NVIDIA DGX Spark (GB10 / SM 12.1).

## License

MIT — matching the base model. Calibration datasets remain under their own
licenses.
