#!/usr/bin/env bash
# RTX 5090 (Blackwell SM 12.0, 32 GB GDDR7, x86-64) serve profile for
# Ornith-1.5-35B-A3B-NVFP4-W4A16.
#
# Memory budget on one 32 GB card (measured on the validated GB10 runs,
# scaled to a dedicated-GPU part):
#   quantized target weights   ~21.1 GB
#   BF16 MTP draft (EAGLE)      ~3.6 GB   (disable with MTP=0 to reclaim)
#   CUDA graphs (verify+draft)  ~1.1 GB
#   mamba state cache (24 slots)~1.4 GB   (hybrid GDN layers own no KV)
#   FP8 KV for 4x32768 ctx      ~2.5 GB
#   => ~29.7 GB @ mem-fraction 0.92; lower MAX_RUNNING_REQUESTS/CONTEXT_LENGTH
#      if you OOM during graph capture.
#
# First boot on a cold cache JIT-compiles the FlashInfer CUTLASS FP4 GEMMs
# for sm_120 (10-20 min, runs in system RAM on a discrete-GPU host — safe,
# unlike unified-memory parts). Mount /home/sglang/.cache to persist it.
set -euo pipefail
MODEL_PATH="${MODEL_PATH:-r0b0tlab/Ornith-1.5-35B-A3B-NVFP4-W4A16}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.92}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-4}"
MAX_MAMBA_CACHE_SIZE="${MAX_MAMBA_CACHE_SIZE:-24}"
PORT="${PORT:-8000}"

MTP_ARGS=(
  --speculative-algorithm EAGLE
  --speculative-num-steps 1
  --speculative-eagle-topk 1
  --speculative-num-draft-tokens 2
)
if [ "${MTP:-1}" = "0" ]; then MTP_ARGS=(); fi

exec python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --served-model-name Ornith-1.5-35B-A3B \
  --trust-remote-code \
  --attention-backend triton \
  --moe-runner-backend marlin \
  --tool-call-parser qwen3_coder \
  --context-length "$CONTEXT_LENGTH" \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static "$MEM_FRACTION_STATIC" \
  --max-running-requests "$MAX_RUNNING_REQUESTS" \
  --max-mamba-cache-size "$MAX_MAMBA_CACHE_SIZE" \
  "${MTP_ARGS[@]}" \
  --host 0.0.0.0 --port "$PORT" \
  ${EXTRA_ARGS:-}
