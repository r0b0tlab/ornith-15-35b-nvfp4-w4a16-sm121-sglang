#!/usr/bin/env bash
# RTX PRO 6000 (Blackwell SM 12.0, 96 GB, x86-64) serve profile.
# Same x86 image as the 5090 profile; more KV / running requests.
# Not physically signed off on this checkpoint — sized from the GB10 ledger.
set -euo pipefail
MODEL_PATH="${MODEL_PATH:-r0b0tlab/Ornith-1.5-35B-A3B-NVFP4-W4A16}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-131072}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.90}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-8}"
MAX_MAMBA_CACHE_SIZE="${MAX_MAMBA_CACHE_SIZE:-64}"
PORT="${PORT:-8000}"

MTP_ARGS=(
  --speculative-algorithm EAGLE
  --speculative-num-steps 1
  --speculative-eagle-topk 1
  --speculative-num-draft-tokens 2
)
if [ "${MTP:-1}" = "0" ]; then MTP_ARGS=(); fi

METRICS_ARGS=()
if [ "${ENABLE_METRICS:-1}" != "0" ]; then METRICS_ARGS=(--enable-metrics); fi

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
  "${METRICS_ARGS[@]}" \
  --host 0.0.0.0 --port "$PORT" \
  ${EXTRA_ARGS:-}
