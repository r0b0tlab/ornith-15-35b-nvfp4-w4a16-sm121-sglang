#!/usr/bin/env bash
# Launch SGLang for Ornith-1.5-35B-A3B-NVFP4-W4A16 with the validated profile.
#
# Env overrides:
#   MODEL_PATH          local dir or HF repo id (default: the HF repo below)
#   CONTEXT_LENGTH      default 32768 (compose claim profile sets 262144)
#   MEM_FRACTION_STATIC default 0.75
#   PORT                default 8000
#   MTP                 set 0 to disable EAGLE
#   ENABLE_METRICS      default 1
#   EXTRA_ARGS          additional SGLang flags (quoted string)
set -euo pipefail
MODEL_PATH="${MODEL_PATH:-r0b0tlab/Ornith-1.5-35B-A3B-NVFP4-W4A16}"
MTP_ARGS="--speculative-algorithm EAGLE --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2"
if [ "${MTP:-1}" = "0" ]; then MTP_ARGS=""; fi
METRICS_ARGS="--enable-metrics"
if [ "${ENABLE_METRICS:-1}" = "0" ]; then METRICS_ARGS=""; fi
exec python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --served-model-name Ornith-1.5-35B-A3B \
  --trust-remote-code \
  --attention-backend triton \
  --moe-runner-backend marlin \
  --tool-call-parser qwen3_coder \
  --context-length "${CONTEXT_LENGTH:-32768}" \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static "${MEM_FRACTION_STATIC:-0.75}" \
  ${MTP_ARGS} \
  ${METRICS_ARGS} \
  --host 0.0.0.0 --port "${PORT:-8000}" \
  ${EXTRA_ARGS:-}
