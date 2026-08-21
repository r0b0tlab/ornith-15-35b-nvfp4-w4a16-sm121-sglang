#!/usr/bin/env bash
# Launch SGLang for Ornith-1.5-35B-A3B-NVFP4-W4A16 with the validated profile.
#
# Env overrides:
#   MODEL_PATH          local dir or HF repo id (default: the HF repo below)
#   CONTEXT_LENGTH      default 32768
#   MEM_FRACTION_STATIC default 0.80
#   PORT                default 8000
#   EXTRA_ARGS          additional SGLang flags (quoted string)
set -euo pipefail
MODEL_PATH="${MODEL_PATH:-r0b0tlab/Ornith-1.5-35B-A3B-NVFP4-W4A16}"
exec python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --served-model-name Ornith-1.5-35B-A3B \
  --trust-remote-code \
  --attention-backend triton \
  --moe-runner-backend marlin \
  --tool-call-parser qwen3_coder \
  --context-length "${CONTEXT_LENGTH:-32768}" \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static "${MEM_FRACTION_STATIC:-0.80}" \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 1 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 2 \
  --host 0.0.0.0 --port "${PORT:-8000}" \
  ${EXTRA_ARGS:-}
