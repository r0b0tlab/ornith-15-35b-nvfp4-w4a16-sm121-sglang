#!/usr/bin/env bash
set -euo pipefail

REPO=/srv/ssd/intel/llm/benchmarks/aider
RUN_ROOT=/srv/ssd/intel/llm/bench-results/ornith15-35b-a3b-r0b0t-aider-polyglot-20260824
PROFILE_DIR=/srv/ssd/intel/llm/aider-profiles/ornith15-35b-a3b-r0b0t
LOG="$RUN_ROOT/logs/full.log"
METRICS="$RUN_ROOT/telemetry/live-metrics-status.tsv"
GPU="$RUN_ROOT/telemetry/full-gpu.csv"
STOP="$RUN_ROOT/telemetry/full.stop"

test -f "$RUN_ROOT/FULL_RUN_APPROVED"
lock_sha256=$(sha256sum "$RUN_ROOT/protocol/protocol.lock.json" | awk '{print $1}')
grep -Fxq "protocol_lock_sha256=$lock_sha256" "$RUN_ROOT/FULL_RUN_APPROVED"
test ! -e "$RUN_ROOT/full/.full-complete"
test "$(find "$RUN_ROOT/polyglot-benchmark" -mindepth 4 -maxdepth 4 -type d | wc -l)" -eq 225
test "$(find "$RUN_ROOT/full" -name .aider.results.json 2>/dev/null | wc -l)" -eq 0
curl -fsS http://127.0.0.1:18753/v1/models \
  | jq -e '.data[0].id == "main" and .data[0].max_model_len == 262144' >/dev/null
docker inspect ornith15-aider-sglang-main >/dev/null
grep -F 'ornith15-35b-a3b-nvfp4-aider' /srv/ssd/intel/llm/state/model/active.env >/dev/null

cp "$RUN_ROOT/protocol/protocol.lock.json" "$RUN_ROOT/full/protocol.lock.json"
printf 'timestamp_utc\trequests_processing\tprompt_tps\tpredicted_tps\tprompt_tokens_total\ttokens_predicted_total\n' >"$METRICS"
printf 'timestamp_utc,index,name,memory_used_mib,utilization_gpu_pct,power_draw_w\n' >"$GPU"

sample_telemetry() {
  local previous_epoch previous_prompt previous_generation
  previous_epoch=$(date +%s)
  previous_prompt=0
  previous_generation=0
  while [[ ! -f "$STOP" ]]; do
    local ts epoch metrics running prompt_total generation_total elapsed prompt_tps generation_tps
    ts=$(date -u +%FT%TZ)
    epoch=$(date +%s)
    metrics=$(curl -fsS --max-time 3 http://127.0.0.1:18753/metrics 2>/dev/null || true)
    running=$(awk '/^sglang:num_running_reqs[{ ]/ {sum += $NF} END {print sum+0}' <<<"$metrics")
    prompt_total=$(awk '/^sglang:prompt_tokens_total[{ ]/ {sum += $NF} END {print sum+0}' <<<"$metrics")
    generation_total=$(awk '/^sglang:generation_tokens_total[{ ]/ {sum += $NF} END {print sum+0}' <<<"$metrics")
    elapsed=$((epoch - previous_epoch))
    if ((elapsed > 0)) && ((previous_prompt > 0 || previous_generation > 0)); then
      prompt_tps=$(awk -v cur="$prompt_total" -v prev="$previous_prompt" -v dt="$elapsed" 'BEGIN {printf "%.3f", (cur-prev)/dt}')
      generation_tps=$(awk -v cur="$generation_total" -v prev="$previous_generation" -v dt="$elapsed" 'BEGIN {printf "%.3f", (cur-prev)/dt}')
    else
      prompt_tps=0
      generation_tps=0
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$ts" "$running" "$prompt_tps" "$generation_tps" "$prompt_total" "$generation_total" >>"$METRICS"
    nvidia-smi --query-gpu=timestamp,index,name,memory.used,utilization.gpu,power.draw --format=csv,noheader,nounits >>"$GPU" || true
    previous_epoch=$epoch
    previous_prompt=$prompt_total
    previous_generation=$generation_total
    sleep 5
  done
}

rm -f "$STOP"
sample_telemetry &
telemetry_pid=$!
cleanup() {
  touch "$STOP"
  wait "$telemetry_pid" 2>/dev/null || true
}
trap cleanup EXIT

docker run --rm \
  --name aider-ornith15-r0b0t-full \
  --network host \
  --memory=48g --memory-swap=48g --cpus=32 \
  -v "$REPO:/aider" \
  -v "$RUN_ROOT:/benchmarks" \
  -v "$PROFILE_DIR/model-metadata.json:/aider/aider/resources/model-metadata.json:ro" \
  -e OPENAI_API_KEY=local \
  -e OPENAI_API_BASE=http://127.0.0.1:18753/v1 \
  -e AIDER_DOCKER=1 \
  -e AIDER_BENCHMARK_DIR=/benchmarks \
  -e PYTHONHASHSEED=0 \
  aider-benchmark \
  bash -lc './benchmark/benchmark.py /benchmarks/full \
    --model openai/main --edit-format whole --threads 12 --tries 2 \
    --num-ctx 262144 \
    --read-model-settings /benchmarks/protocol/model-settings.yml \
    --exercises-dir polyglot-benchmark' \
  >"$LOG" 2>&1

docker run --rm \
  -v "$REPO:/aider" \
  -v "$RUN_ROOT:/benchmarks" \
  -e AIDER_DOCKER=1 \
  -e AIDER_BENCHMARK_DIR=/benchmarks \
  aider-benchmark \
  bash -lc './benchmark/benchmark.py /benchmarks/full --stats' \
  >"$RUN_ROOT/logs/full-stats.log" 2>&1

cleanup
trap - EXIT
python3 "$RUN_ROOT/scripts/audit-full.py" >"$RUN_ROOT/logs/final-audit.log"
jq -e '.status == "valid-complete"' "$RUN_ROOT/protocol/final-audit.json" >/dev/null
touch "$RUN_ROOT/full/.full-complete"
