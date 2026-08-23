#!/usr/bin/env bash
set -euo pipefail

REPO=/srv/ssd/intel/llm/benchmarks/aider
RUN_ROOT=/srv/ssd/intel/llm/bench-results/ornith15-35b-a3b-nvfp4-aider-polyglot-20260823
LOG="$RUN_ROOT/logs/full.log"
METRICS="$RUN_ROOT/telemetry/live-metrics-status.tsv"
GPU="$RUN_ROOT/telemetry/full-gpu.csv"
STOP="$RUN_ROOT/telemetry/full.stop"

test ! -e "$RUN_ROOT/full/.full-complete"
test "$(find "$RUN_ROOT/full" -mindepth 4 -maxdepth 4 -type d | wc -l)" -eq 225
model_json=$(curl -fsS --max-time 5 http://127.0.0.1:18753/v1/models)
jq -e '.data[0].id == "main" and .data[0].max_model_len == 98304' <<<"$model_json" >/dev/null
docker inspect ornith15-aider-sglang-main >/dev/null
grep -F 'ornith15-35b-a3b-nvfp4-aider' /srv/ssd/intel/llm/state/model/active.env >/dev/null

printf 'timestamp_utc\trequests_processing\trequests_queued\tprompt_tokens_total\tgeneration_tokens_total\n' >"$METRICS"
printf 'timestamp_utc,index,name,memory_used_mib,utilization_gpu_pct,power_draw_w\n' >"$GPU"

sample_telemetry() {
  while [[ ! -f "$STOP" ]]; do
    ts=$(date -u +%FT%TZ)
    metrics=$(curl -fsS --max-time 3 http://127.0.0.1:18753/metrics 2>/dev/null || true)
    running=$(awk '/^sglang:num_running_reqs[{ ]/ {sum += $NF} END {print sum+0}' <<<"$metrics")
    queued=$(awk '/^sglang:num_queue_reqs[{ ]/ {sum += $NF} END {print sum+0}' <<<"$metrics")
    prompt_total=$(awk '/^sglang:prompt_tokens_total[{ ]/ {sum += $NF} END {print sum+0}' <<<"$metrics")
    generation_total=$(awk '/^sglang:generation_tokens_total[{ ]/ {sum += $NF} END {print sum+0}' <<<"$metrics")
    printf '%s\t%s\t%s\t%s\t%s\n' "$ts" "$running" "$queued" "$prompt_total" "$generation_total" >>"$METRICS"
    nvidia-smi --query-gpu=timestamp,index,name,memory.used,utilization.gpu,power.draw --format=csv,noheader,nounits >>"$GPU" || true
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
  --name aider-ornith15-nvfp4-full \
  --network host \
  --memory=48g --memory-swap=48g --cpus=32 \
  -v "$REPO:/aider" \
  -v "$RUN_ROOT:/benchmarks" \
  -e OPENAI_API_KEY=local \
  -e OPENAI_API_BASE=http://127.0.0.1:18753/v1 \
  -e AIDER_DOCKER=1 \
  -e AIDER_BENCHMARK_DIR=/benchmarks \
  -e PYTHONHASHSEED=0 \
  aider-benchmark \
  bash -lc './benchmark/benchmark.py /benchmarks/full --model openai/main --edit-format whole --threads 32 --tries 2 --read-model-settings /benchmarks/protocol/model-settings.yml --exercises-dir polyglot-benchmark' \
  >"$LOG" 2>&1

docker run --rm \
  -v "$REPO:/aider" \
  -v "$RUN_ROOT:/benchmarks" \
  -e AIDER_DOCKER=1 \
  -e AIDER_BENCHMARK_DIR=/benchmarks \
  aider-benchmark \
  bash -lc './benchmark/benchmark.py /benchmarks/full --stats' \
  >"$RUN_ROOT/logs/full-stats.log" 2>&1

touch "$RUN_ROOT/full/.full-complete"
