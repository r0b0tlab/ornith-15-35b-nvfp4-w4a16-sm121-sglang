#!/usr/bin/env bash
# Run the SM12X-LLM-BENCH `full` claim profile (native thinking on)
# against a local Ornith serve. Does not start the server.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
MODEL="${MODEL:-Ornith-1.5-35B-A3B}"
OUT="${OUT:-$ROOT/out/sm12x-full}"
BENCH_DIR="${SM12X_BENCH_DIR:-$ROOT/.deps/SM12X-LLM-BENCH}"

if ! curl -sf -m 5 "${BASE_URL%/v1}/health" >/dev/null 2>&1 \
   && ! curl -sf -m 5 "$BASE_URL/models" >/dev/null 2>&1; then
  echo "no live endpoint at $BASE_URL — run ./scripts/click-run.sh first" >&2
  exit 2
fi

if [ ! -d "$BENCH_DIR/.git" ]; then
  mkdir -p "$(dirname "$BENCH_DIR")"
  git clone --depth 1 https://github.com/SM12X-SOCOM/SM12X-LLM-BENCH.git "$BENCH_DIR"
fi

mkdir -p "$OUT"
printf '%s\n' '{"chat_template_kwargs":{"enable_thinking":true,"thinking":true}}' > "$OUT/extra-body-thinkon.json"

export BASE_URL MODEL
export OUT
export SM12X_NO_TMUX="${SM12X_NO_TMUX:-0}"
cd "$BENCH_DIR"
exec env BASE_URL="$BASE_URL" MODEL="$MODEL" OUT="$OUT" \
  ./scripts/click_run.sh --profile full --extra-body /out/extra-body-thinkon.json --concurrency "${CONCURRENCY:-2}"
