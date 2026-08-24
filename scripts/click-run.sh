#!/usr/bin/env bash
# Click-run Ornith-1.5-35B NVFP4-W4A16 on SM12X:
#   SM 12.1  DGX Spark / GB10
#   SM 12.0  RTX 50-class (32 GB) and RTX PRO 6000 (96 GB)
#
# Usage:
#   ./scripts/click-run.sh              # detect GPU, pull ckpt, compose up, wait health
#   ./scripts/click-run.sh --profile spark|pro6000|rtx5090
#   CONTEXT_LENGTH=32768 ./scripts/click-run.sh
#   ./scripts/click-run.sh --no-wait
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PROFILE="${PROFILE:-auto}"
WAIT=1
while [ $# -gt 0 ]; do
  case "$1" in
    --profile) PROFILE="${2:?}"; shift 2 ;;
    --no-wait) WAIT=0; shift ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

need() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing dependency: $1" >&2
    exit 2
  fi
}

need docker
if ! docker info >/dev/null 2>&1; then
  echo "docker daemon is not reachable" >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found — this image needs an SM12.0 or SM12.1 NVIDIA GPU" >&2
  exit 2
fi

GPU_LINE="$(nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader,nounits | head -1)"
GPU_NAME="$(printf '%s\n' "$GPU_LINE" | awk -F', ' '{print $1}')"
GPU_CC="$(printf '%s\n' "$GPU_LINE" | awk -F', ' '{print $2}')"
GPU_MEM_MIB="$(printf '%s\n' "$GPU_LINE" | awk -F', ' '{print $NF}')"
GPU_MEM_MIB="${GPU_MEM_MIB%.*}"
echo "detected: $GPU_NAME  cc=$GPU_CC  mem=${GPU_MEM_MIB} MiB"

if [ "$PROFILE" = "auto" ]; then
  case "$GPU_CC" in
    12.1|12.1.0) PROFILE=spark ;;
    12.0|12.0.0)
      if [ "${GPU_MEM_MIB:-0}" -ge 80000 ]; then
        PROFILE=pro6000
      else
        PROFILE=rtx5090
      fi
      ;;
    *)
      echo "this checkpoint is for SM12X (compute 12.0 / 12.1). got cc=$GPU_CC" >&2
      exit 2
      ;;
  esac
fi

MODEL_DIR="${MODEL_DIR:-$ROOT/models/ornith-15-35b-a3b-nvfp4-w4a16-B}"
mkdir -p "$(dirname "$MODEL_DIR")"

case "$PROFILE" in
  spark)
    COMPOSE=(-f docker-compose.yml)
    DEFAULT_CTX=262144
    DEFAULT_MEM=0.75
    NOTE="GB10/SM121 claim profile (262144 ctx, MTP K=1). Validated."
    ;;
  pro6000)
    COMPOSE=(-f docker-compose.rtx-pro-6000.yml)
    DEFAULT_CTX=131072
    DEFAULT_MEM=0.90
    NOTE="RTX PRO 6000 / SM120 ~96 GB. Sized from the memory ledger — not physically signed off on this checkpoint."
    ;;
  rtx5090)
    COMPOSE=(-f docker-compose.rtx5090.yml)
    DEFAULT_CTX=32768
    DEFAULT_MEM=0.92
    NOTE="RTX 50-class / SM120 32 GB. Sized from the memory ledger — not physically signed off on this checkpoint."
    ;;
  *)
    echo "unknown profile: $PROFILE (spark|pro6000|rtx5090)" >&2
    exit 2
    ;;
esac

export CONTEXT_LENGTH="${CONTEXT_LENGTH:-$DEFAULT_CTX}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-$DEFAULT_MEM}"
export MODEL_DIR
echo "profile=$PROFILE  context=$CONTEXT_LENGTH  mem_fraction=$MEM_FRACTION_STATIC"
echo "$NOTE"

if [ ! -f "$MODEL_DIR/config.json" ]; then
  echo "checkpoint missing at $MODEL_DIR — downloading r0b0tlab/Ornith-1.5-35B-A3B-NVFP4-W4A16"
  if command -v hf >/dev/null 2>&1; then
    hf download r0b0tlab/Ornith-1.5-35B-A3B-NVFP4-W4A16 --local-dir "$MODEL_DIR"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download r0b0tlab/Ornith-1.5-35B-A3B-NVFP4-W4A16 --local-dir "$MODEL_DIR"
  else
    echo "install Hugging Face CLI: pip install -U 'huggingface_hub[cli]'" >&2
    exit 2
  fi
fi

docker compose "${COMPOSE[@]}" up -d

if [ "$WAIT" = 1 ]; then
  echo "waiting for /health (first boot: load + CUDA graphs; 5090 cold JIT can take 10–20 min)"
  ok=0
  for i in $(seq 1 180); do
    if curl -sf -m 3 http://127.0.0.1:8000/health >/dev/null 2>&1; then
      ok=1
      break
    fi
    sleep 5
  done
  if [ "$ok" != 1 ]; then
    echo "health check did not pass. logs:" >&2
    docker compose "${COMPOSE[@]}" logs --tail 80 >&2
    exit 1
  fi
  echo "healthy. smoke: curl http://127.0.0.1:8000/v1/models"
  echo
  echo "native think-on claim bench (separate clone):"
  echo "  git clone https://github.com/SM12X-SOCOM/SM12X-LLM-BENCH.git"
  echo "  cd SM12X-LLM-BENCH && ./scripts/click_run.sh \\"
  echo "    --profile full --extra-body extra-body-thinkon.json \\"
  echo "    # with BASE_URL=http://127.0.0.1:8000/v1 MODEL=Ornith-1.5-35B-A3B"
  echo "or: ./scripts/bench-sm12x.sh"
fi
