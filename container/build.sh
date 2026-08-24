#!/usr/bin/env bash
# Materialize the GB10 image build context (pinned venv + JIT caches) and push.
# Run on the aarch64 GB10 builder that holds the validated venv/caches:
#   bash build.sh
set -euo pipefail
cd "$(dirname "$0")"

echo "== staging venv (excludes + fix symlinks) =="
mkdir -p context/venv context/caches
rsync -a --delete \
  --exclude '__pycache__' --exclude '*.pyc' \
  --exclude 'sglang/*.egg-info' \
  ~/.venvs/sglang-main/ context/venv/
# point the venv python at the image's interpreter
ln -sf /usr/bin/python3.12 context/venv/bin/python

echo "== staging JIT caches =="
rsync -a ~/.cache/sglang/.cache/flashinfer/ context/caches/flashinfer/
rsync -a ~/.cache/sglang/flashinfer/autotune/ context/caches/flashinfer-autotune/
rsync -a ~/.triton/ context/caches/triton/

cp Dockerfile serve.sh context/
echo "== context size =="
du -sh context

echo "== building image =="
docker build -t ghcr.io/r0b0tlab/ornith-15-35b-nvfp4-w4a16-sm121-sglang:latest context/

echo "== pushing =="
docker push ghcr.io/r0b0tlab/ornith-15-35b-nvfp4-w4a16-sm121-sglang:latest
echo "DONE"
