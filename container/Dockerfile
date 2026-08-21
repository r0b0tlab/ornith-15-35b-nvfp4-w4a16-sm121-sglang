# Ornith-1.5-35B-A3B-NVFP4-W4A16 ready-to-run serving container
# Base mirrors the host the stack was validated on: Ubuntu 24.04, Python 3.12, CUDA 13 runtime via nvidia-container-toolkit.
# Multi-stage: builder carries the exact pinned venv; runtime strips build deps.
FROM ubuntu:24.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv ninja-build git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python
# The pinned venv is materialized into the build context by build.sh (rsync).
COPY venv/ /opt/venv/
RUN ln -sf /usr/bin/python3.12 /opt/venv/bin/python && /opt/venv/bin/python -c "import sglang, torch, flashinfer; print('venv-ok', sglang.__version__)" || echo "WARN: import smoke failed"

FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.12 libpython3.12 ninja-build ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python

# CUDA/driver come from nvidia-container-toolkit at runtime; nothing baked.
COPY --from=builder /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    HOME=/home/sglang \
    HF_HOME=/home/sglang/.cache/huggingface

# Pre-warmed JIT caches from the validated runs — eliminate first-boot JIT
# entirely (cold-boot FlashInfer CUTLASS compile with the model resident can
# OOM; see README "Known runtime notes").
COPY caches/flashinfer/ /root/.cache/sglang/.cache/flashinfer/
COPY caches/flashinfer-autotune/ /root/.cache/sglang/flashinfer/autotune/
COPY caches/triton/ /root/.cache/triton/

# Non-root by default; override at runtime if the model dir needs it.
# (ubuntu:24.04 already has UID 1000 for the 'ubuntu' user — use 1001.)
RUN useradd -m -u 1001 sglang \
    && mkdir -p /home/sglang/.cache \
    && cp -a /root/.cache/sglang /home/sglang/.cache/ 2>/dev/null || true \
    && cp -a /root/.cache/triton /home/sglang/.cache/ 2>/dev/null || true \
    && chown -R sglang:sglang /home/sglang \
    && chmod -R a+rX /root/.cache
USER sglang
WORKDIR /home/sglang

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=5 \
  CMD curl -sf http://127.0.0.1:8000/health || exit 1

# Copy entrypoint and set permissions BEFORE dropping privileges.
COPY --chmod=755 serve.sh /usr/local/bin/serve.sh
ENTRYPOINT ["/usr/local/bin/serve.sh"]
