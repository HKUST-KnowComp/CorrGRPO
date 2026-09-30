# Build from the repository root: docker build -t corrgrpo:latest .
# vLLM's CUDA/PyTorch binaries stay together; do not upgrade torch independently.
ARG BASE_IMAGE=vllm/vllm-openai:v0.19.1
FROM ${BASE_IMAGE} AS runtime
USER root
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    HF_HOME=/workspace/.cache/huggingface \
    PYTHONPATH=/workspace \
    PATH=/usr/local/bin:${PATH}
RUN apt-get update && apt-get install -y --no-install-recommends \
      bash build-essential git ca-certificates python3.12-dev docker.io \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf "$(command -v python3)" /usr/local/bin/python
ENV USER=corrgrpo LOGNAME=corrgrpo HOME=/tmp
WORKDIR /workspace
COPY requirements.txt docker/constraints.txt /tmp/corrgrpo-deps/
RUN python -m pip install --no-cache-dir -c /tmp/corrgrpo-deps/constraints.txt \
      -r /tmp/corrgrpo-deps/requirements.txt
# Keep task dependencies separate from VERL's development requirements.
COPY docker/requirements.txt /tmp/corrgrpo-task-requirements.txt
RUN python -m pip install --no-cache-dir -c /tmp/corrgrpo-deps/constraints.txt \
      -r /tmp/corrgrpo-task-requirements.txt
# Build against the image's exact torch/CUDA ABI when no matching wheel exists.
ARG MAX_JOBS=2
RUN MAX_JOBS=${MAX_JOBS} python -m pip install --no-cache-dir --no-build-isolation \
      --no-deps flash-attn==2.8.3.post1
COPY . /workspace
RUN python -m pip install --no-cache-dir --no-deps -e . \
    && python -m pip check
ENTRYPOINT []
CMD ["bash"]

# Optional external benchmark adapters; the three default pipelines use runtime.
FROM runtime AS benchmarks
COPY docker/requirements-benchmarks.txt /tmp/corrgrpo-benchmarks.txt
RUN python -m pip install --no-cache-dir -c /tmp/corrgrpo-deps/constraints.txt \
      -r /tmp/corrgrpo-benchmarks.txt \
    && python -m pip check

# Keep the default build focused on the three training tasks.
FROM runtime AS final
