# Docker setup

Run all commands below from the CorrGRPO repository root, which contains `verl/`,
`corrgrpo-src/`, and `Dockerfile`. Distribute the complete repository and its bundled
benchmark data, not only `corrgrpo-src/`. No original author home directory, Conda
environment, or host Python packages are needed.

The training image targets Linux x86-64 with NVIDIA GPUs. Install Docker Engine
and NVIDIA Container Toolkit on the host first. The public build recipe uses the
[official vLLM Docker image](https://docs.vllm.ai/en/latest/deployment/docker/)
with vLLM 0.19.1, PyTorch 2.10.0, and the task dependencies listed in this directory.
The host driver must support the base image's CUDA runtime. Build and CPU checks
do not require a GPU. Full training and model generation do require GPUs.

## Use the verified image

The tested image is published at [huwenbin2024/corrgrpo on Docker Hub](https://hub.docker.com/r/huwenbin2024/corrgrpo). Pull it and create the local aliases used in this guide:

```bash
docker pull huwenbin2024/corrgrpo:latest
docker tag huwenbin2024/corrgrpo:latest corrgrpo:latest
docker tag huwenbin2024/corrgrpo:latest corrgrpo:matched
```

The `latest` and `matched` registry tags contain the same Linux x86-64 image, without model weights.

Published on 2026-09-30. Verified manifest digest: `sha256:036b21b9a93b765f481a6106bbe26c54050354de63f40c98fbea3f53fe0431ea`. Anonymous access and all 22 layer digests were checked against the tested source image.

The local image `corrgrpo:matched` (also tagged `corrgrpo:latest`) was validated
on GPU2 on 2026-09-29. It contains Python, dependencies, project code, and bundled
benchmark data. The image is approximately 6.3 GB before export compression.
See [RETEST.md](RETEST.md) for the latest script tests and [VALIDATION.md](VALIDATION.md) for the initial snapshot checks.

Export it on the build machine, transfer the archive and repository, then load it
on another Linux x86-64 machine:

```bash
# Build machine:
docker save corrgrpo:matched | gzip -1 > corrgrpo-image.tar.gz
# Receiving machine:
gunzip -c corrgrpo-image.tar.gz | docker load
docker tag corrgrpo:matched corrgrpo:latest
docker run --rm --network none --user "$(id -u):$(id -g)" \
  corrgrpo:matched python docker/smoke_test.py
```

Use the model layout and task commands below. Loading this snapshot avoids
rebuilding Python packages or accessing the original virtual environment.

## Build from public dependencies and check

```bash
docker build -t corrgrpo:latest .
mkdir -p models outputs .cache/huggingface
docker run --rm --network none \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  corrgrpo:latest python docker/smoke_test.py
```

The check loads real training configurations and datasets and runs CPU reward tests;
it does not start Ray workers or a training job. For a small CUDA forward/backward
check, add `--gpus '"device=0"'` before the image name and `--gpu` after the script.

The first image build downloads dependencies and may compile FlashAttention without
a GPU. `--build-arg MAX_JOBS=2` limits compilation concurrency. A registry mirror can
be supplied with `--build-arg BASE_IMAGE=your-registry/vllm-openai:v0.19.1`.
A Docker daemon proxy or registry failure must be fixed on the build host.
This recipe passed dependency resolution, but its full build was not completed
on GPU2 because the public base-image download timed out. The snapshot above is
the image that was actually built and tested.

## Put model weights in a relative directory

For example, place the complete Hugging Face checkpoint in
`models/Qwen2.5-3B-Instruct/`. It must include config, tokenizer, and weight files.
To download it using the image's Hugging Face CLI:

```bash
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$PWD/models:/workspace/models" \
  -v "$PWD/.cache:/workspace/.cache" \
  corrgrpo:latest hf download Qwen/Qwen2.5-3B-Instruct \
  --local-dir models/Qwen2.5-3B-Instruct
```

Weights and outputs are excluded from the image. Share the image using your own
registry, or use `docker save corrgrpo:latest | gzip > corrgrpo-image.tar.gz` and
`gunzip -c corrgrpo-image.tar.gz | docker load` on the receiving machine.
The published image can also be pulled directly as `huwenbin2024/corrgrpo:latest`.

## Train, convert, and evaluate

Choose one task command below. The defaults use one visible GPU. Use an explicit
`MODEL_PATH` to avoid any dependency on a Hugging Face cache layout.

```bash
docker run --rm --gpus '"device=0"' --shm-size 8g \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$PWD:/workspace" \
  -e MODEL=qwen25-3b -e MODEL_PATH=models/Qwen2.5-3B-Instruct \
  -e CUDA_VISIBLE_DEVICES=0 \
  corrgrpo:latest bash corrgrpo-src/toolcall_rl/run.sh
```

Replace the final script with either:

```bash
bash corrgrpo-src/code_rl/run.sh
bash corrgrpo-src/agent_security_rl/run.sh
```

For code tasks, choose a suitable coder checkpoint, e.g.
`MODEL=qwen25-coder-7b MODEL_PATH=models/Qwen2.5-Coder-7B-Instruct`.
For multiple GPUs, expose them with Docker's `--gpus` and set container-local CUDA
indices, for example `--gpus '"device=2,3"' -e CUDA_VISIBLE_DEVICES=0,1`.
Choose batch sizes and model size for the available GPU memory.

Add `-e DRY_RUN=1` to print the full pipeline without loading a model. Hyperparameters
can be overridden with environment variables, or Hydra arguments after `run.sh`.
Results are written to `outputs/<task>/<run-name>/` on the host. Set `RUN_NAME` or
`RUN_DIR=outputs/my-run` to use a fresh output location.

All built-in paths are derived from the script location. User-supplied relative
model/data/output paths are resolved from the invocation directory before scripts
change their working directory. Absolute paths remain accepted as overrides.

## Other benchmarks

The default image includes the local RLLA, LeetCode, and AgentDojo flows, InjecAgent's
local adapter, and code scoring dependencies. For API-Bank or ASB adapters:

```bash
docker build --target benchmarks -t corrgrpo:benchmarks .
```

Use `corrgrpo:benchmarks` in the run command and select `BENCHMARK=api-bank`,
`BENCHMARK=asb`, or another choice documented in `corrgrpo-src/README.md`.
Optional API providers still require their own credentials if explicitly selected.
Local workflows do not need Cohere or Google GenAI SDKs.

HumanEval, MBPP, and LiveCodeBench scoring starts a separate, restricted Docker
container. To dispatch it from inside the training container, add these options
before the image name:

```bash
-v /var/run/docker.sock:/var/run/docker.sock \
--group-add "$(stat -c '%g' /var/run/docker.sock)" \
-e HOST_REPO_ROOT="$PWD" -e EVAL_IMAGE=corrgrpo:latest
```

This socket grants access to the host Docker daemon; only the trusted launcher gets
it. The generated-code sandbox has no socket, no network, read-only benchmark
files, no Linux capabilities, and bounded CPU/memory. Its only persistent writable
mount is the output directory. For output directories outside the mounted checkout,
set `HOST_OUTPUT_DIR` to the matching path on the Docker host.

No sandbox mounts the host's site-packages or virtual environment.

## Package an already-tested environment

The self-contained `corrgrpo:matched` image can be built from an existing Linux
x86-64 virtual environment. The image embeds its own Python interpreter and
packages; it never mounts the original environment at runtime. The helper uses
`uv` to add any missing code-scoring dependencies to the copy only.

```bash
source .venv/bin/activate
python docker/package_environment.py --output ../corrgrpo-docker-context
docker build -t corrgrpo:matched ../corrgrpo-docker-context
docker run --rm --network none corrgrpo:matched python docker/smoke_test.py
docker save corrgrpo:matched | gzip -1 > corrgrpo-image.tar.gz
# On the receiving machine:
gunzip -c corrgrpo-image.tar.gz | docker load
```

Use `corrgrpo:matched` in the run examples above after loading it. The snapshot
records package versions in `/opt/corrgrpo-environment.txt`. Model weights are
still supplied through `models/`. The optional benchmark image target is separate.

## GPU2 test note

The 2026-09-29 script retest fixed the default logger list in all three launchers.
The image tags now contain that fix and the CPU smoke test includes 23 checks.
During this retest, the host Docker default bridge was unavailable. Offline checks
used `--network none`; no host network configuration was changed. Training with
local model weights can also use this mode for a single-host smoke test, while
network-dependent tools and multi-host jobs need working container networking.
