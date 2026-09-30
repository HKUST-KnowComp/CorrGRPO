#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

usage() {
  cat <<'EOF'
Usage: run_sandboxed_eval.sh BENCHMARK OUTPUT_DIR [VERSION] [MBPP_SUBSET]

BENCHMARK is all, humaneval, mbpp, or livecodebench. VERSION defaults to v6,
and MBPP_SUBSET defaults to the standard 500-task test split.
Generated code is executed only inside a locked-down Docker container with no
network, a read-only root filesystem, no Linux capabilities, no-new-privileges,
and explicit process, memory, and CPU limits. Only OUTPUT_DIR is writable.
EOF
  exit 2
}

[[ $# -ge 2 && $# -le 4 ]] || usage
BENCHMARK="$1"
OUTPUT_DIR="$(realpath "$2")"
VERSION="${3:-v6}"
MBPP_SUBSET="${4:-test}"

if [[ ! "$BENCHMARK" =~ ^(all|humaneval|mbpp|livecodebench)$ ]]; then
  echo "Invalid benchmark: $BENCHMARK" >&2
  exit 2
fi
if [[ ! "$MBPP_SUBSET" =~ ^(test|full|sanitized)$ ]]; then
  echo "Invalid MBPP subset: $MBPP_SUBSET" >&2
  exit 2
fi
if [[ ! "$VERSION" =~ ^(release_)?v[1-6]$ ]]; then
  echo "Invalid LiveCodeBench version: $VERSION" >&2
  exit 2
fi
if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required for sandboxed code execution." >&2
  exit 1
fi

EVAL_IMAGE="${EVAL_IMAGE:-corrgrpo:latest}"
# Docker bind sources are resolved by the host daemon, including when this
# launcher runs inside the training container.
HOST_CODE_DIR="${HOST_CODE_DIR:-${HOST_REPO_ROOT:-$PROJECT_DIR}/corrgrpo-src/code_rl}"
if [[ -z "${HOST_OUTPUT_DIR:-}" ]]; then
  if [[ -n "${HOST_REPO_ROOT:-}" ]]; then
    case "$OUTPUT_DIR" in
      "$PROJECT_DIR"/*) HOST_OUTPUT_DIR="$HOST_REPO_ROOT/${OUTPUT_DIR#"$PROJECT_DIR"/}" ;;
      *) echo "Set HOST_OUTPUT_DIR for outputs outside the mounted repository." >&2; exit 2 ;;
    esac
  else
    HOST_OUTPUT_DIR="$OUTPUT_DIR"
  fi
fi
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"
CONTAINER_PYTHONPATH="/bench/humaneval/vendor:/bench/livecodebench/vendor"

docker_eval() {
  local container_workdir="${DOCKER_WORKDIR:-/output}"
  docker run --rm \
    --network none \
    --read-only \
    --cap-drop ALL \
    --security-opt no-new-privileges \
    --pids-limit 512 \
    --memory 8g \
    --memory-swap 8g \
    --cpus 16 \
    --ulimit nofile=4096:4096 \
    --tmpfs /tmp:rw,noexec,nosuid,size=2g \
    --user "$HOST_UID:$HOST_GID" \
    -v "$HOST_CODE_DIR:/bench:ro" \
    -v "$HOST_OUTPUT_DIR:/output:rw" \
    -w "$container_workdir" \
    -e HOME=/tmp \
    -e PYTHONPATH="$CONTAINER_PYTHONPATH" \
    -e HUMANEVAL_ALLOW_UNSAFE_EXECUTION=1 \
    -e MBPP_ALLOW_UNSAFE_EXECUTION=1 \
    -e LCB_CODE_GENERATION_DATASET=/bench/livecodebench/data \
    "$EVAL_IMAGE" "$@"
}

if [[ "$BENCHMARK" == all || "$BENCHMARK" == humaneval ]]; then
  [[ -f "$OUTPUT_DIR/humaneval.samples.jsonl" ]] || {
    echo "Missing HumanEval samples: $OUTPUT_DIR/humaneval.samples.jsonl" >&2
    exit 2
  }
  echo "Evaluating HumanEval inside Docker sandbox"
  docker_eval python3 -m human_eval.evaluate_functional_correctness \
    /output/humaneval.samples.jsonl \
    --problem_file=/bench/humaneval/vendor/data/HumanEval.jsonl.gz \
    --n_workers=16 \
    --timeout=3
fi

if [[ "$BENCHMARK" == all || "$BENCHMARK" == mbpp ]]; then
  [[ -f "$OUTPUT_DIR/mbpp_${MBPP_SUBSET}.samples.jsonl" ]] || {
    echo "Missing MBPP samples: $OUTPUT_DIR/mbpp_${MBPP_SUBSET}.samples.jsonl" >&2
    exit 2
  }
  echo "Evaluating MBPP $MBPP_SUBSET inside Docker sandbox"
  DOCKER_WORKDIR=/tmp docker_eval \
    python3 /bench/mbpp/evaluate.py \
    "/output/mbpp_${MBPP_SUBSET}.samples.jsonl" \
    --subset "$MBPP_SUBSET" --base-dir /bench --workers 16 --timeout 10
fi

if [[ "$BENCHMARK" == all || "$BENCHMARK" == livecodebench ]]; then
  [[ -f "$OUTPUT_DIR/lcb_${VERSION}.custom_outputs.json" ]] || {
    echo "Missing LiveCodeBench outputs: $OUTPUT_DIR/lcb_${VERSION}.custom_outputs.json" >&2
    exit 2
  }
  echo "Evaluating LiveCodeBench $VERSION inside Docker sandbox"
  DOCKER_WORKDIR=/bench/livecodebench/vendor docker_eval \
    python3 -m lcb_runner.runner.custom_evaluator \
    --scenario codegeneration \
    --release_version "$VERSION" \
    --custom_output_file "/output/lcb_${VERSION}.custom_outputs.json" \
    --num_process_evaluate 16 \
    --timeout 10
fi

echo "Sandboxed evaluation complete: $OUTPUT_DIR"
