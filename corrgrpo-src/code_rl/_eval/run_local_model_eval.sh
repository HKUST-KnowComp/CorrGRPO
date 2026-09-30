#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

usage() {
  cat <<'EOF'
Usage: run_local_model_eval.sh [options]

Options:
  --benchmark NAME       all, humaneval, mbpp, or livecodebench (default: all)
  --version VERSION      v1..v6 or release_v1..release_v6 (default: v6)
  --mbpp-subset NAME     test, sanitized, or full (default: test)
  --model-path PATH      Local Hugging Face model directory (required)
  --gpus LIST            CUDA_VISIBLE_DEVICES value (default: 0)
  --tensor-parallel N    GPUs used by each vLLM replica (default: 1)
  --data-parallel N      Isolated task-sharded vLLM replicas (default: 1)
  --max-tokens N         Maximum generated tokens (default: 4096)
  --batch-size N         Checkpoint batch size (default: 32)
  --temperature FLOAT    Sampling temperature (default: 0)
  --top-p FLOAT          Top-p sampling value (default: 1)
  --samples N            Samples per problem (default: 1)
  --output-dir PATH      Output directory
  --limit N              Generate only the first N tasks (smoke test)
  --no-evaluate          Generate outputs but do not execute official tests
  -h, --help             Show this help

The generation checkpoint is append-only and automatically resumed. Delete or
rename a benchmark's *.generations.jsonl file to deliberately start it over.
Official evaluation executes generated code inside the locked-down Docker
sandbox configured by run_sandboxed_eval.sh.
EOF
}

BENCHMARK=all
VERSION=v6
MBPP_SUBSET=test
MODEL_PATH="${MODEL_PATH:-}"
GPUS=0
TENSOR_PARALLEL=1
DATA_PARALLEL=1
MAX_TOKENS=4096
BATCH_SIZE=32
TEMPERATURE=0
TOP_P=1
SAMPLES=1
OUTPUT_DIR=""
LIMIT=""
EVALUATE=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --benchmark) BENCHMARK="$2"; shift 2 ;;
    --version) VERSION="$2"; shift 2 ;;
    --mbpp-subset) MBPP_SUBSET="$2"; shift 2 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --tensor-parallel) TENSOR_PARALLEL="$2"; shift 2 ;;
    --data-parallel) DATA_PARALLEL="$2"; shift 2 ;;
    --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --top-p) TOP_P="$2"; shift 2 ;;
    --samples) SAMPLES="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --no-evaluate) EVALUATE=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

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
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Model directory not found: $MODEL_PATH" >&2
  exit 2
fi
for value_name in TENSOR_PARALLEL DATA_PARALLEL MAX_TOKENS BATCH_SIZE SAMPLES; do
  value="${!value_name}"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$value_name must be a positive integer: $value" >&2
    exit 2
  fi
done

IFS=',' read -r -a GPU_ARRAY <<< "$GPUS"
for index in "${!GPU_ARRAY[@]}"; do
  GPU_ARRAY[$index]="${GPU_ARRAY[$index]//[[:space:]]/}"
  if [[ -z "${GPU_ARRAY[$index]}" ]]; then
    echo "Invalid empty GPU entry in --gpus: $GPUS" >&2
    exit 2
  fi
done
REQUIRED_GPUS=$((DATA_PARALLEL * TENSOR_PARALLEL))
if (( ${#GPU_ARRAY[@]} < REQUIRED_GPUS )); then
  echo "Need $REQUIRED_GPUS GPUs for data_parallel=$DATA_PARALLEL and tensor_parallel=$TENSOR_PARALLEL, but --gpus exposes ${#GPU_ARRAY[@]}" >&2
  exit 2
fi

MODEL_NAME="$(basename "$MODEL_PATH")"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs/code/$MODEL_NAME/eval}"
mkdir -p "$OUTPUT_DIR"

GEN_ARGS=(
  --benchmark "$BENCHMARK"
  --version "$VERSION"
  --mbpp-subset "$MBPP_SUBSET"
  --model-path "$MODEL_PATH"
  --tensor-parallel-size "$TENSOR_PARALLEL"
  --data-parallel-size 1
  --max-tokens "$MAX_TOKENS"
  --batch-size "$BATCH_SIZE"
  --temperature "$TEMPERATURE"
  --top-p "$TOP_P"
  --samples-per-problem "$SAMPLES"
)
if [[ -n "$LIMIT" ]]; then
  GEN_ARGS+=(--limit "$LIMIT")
fi

echo "Starting local generation"
echo "  benchmark=$BENCHMARK version=$VERSION model=$MODEL_PATH GPUs=$GPUS"
echo "  data_parallel=$DATA_PARALLEL tensor_parallel=$TENSOR_PARALLEL"
echo "  output=$OUTPUT_DIR"
# This host has the FlashInfer Python package but no CUDA 12.x nvcc on PATH.
# Use vLLM's native PyTorch sampler so startup does not trigger FlashInfer JIT.
if (( DATA_PARALLEL == 1 )); then
  ACTIVE_GPUS="$(IFS=,; echo "${GPU_ARRAY[*]:0:TENSOR_PARALLEL}")"
  VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_VISIBLE_DEVICES="$ACTIVE_GPUS" \
    "$PYTHON_BIN" "$SCRIPT_DIR/generate_local_eval.py" \
      "${GEN_ARGS[@]}" --num-shards 1 --shard-index 0 --output-dir "$OUTPUT_DIR"
else
  if ! command -v setsid >/dev/null 2>&1; then
    echo "setsid is required for isolated multi-GPU worker cleanup" >&2
    exit 1
  fi

  SHARD_ROOT="$OUTPUT_DIR/.generation_shards/dp${DATA_PARALLEL}_tp${TENSOR_PARALLEL}"
  mkdir -p "$SHARD_ROOT"
  WORKER_PIDS=()
  WORKER_LOGS=()
  # vLLM's automatic free-port lookup can race when many isolated workers
  # start together. Give every worker its own small, non-overlapping range.
  RUN_PORT_BASE=$((20000 + ($$ % 100) * 320))

  cleanup_workers() {
    local pid attempt active
    for pid in "${WORKER_PIDS[@]:-}"; do
      kill -TERM -- "-$pid" 2>/dev/null || true
    done
    for attempt in {1..10}; do
      active=0
      for pid in "${WORKER_PIDS[@]:-}"; do
        if kill -0 -- "-$pid" 2>/dev/null; then
          ((active += 1))
        fi
      done
      (( active == 0 )) && break
      sleep 1
    done
    for pid in "${WORKER_PIDS[@]:-}"; do
      kill -KILL -- "-$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    done
  }
  interrupted() {
    trap - INT TERM HUP
    echo "Stopping data-parallel workers..." >&2
    cleanup_workers
    exit 130
  }
  trap interrupted INT TERM HUP

  for ((rank = 0; rank < DATA_PARALLEL; rank++)); do
    offset=$((rank * TENSOR_PARALLEL))
    WORKER_GPUS="$(IFS=,; echo "${GPU_ARRAY[*]:offset:TENSOR_PARALLEL}")"
    WORKER_PORT_BASE=$((RUN_PORT_BASE + rank * 32))
    WORKER_DIR="$SHARD_ROOT/shard_$rank"
    WORKER_LOG="$WORKER_DIR/worker.log"
    mkdir -p "$WORKER_DIR"
    echo "  worker=$rank GPUs=$WORKER_GPUS port_base=$WORKER_PORT_BASE log=$WORKER_LOG"
    setsid env VLLM_PORT="$WORKER_PORT_BASE" \
      VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_VISIBLE_DEVICES="$WORKER_GPUS" \
      "$PYTHON_BIN" "$SCRIPT_DIR/generate_local_eval.py" \
        "${GEN_ARGS[@]}" \
        --num-shards "$DATA_PARALLEL" --shard-index "$rank" \
        --output-dir "$WORKER_DIR" >"$WORKER_LOG" 2>&1 &
    WORKER_PIDS+=("$!")
    WORKER_LOGS+=("$WORKER_LOG")
  done

  FAILED=0
  for index in "${!WORKER_PIDS[@]}"; do
    if wait "${WORKER_PIDS[$index]}"; then
      echo "  worker=$index complete"
    else
      status=$?
      FAILED=1
      echo "Worker $index failed with status $status; tail of ${WORKER_LOGS[$index]}:" >&2
      tail -n 80 "${WORKER_LOGS[$index]}" >&2 || true
    fi
  done
  trap - INT TERM HUP
  if (( FAILED != 0 )); then
    cleanup_workers
    exit 1
  fi

  "$PYTHON_BIN" "$SCRIPT_DIR/merge_local_eval_shards.py" \
    --benchmark "$BENCHMARK" --version "$VERSION" \
    --mbpp-subset "$MBPP_SUBSET" --output-dir "$OUTPUT_DIR" \
    --shard-root "$SHARD_ROOT" --num-shards "$DATA_PARALLEL" \
    --samples-per-problem "$SAMPLES" \
    --tensor-parallel-size "$TENSOR_PARALLEL"
fi

if [[ "$EVALUATE" -eq 0 ]]; then
  echo "Generation complete; evaluation skipped."
  exit 0
fi
if [[ -n "$LIMIT" ]]; then
  echo "Refusing official evaluation of a partial --limit run; generation smoke test is complete."
  exit 0
fi

"$SCRIPT_DIR/run_sandboxed_eval.sh" \
  "$BENCHMARK" "$OUTPUT_DIR" "$VERSION" "$MBPP_SUBSET"
"$PYTHON_BIN" "$SCRIPT_DIR/collect_eval_results.py" \
  --benchmark "$BENCHMARK" --output-dir "$OUTPUT_DIR" \
  --version "$VERSION" --mbpp-subset "$MBPP_SUBSET"

echo "Local model evaluation complete: $OUTPUT_DIR"
