#!/usr/bin/env bash
# Evaluate a merged Hugging Face checkpoint on the official 228-problem test split.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}
PYTHON_BIN=${PYTHON_BIN:-python3}
DATASET=${DATASET:-${SCRIPT_DIR}/raw/LeetCodeDataset-test.jsonl}
MODEL_PATH=${MODEL_PATH:?Set MODEL_PATH to a merged Hugging Face model}
OUTPUT_DIR=${OUTPUT_DIR:-${REPO_ROOT}/outputs/code/eval}
NUM_SAMPLES=${NUM_SAMPLES:-1}
PASS_K=${PASS_K:-1}
TEMPERATURE=${TEMPERATURE:-0.01}
TOP_P=${TOP_P:-1.0}
MAX_TOKENS=${MAX_TOKENS:-2048}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
DATA_PARALLEL_SIZE=${DATA_PARALLEL_SIZE:-1}
BATCH_SIZE=${BATCH_SIZE:-32}
SCORE_WORKERS=${SCORE_WORKERS:-8}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-5}
MEMORY_LIMIT_MB=${MEMORY_LIMIT_MB:-1024}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
SEED=${SEED:-42}
OUTPUT_FILE=${OUTPUT_FILE:-${OUTPUT_DIR}/test_n${NUM_SAMPLES}.jsonl}
OVERWRITE=${OVERWRITE:-0}

for path_var in MODEL_PATH DATASET OUTPUT_DIR OUTPUT_FILE; do
    [[ "${!path_var}" = /* ]] || printf -v "$path_var" '%s/%s' "$PWD" "${!path_var}"
done

if [[ ! ${CUDA_VISIBLE_DEVICES+x} ]]; then
    TOTAL_GPUS=$((TENSOR_PARALLEL_SIZE * DATA_PARALLEL_SIZE))
    GPU_LIST=0
    for ((GPU_INDEX = 1; GPU_INDEX < TOTAL_GPUS; GPU_INDEX++)); do
        GPU_LIST+=",${GPU_INDEX}"
    done
    export CUDA_VISIBLE_DEVICES=${GPU_LIST}
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "Merged Hugging Face model not found: ${MODEL_PATH}" >&2
    echo "Set MODEL_PATH=/path/to/merged/hf/model and run again." >&2
    exit 2
fi

mkdir -p "${OUTPUT_DIR}"

ARGS=(
    --dataset "${DATASET}"
    --model "${MODEL_PATH}"
    --output-file "${OUTPUT_FILE}"
    --num-samples "${NUM_SAMPLES}"
    --pass-k "${PASS_K}"
    --temperature "${TEMPERATURE}"
    --top-p "${TOP_P}"
    --max-tokens "${MAX_TOKENS}"
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --data-parallel-size "${DATA_PARALLEL_SIZE}"
    --batch-size "${BATCH_SIZE}"
    --score-workers "${SCORE_WORKERS}"
    --timeout-seconds "${TIMEOUT_SECONDS}"
    --memory-limit-mb "${MEMORY_LIMIT_MB}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --max-model-len "${MAX_MODEL_LEN}"
    --seed "${SEED}"
    --enable-prefix-caching
)

if [[ "${OVERWRITE}" == "1" ]]; then
    ARGS+=(--overwrite)
fi

cd "${SCRIPT_DIR}"
echo "Evaluation parallelism: DP=${DATA_PARALLEL_SIZE}, TP=${TENSOR_PARALLEL_SIZE}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
"${PYTHON_BIN}" evaluate_pass_rate.py "${ARGS[@]}" "$@"
