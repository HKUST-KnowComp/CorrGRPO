#!/usr/bin/env bash
set -euo pipefail

# Paths and defaults. These values can be overridden before running the script.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CALLER_DIR="$PWD"

PYTHON="${PYTHON_BIN:-python3}"
VLLM="${VLLM_BIN:-$(dirname "$(command -v "$PYTHON")")/vllm}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a merged Hugging Face model}"
GPU_IDS="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -r -a GPU_ID_LIST <<<"${GPU_IDS}"
GPU_COUNT="${#GPU_ID_LIST[@]}"
VLLM_DATA_PARALLEL_SIZE="${VLLM_DATA_PARALLEL_SIZE:-${GPU_COUNT}}"
PORT="${LOCAL_LLM_PORT:-8000}"
OUTPUT_PATH="${OUTPUT_PATH:-${REPO_ROOT:-$(cd "$ROOT/../../.." && pwd)}/outputs/agentdojo/official/metrics.json}"
OUTPUT_MD_PATH="${OUTPUT_MD_PATH:-}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
AGENTDOJO_MAX_TOKENS="${AGENTDOJO_MAX_TOKENS:-2048}"
AGENTDOJO_MAX_MODEL_LEN="${AGENTDOJO_MAX_MODEL_LEN:-${VLLM_MAX_MODEL_LEN}}"
AGENTDOJO_MIN_OUTPUT_TOKENS="${AGENTDOJO_MIN_OUTPUT_TOKENS:-512}"
AGENTDOJO_CONTEXT_MARGIN="${AGENTDOJO_CONTEXT_MARGIN:-256}"
AGENTDOJO_ENABLE_THINKING="${AGENTDOJO_ENABLE_THINKING:-0}"
AGENTDOJO_EVAL_WORKERS="${AGENTDOJO_EVAL_WORKERS:-${VLLM_DATA_PARALLEL_SIZE}}"
export AGENTDOJO_EVAL_WORKERS
LOG_FILE="${OUTPUT_DIR:-$(dirname "$OUTPUT_PATH")}/vllm_server.log"
PID_FILE="${OUTPUT_DIR:-$(dirname "$OUTPUT_PATH")}/vllm_server.pid"
VLLM_PID=""
VLLM_STARTED_BY_SCRIPT=0

# Stop only a vLLM service managed by this wrapper. An unrelated server that
# was already running on LOCAL_LLM_PORT is reused and left untouched.
managed_vllm_is_running() {
    kill -0 "${VLLM_PID}" 2>/dev/null || kill -0 -- "-${VLLM_PID}" 2>/dev/null
}

stop_managed_vllm() {
    local exit_code=$?

    trap - EXIT INT TERM
    if [[ "${VLLM_STARTED_BY_SCRIPT}" == "1" && -n "${VLLM_PID}" ]]; then
        echo "Stopping vLLM (PID ${VLLM_PID})..."

        # vLLM is started in its own process group, so this also stops its
        # data-parallel workers and engine subprocesses.
        kill -TERM -- "-${VLLM_PID}" 2>/dev/null || true
        kill -TERM "${VLLM_PID}" 2>/dev/null || true
        for _ in {1..30}; do
            if ! managed_vllm_is_running; then
                break
            fi
            sleep 1
        done
        if managed_vllm_is_running; then
            echo "vLLM did not stop within 30 seconds; forcing shutdown."
            kill -KILL -- "-${VLLM_PID}" 2>/dev/null || true
            kill -KILL "${VLLM_PID}" 2>/dev/null || true
        fi
        wait "${VLLM_PID}" 2>/dev/null || true

        if [[ -f "${PID_FILE}" && "$(<"${PID_FILE}")" == "${VLLM_PID}" ]]; then
            rm -f "${PID_FILE}"
        fi
        echo "vLLM stopped."
    fi

    exit "${exit_code}"
}

trap stop_managed_vllm EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

TRACE_LOGDIR="${TRACE_LOGDIR:-}"

# Save all evaluator arguments, then read the paths and port used by this wrapper.
EVAL_ARGS=("$@")
MODEL_ARG_SET=0
OUTPUT_ARG_SET=0
OUTPUT_MD_ARG_SET=0
LOGDIR_ARG_SET=0
WORKERS_ARG_SET=0
SMOKE=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-id)
            MODEL_PATH="$2"
            MODEL_ARG_SET=1
            shift 2
            ;;
        --model-id=*)
            MODEL_PATH="${1#*=}"
            MODEL_ARG_SET=1
            shift
            ;;
        --local-port)
            PORT="$2"
            shift 2
            ;;
        --local-port=*)
            PORT="${1#*=}"
            shift
            ;;
        --output-json)
            OUTPUT_PATH="$2"
            OUTPUT_ARG_SET=1
            shift 2
            ;;
        --output-json=*)
            OUTPUT_PATH="${1#*=}"
            OUTPUT_ARG_SET=1
            shift
            ;;
        --output-md)
            OUTPUT_MD_PATH="$2"
            OUTPUT_MD_ARG_SET=1
            shift 2
            ;;
        --output-md=*)
            OUTPUT_MD_PATH="${1#*=}"
            OUTPUT_MD_ARG_SET=1
            shift
            ;;
        --logdir)
            TRACE_LOGDIR="$2"
            LOGDIR_ARG_SET=1
            shift 2
            ;;
        --logdir=*)
            TRACE_LOGDIR="${1#*=}"
            LOGDIR_ARG_SET=1
            shift
            ;;
        --workers)
            AGENTDOJO_EVAL_WORKERS="$2"
            WORKERS_ARG_SET=1
            shift 2
            ;;
        --workers=*)
            AGENTDOJO_EVAL_WORKERS="${1#*=}"
            WORKERS_ARG_SET=1
            shift
            ;;
        -h|--help)
            exec "${PYTHON}" "${ROOT}/evaluate_agentdojo_official.py" "${EVAL_ARGS[@]}"
            ;;
        --smoke)
            SMOKE=1
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# Resolve caller-relative paths before entering the benchmark directory.
for path_var in MODEL_PATH OUTPUT_PATH OUTPUT_MD_PATH TRACE_LOGDIR OUTPUT_DIR LOG_FILE PID_FILE; do
    if [[ -n "${!path_var:-}" && "${!path_var}" != /* ]]; then
        printf -v "$path_var" '%s/%s' "$CALLER_DIR" "${!path_var}"
    fi
done
# Append the normalized forms after user arguments so argparse receives them.
MODEL_ARG_SET=0
OUTPUT_ARG_SET=0
OUTPUT_MD_ARG_SET=0
LOGDIR_ARG_SET=0
cd "$ROOT"

# Use a model-specific trace directory so results from different checkpoints
# can never be silently reused as AgentDojo's cache.
if [[ -z "${TRACE_LOGDIR}" ]]; then
    MODEL_ROOT="$(basename "$(dirname "$(dirname "${MODEL_PATH}")")")"
    MODEL_TAG="${MODEL_ROOT#models--}"
    MODEL_TAG="${MODEL_TAG//--/_}"
    MODEL_TAG="${MODEL_TAG//\//_}"
    TRACE_LOGDIR="$(dirname "$OUTPUT_PATH")/traces/${MODEL_TAG}"
fi

# Always save a JSON report. --output-json takes precedence over OUTPUT_PATH.
if [[ "${MODEL_ARG_SET}" == "0" ]]; then
    EVAL_ARGS+=(--model-id "${MODEL_PATH}")
fi
if [[ "${OUTPUT_ARG_SET}" == "0" ]]; then
    EVAL_ARGS+=(--output-json "${OUTPUT_PATH}")
fi
if [[ -z "${OUTPUT_MD_PATH}" ]]; then
    if [[ "${OUTPUT_PATH}" == *.json ]]; then
        OUTPUT_MD_PATH="${OUTPUT_PATH%.json}.md"
    else
        OUTPUT_MD_PATH="${OUTPUT_PATH}.md"
    fi
fi
if [[ "${OUTPUT_MD_ARG_SET}" == "0" ]]; then
    EVAL_ARGS+=(--output-md "${OUTPUT_MD_PATH}")
fi
if [[ "${LOGDIR_ARG_SET}" == "0" ]]; then
    EVAL_ARGS+=(--logdir "${TRACE_LOGDIR}")
fi
echo "Result JSON: ${OUTPUT_PATH}"
echo "Summary MD:  ${OUTPUT_MD_PATH}"
echo "Trace cache: ${TRACE_LOGDIR}"
echo "Context/output: ${AGENTDOJO_MAX_MODEL_LEN}/${AGENTDOJO_MAX_TOKENS} tokens"
echo "Minimum output reserve/context margin: ${AGENTDOJO_MIN_OUTPUT_TOKENS}/${AGENTDOJO_CONTEXT_MARGIN} tokens"
echo "Visible GPUs / vLLM replicas: ${GPU_COUNT}/${VLLM_DATA_PARALLEL_SIZE}"
if [[ "${SMOKE}" == "1" && "${WORKERS_ARG_SET}" == "0" ]]; then
    echo "Evaluator workers: 1 (smoke default)"
else
    echo "Evaluator workers: ${AGENTDOJO_EVAL_WORKERS}"
fi

# Smoke mode exercises the official tasks and checkers without a model server.
if [[ "${SMOKE}" == "1" ]]; then
    exec "${PYTHON}" "${ROOT}/evaluate_agentdojo_official.py" "${EVAL_ARGS[@]}"
fi

MODELS_URL="http://127.0.0.1:${PORT}/v1/models"

# Reuse the server when the requested model is already loaded.
if SERVER_INFO="$(curl -fsS --max-time 3 "${MODELS_URL}" 2>/dev/null)"; then
    SERVER_MODEL="$("${PYTHON}" -c 'import json,sys; print(json.loads(sys.argv[1])["data"][0].get("id", ""))' "${SERVER_INFO}")"
    SERVER_MAX_LEN="$("${PYTHON}" -c 'import json,sys; print(json.loads(sys.argv[1])["data"][0].get("max_model_len", 0))' "${SERVER_INFO}")"
    if [[ "${SERVER_MODEL}" != "${MODEL_PATH}" ]]; then
        echo "ERROR: port ${PORT} is serving a different model." >&2
        echo "Requested: ${MODEL_PATH}" >&2
        echo "Running:   ${SERVER_MODEL}" >&2
        echo "Stop that server or choose another LOCAL_LLM_PORT." >&2
        exit 1
    fi
    if (( SERVER_MAX_LEN < VLLM_MAX_MODEL_LEN )); then
        echo "ERROR: vLLM on port ${PORT} only supports ${SERVER_MAX_LEN} tokens." >&2
        echo "This run requires ${VLLM_MAX_MODEL_LEN}; restart the managed vLLM server first." >&2
        exit 1
    fi
    echo "Reusing vLLM on port ${PORT} (${SERVER_MAX_LEN} tokens)."

    # A service left by an older invocation of this wrapper is still ours to
    # clean up. Do not adopt an unrelated process that merely uses this port.
    if [[ -f "${PID_FILE}" ]]; then
        MANAGED_PID="$(<"${PID_FILE}")"
        if [[ "${MANAGED_PID}" =~ ^[0-9]+$ ]] \
            && kill -0 "${MANAGED_PID}" 2>/dev/null \
            && [[ "$(ps -p "${MANAGED_PID}" -o args= 2>/dev/null)" == *"vllm serve"* ]]; then
            VLLM_PID="${MANAGED_PID}"
            VLLM_STARTED_BY_SCRIPT=1
            echo "This vLLM is managed by the wrapper and will stop after evaluation."
        fi
    fi
else
    echo "Starting vLLM on GPUs ${GPU_IDS}..."

    export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
    mkdir -p "$(dirname "$LOG_FILE")"
    nohup setsid "${VLLM}" serve "${MODEL_PATH}" \
        --tensor-parallel-size 1 \
        --data-parallel-size "${VLLM_DATA_PARALLEL_SIZE}" \
        --host 127.0.0.1 --port "${PORT}" \
        --dtype bfloat16 --max-model-len "${VLLM_MAX_MODEL_LEN}" \
        --gpu-memory-utilization 0.90 \
        >"${LOG_FILE}" 2>&1 </dev/null &

    VLLM_PID=$!
    VLLM_STARTED_BY_SCRIPT=1
    printf '%s\n' "${VLLM_PID}" >"${PID_FILE}"

    # Wait up to 10 minutes for vLLM to become ready.
    for ATTEMPT in {1..120}; do
        if curl -fsS --max-time 3 "${MODELS_URL}" >/dev/null 2>&1; then
            echo "vLLM is ready. Log: ${LOG_FILE}"
            break
        fi
        if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
            echo "ERROR: vLLM exited during startup." >&2
            tail -80 "${LOG_FILE}" >&2 || true
            exit 1
        fi
        if [[ "${ATTEMPT}" == "120" ]]; then
            echo "ERROR: vLLM startup timed out." >&2
            tail -80 "${LOG_FILE}" >&2 || true
            exit 1
        fi
        sleep 5
    done
fi

# Run AgentDojo after the model server is ready.
export LOCAL_LLM_PORT="${PORT}"
export AGENTDOJO_MAX_TOKENS AGENTDOJO_MAX_MODEL_LEN
export AGENTDOJO_MIN_OUTPUT_TOKENS AGENTDOJO_CONTEXT_MARGIN
export AGENTDOJO_ENABLE_THINKING
"${PYTHON}" "${ROOT}/evaluate_agentdojo_official.py" "${EVAL_ARGS[@]}"
