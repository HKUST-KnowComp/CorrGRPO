#!/usr/bin/env bash
# Shared local model server setup; sourced by each benchmark's run_eval.sh.
set -euo pipefail
SECURITY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$(command -v "${PYTHON_BIN:-python3}")"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOCAL_BACKEND="${LOCAL_BACKEND:-vllm_server}"
VLLM_SERVER_PID=""

if [[ -z "${LOCAL_MODEL:-}" ]]; then
  echo "Set LOCAL_MODEL to a Hugging Face model ID or local model path." >&2
  exit 2
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing ${PYTHON}; activate the evaluation environment and set PYTHON_BIN." >&2
  exit 2
fi
if [[ "${LOCAL_BACKEND}" != "vllm_server" && "${LOCAL_BACKEND}" != "transformers" ]]; then
  echo "LOCAL_BACKEND must be vllm_server or transformers." >&2
  exit 2
fi

if [[ "$LOCAL_MODEL" != /* && -e "$LOCAL_MODEL" ]]; then
  LOCAL_MODEL="$PWD/$LOCAL_MODEL"
fi
MODEL_SLUG="$(printf '%s' "${LOCAL_MODEL}" | tr '/ :' '___')"
LOCAL_SERVED_MODEL_NAME="${LOCAL_SERVED_MODEL_NAME:-local-${MODEL_SLUG}}"
RUN_DIR="${RUN_DIR:-${SECURITY_ROOT}/../../outputs/security/${BENCHMARK}/${MODEL_SLUG}/${RUN_ID}}"
[[ "$RUN_DIR" = /* ]] || RUN_DIR="$PWD/$RUN_DIR"
mkdir -p "${RUN_DIR}"
export ASB_SKIP_JUDGE="${ASB_SKIP_JUDGE:-1}"
export ASB_SCHEDULER_WORKERS="${ASB_SCHEDULER_WORKERS:-16}"
export ASB_OPENAI_TOOL_MODE="${ASB_OPENAI_TOOL_MODE:-text}"

if [[ "${LOCAL_BACKEND}" == "vllm_server" ]]; then
  INJECAGENT_MODEL_TYPE=GPT
  EVAL_MODEL_NAME="${LOCAL_SERVED_MODEL_NAME}"
  ASB_BACKEND=openai
  export INJECAGENT_SIMULATOR_MODEL="${INJECAGENT_SIMULATOR_MODEL:-${EVAL_MODEL_NAME}}"
else
  INJECAGENT_MODEL_TYPE=Llama
  EVAL_MODEL_NAME="${LOCAL_MODEL}"
  ASB_BACKEND=vllm
fi
export INJECAGENT_RESULT_DIR="${INJECAGENT_RESULT_DIR:-${RUN_DIR}/injecagent}"
[[ "$INJECAGENT_RESULT_DIR" = /* ]] || export INJECAGENT_RESULT_DIR="$PWD/$INJECAGENT_RESULT_DIR"

wait_for_vllm() {
  local health_url="$1"
  local timeout_seconds="${VLLM_START_TIMEOUT:-600}"
  local waited=0
  until "${PYTHON}" -c \
    'import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=2).read()' \
    "${health_url}" >/dev/null 2>&1; do
    if [[ -n "${VLLM_SERVER_PID}" ]] && ! kill -0 "${VLLM_SERVER_PID}" 2>/dev/null; then
      echo "vLLM server exited during startup. Last log lines:" >&2
      tail -100 "${RUN_DIR}/vllm_server.log" >&2 || true
      return 1
    fi
    if (( waited >= timeout_seconds )); then
      echo "Timed out waiting for vLLM at ${health_url}." >&2
      tail -100 "${RUN_DIR}/vllm_server.log" >&2 || true
      return 1
    fi
    sleep 2
    waited=$((waited + 2))
  done
}

start_vllm_server() {
  if [[ -n "${LOCAL_OPENAI_BASE_URL:-}" ]]; then
    export OPENAI_BASE_URL="${LOCAL_OPENAI_BASE_URL%/}"
    export OPENAI_API_KEY="${LOCAL_OPENAI_API_KEY:-local}"
    wait_for_vllm "${OPENAI_BASE_URL%/v1}/health"
    echo "Reusing vLLM/OpenAI endpoint ${OPENAI_BASE_URL}"
    return
  fi

  local port="${LOCAL_VLLM_PORT:-8100}"
  local visible_devices="${LOCAL_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
  local tensor_parallel_size="${LOCAL_TENSOR_PARALLEL_SIZE:-1}"
  local comma_list="${visible_devices//[^,]/}"
  local visible_device_count=$(( ${#comma_list} + 1 ))
  local data_parallel_size="${LOCAL_DATA_PARALLEL_SIZE:-$((visible_device_count / tensor_parallel_size))}"
  local -a server_args extra_args
  server_args=(
    --model "${LOCAL_MODEL}"
    --served-model-name "${LOCAL_SERVED_MODEL_NAME}"
    --host 127.0.0.1
    --port "${port}"
    --dtype "${LOCAL_DTYPE:-bfloat16}"
    --max-model-len "${LOCAL_MAX_MODEL_LEN:-8192}"
    --gpu-memory-utilization "${LOCAL_GPU_MEMORY_UTILIZATION:-0.8}"
    --max-num-seqs "${LOCAL_MAX_NUM_SEQS:-64}"
    --generation-config "${LOCAL_GENERATION_CONFIG:-vllm}"
    --enable-prefix-caching
    --enable-auto-tool-choice
    --tool-call-parser "${LOCAL_TOOL_CALL_PARSER:-hermes}"
  )
  if (( data_parallel_size > 1 )); then
    server_args+=(--data-parallel-size "${data_parallel_size}")
  fi
  if (( tensor_parallel_size > 1 )); then
    server_args+=(--tensor-parallel-size "${tensor_parallel_size}")
  fi
  if [[ -n "${VLLM_EXTRA_ARGS:-}" ]]; then
    read -r -a extra_args <<< "${VLLM_EXTRA_ARGS}"
    server_args+=("${extra_args[@]}")
  fi

  echo "Starting one shared vLLM server on GPUs ${visible_devices} (DP=${data_parallel_size}, TP=${tensor_parallel_size})"
  CUDA_VISIBLE_DEVICES="${visible_devices}" \
    VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}" \
    "${PYTHON}" -m vllm.entrypoints.openai.api_server "${server_args[@]}" \
    > "${RUN_DIR}/vllm_server.log" 2>&1 &
  VLLM_SERVER_PID=$!
  export OPENAI_BASE_URL="http://127.0.0.1:${port}/v1"
  export OPENAI_API_KEY="${LOCAL_OPENAI_API_KEY:-local}"
  wait_for_vllm "http://127.0.0.1:${port}/health"
  echo "vLLM server is ready (PID ${VLLM_SERVER_PID})"
}

