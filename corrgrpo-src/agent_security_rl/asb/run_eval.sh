#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == --help ]]; then
  echo 'LOCAL_MODEL=/path/to/hf bash asb/run_eval.sh'
  exit 0
fi
BENCHMARK=asb
BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$BENCH_DIR/../_eval/local_server.sh"
collect_metrics() {
  "$PYTHON" "$SECURITY_ROOT/_eval/compute_metrics.py" run \
    --run-dir "$RUN_DIR" --benchmark "$BENCHMARK" \
    --injecagent-result-dir "$INJECAGENT_RESULT_DIR" \
    --injecagent-data-dir "$SECURITY_ROOT/injecagent/data" \
    --setting "${INJECAGENT_SETTING:-base}" \
    > "$RUN_DIR/metrics_collection.log" 2>&1 || true
}

on_exit() {
  local exit_code=$?
  trap - EXIT
  collect_metrics
  if [[ -n "${VLLM_SERVER_PID}" ]] && kill -0 "${VLLM_SERVER_PID}" 2>/dev/null; then
    kill "${VLLM_SERVER_PID}" 2>/dev/null || true
    wait "${VLLM_SERVER_PID}" 2>/dev/null || true
  fi
  exit "${exit_code}"
}
trap on_exit EXIT

ASB_DATA_DIR="$BENCH_DIR/data"
asb_attack_flag() {
  local method="$1"
  case "${method}" in
    direct_prompt_injection) printf '%s\n' --direct_prompt_injection ;;
    ipi|indirect_prompt_injection|opi|observation_prompt_injection)
      printf '%s\n' --observation_prompt_injection
      ;;
    memory_attack) printf '%s\n' --memory_attack ;;
    clean) printf '%s\n' --clean ;;
    mixed_attack) printf '%s\n' --direct_prompt_injection --observation_prompt_injection ;;
    *) echo "Unsupported ASB injection method: ${method}" >&2; return 2 ;;
  esac
}

asb_method_label() {
  case "$1" in
    ipi|indirect_prompt_injection|opi|observation_prompt_injection)
      printf '%s\n' indirect_prompt_injection
      ;;
    *) printf '%s\n' "$1" ;;
  esac
}

asb_default_attack_types() {
  case "$1" in
    direct_prompt_injection) printf '%s\n' "fake_completion escape_characters naive" ;;
    ipi|indirect_prompt_injection|opi|observation_prompt_injection)
      printf '%s\n' context_ignoring
      ;;
    *) printf '%s\n' combined_attack ;;
  esac
}

run_asb_config() {
  local tool_file="$1"
  local method="$2"
  local attack_type="$3"
  local method_label result_base
  local -a injection_flags
  mapfile -t injection_flags < <(asb_attack_flag "${method}")
  method_label="$(asb_method_label "${method}")"
  result_base="${RUN_DIR}/asb_${method_label}_${attack_type}"
  (
    cd "${BENCH_DIR}"
    "${PYTHON}" main_attacker.py \
      --llm_name "${EVAL_MODEL_NAME}" \
      --use_backend "${ASB_BACKEND}" \
      --max_gpu_memory "${MAX_GPU_MEMORY_JSON:-{\"0\":\"75GiB\"}}" \
      --max_new_tokens "${ASB_MAX_NEW_TOKENS:-1024}" \
      --attacker_tools_path "${tool_file}" \
      --tasks_path "${ASB_DATA_DIR}/agent_task.jsonl" \
      --tools_info_path "${ASB_DATA_DIR}/all_normal_tools.jsonl" \
      --attack_type "${attack_type}" \
      --task_num "${ASB_TASK_NUM:-1}" \
      --database "${RUN_DIR}/memory_db" \
      --res_file "${result_base}.csv" \
      "${injection_flags[@]}" \
      > "${result_base}.log" 2>&1
  )
}

run_asb() {
  local tool_file method attack_type attack_types methods pid parallelism failed
  local -a pids
  case "${ASB_ATTACK_TOOL_SET:-all}" in
    all) tool_file="${ASB_DATA_DIR}/all_attack_tools.jsonl" ;;
    agg) tool_file="${ASB_DATA_DIR}/all_attack_tools_aggressive.jsonl" ;;
    non-agg) tool_file="${ASB_DATA_DIR}/all_attack_tools_non_aggressive.jsonl" ;;
    test) tool_file="${ASB_DATA_DIR}/attack_tools_test.jsonl" ;;
    *) echo "Unsupported ASB_ATTACK_TOOL_SET=${ASB_ATTACK_TOOL_SET}" >&2; return 2 ;;
  esac
  methods="${ASB_INJECTION_METHODS:-${ASB_INJECTION_METHOD:-direct_prompt_injection indirect_prompt_injection}}"
  parallelism="${ASB_CONFIG_PARALLELISM:-4}"
  failed=0
  pids=()

  for method in ${methods}; do
    attack_types="${ASB_ATTACK_TYPES:-$(asb_default_attack_types "${method}")}"
    for attack_type in ${attack_types}; do
      run_asb_config "${tool_file}" "${method}" "${attack_type}" &
      pids+=("$!")
      if (( ${#pids[@]} >= parallelism )); then
        if ! wait "${pids[0]}"; then failed=1; fi
        pids=("${pids[@]:1}")
      fi
    done
  done
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then failed=1; fi
  done
  if (( failed != 0 )); then
    echo "One or more ASB configurations failed; inspect ${RUN_DIR}/asb_*.log" >&2
    return 1
  fi
}


if [[ "$LOCAL_BACKEND" == vllm_server ]]; then start_vllm_server; fi
run_asb
