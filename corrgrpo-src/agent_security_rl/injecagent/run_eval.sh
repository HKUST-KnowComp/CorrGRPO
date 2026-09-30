#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == --help ]]; then
  echo 'LOCAL_MODEL=/path/to/hf bash injecagent/run_eval.sh'
  exit 0
fi
BENCHMARK=injecagent
BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$BENCH_DIR/../_eval/local_server.sh"
collect_metrics() {
  "$PYTHON" "$SECURITY_ROOT/_eval/compute_metrics.py" run \
    --run-dir "$RUN_DIR" --benchmark "$BENCHMARK" \
    --injecagent-result-dir "$INJECAGENT_RESULT_DIR" \
    --injecagent-data-dir "$SECURITY_ROOT/injecagent/data" \
    --setting "${INJECAGENT_SETTING:-base}" \
    > "$RUN_DIR/metrics_collection.log" 2>&1 || true
  "$PYTHON" "$BENCH_DIR/compute_injecagent_relaxed_metrics.py" \
    --injecagent-root "$BENCH_DIR" --result-dir "$INJECAGENT_RESULT_DIR" \
    --setting "${INJECAGENT_SETTING:-base}" --output "$RUN_DIR/injecagent_relaxed_metrics.json" \
    >> "$RUN_DIR/metrics_collection.log" 2>&1 || true
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

run_injecagent() {
  cd "${BENCH_DIR}"
  export PYTHONPATH=.
  local -a attacks
  read -r -a attacks <<< "${INJECAGENT_ATTACKS:-dh ds}"
  "${PYTHON}" src/evaluate_prompted_agent.py \
    --model_type "${INJECAGENT_MODEL_TYPE}" \
    --model_name "${EVAL_MODEL_NAME}" \
    --setting "${INJECAGENT_SETTING:-base}" \
    --prompt_type "${INJECAGENT_PROMPT_TYPE:-InjecAgent}" \
    --attacks "${attacks[@]}" \
    --num_workers "${INJECAGENT_NUM_WORKERS:-64}" \
    --max_cases "${INJECAGENT_MAX_CASES:-0}" \
    --use_cache
}


if [[ "$LOCAL_BACKEND" == vllm_server ]]; then start_vllm_server; fi
run_injecagent 2>&1 | tee "$RUN_DIR/injecagent.log"
