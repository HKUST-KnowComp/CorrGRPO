#!/usr/bin/env bash
# Single-model pipeline: train -> convert FSDP to Hugging Face -> evaluate. Stop on any failure.
set -euo pipefail
if [[ "${1:-}" == --help ]]; then
  echo 'bash code_rl/run.sh [Hydra training overrides...]'
  echo 'Set the model, GPUs, training options, and benchmark at the top; DRY_RUN=1 prints commands only.'
  exit 0
fi

# 1. Common settings: edit here or override with MODEL=... bash run.sh.
MODEL="${MODEL:-qwen25-coder-7b}"
PYTHON_BIN="${PYTHON_BIN:-python}"  # Use the activated environment; override when needed.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"
ADV_ESTIMATOR="${ADV_ESTIMATOR:-grpo_covariance_coefficient}"  # CorrGRPO; alternatively, grpo
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-15}"
ACTOR_LR="${ACTOR_LR:-1e-6}"
ROLLOUT_TP="${ROLLOUT_TP:-1}"
SAVE_FREQ="${SAVE_FREQ:-100}"
TEST_FREQ="${TEST_FREQ:-10}"
BENCHMARK="${BENCHMARK:-code}"

# Available benchmarks: code|humaneval|mbpp|livecodebench
case "$BENCHMARK" in code|humaneval|mbpp|livecodebench) ;; *) echo "Invalid BENCHMARK: $BENCHMARK" >&2; exit 2;; esac
case "$ADV_ESTIMATOR" in grpo|grpo_covariance_coefficient) ;; *) echo "Invalid ADV_ESTIMATOR" >&2; exit 2;; esac

# Model selection; set MODEL_PATH to local base or SFT weights to bypass automatic lookup.
case "$MODEL" in
  qwen25-0.5b) MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct ;;
  qwen25-1.5b) MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct ;;
  qwen25-3b) MODEL_ID=Qwen/Qwen2.5-3B-Instruct ;;
  qwen25-7b) MODEL_ID=Qwen/Qwen2.5-7B-Instruct ;;
  qwen25-14b) MODEL_ID=Qwen/Qwen2.5-14B-Instruct ;;
  qwen25-coder-0.5b) MODEL_ID=Qwen/Qwen2.5-Coder-0.5B-Instruct ;;
  qwen25-coder-1.5b) MODEL_ID=Qwen/Qwen2.5-Coder-1.5B-Instruct ;;
  qwen25-coder-3b) MODEL_ID=Qwen/Qwen2.5-Coder-3B-Instruct ;;
  qwen25-coder-7b) MODEL_ID=Qwen/Qwen2.5-Coder-7B-Instruct ;;
  qwen25-coder-14b) MODEL_ID=Qwen/Qwen2.5-Coder-14B-Instruct ;;
  qwen25-coder-32b) MODEL_ID=Qwen/Qwen2.5-Coder-32B-Instruct ;;
  qwen3-4b) MODEL_ID=Qwen/Qwen3-4B ;;
  qwen3-4b-instruct) MODEL_ID=Qwen/Qwen3-4B-Instruct-2507 ;;
  qwen3-4b-think) MODEL_ID=Qwen/Qwen3-4B-Thinking-2507 ;;
  qwen3-8b) MODEL_ID=Qwen/Qwen3-8B ;;
  qwen3-8b-base) MODEL_ID=Qwen/Qwen3-8B-Base ;;
  qwen35-2b) MODEL_ID=Qwen/Qwen3.5-2B ;;
  custom) MODEL_ID=custom ;; # Set MODEL_PATH and a distinct RUN_NAME.
  *) echo "Unknown MODEL: $MODEL (see this script; use custom with MODEL_PATH)" >&2; exit 2 ;;
esac

# Resolve paths from this script so it can run from any working directory.
TASK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_ROOT="$(cd "$TASK_DIR/.." && pwd)"
REPO_ROOT="$(cd "$SRC_ROOT/.." && pwd)"
RUN_NAME="${RUN_NAME:-${MODEL}_${ADV_ESTIMATOR}}"
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/outputs/code/${RUN_NAME}}"
[[ "$RUN_DIR" = /* ]] || RUN_DIR="$PWD/$RUN_DIR"
SAVE_PATH="$RUN_DIR/checkpoints"
HF_DIR="$RUN_DIR/hf"
OUTPUT_DIR="$RUN_DIR/eval/$BENCHMARK"
SCRIPT_DIR="$TASK_DIR/leetcodedataset_train_rl"
TRAIN_FILE="${TRAIN_FILE:-$SCRIPT_DIR/data/grpo/train.parquet}"
VAL_FILE="${VAL_FILE:-$SCRIPT_DIR/data/grpo/test.parquet}"
PYTHON_BIN="$(command -v "$PYTHON_BIN")"
IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-${#GPUS[@]}}"
NNODES="${NNODES:-1}"
LOGGER="${LOGGER:-[console]}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=true
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export WANDB_DIR="$RUN_DIR"
export PYTHON_BIN REPO_ROOT OUTPUT_DIR

# Resolve user-supplied relative paths before changing the working directory.
# Defaults above are anchored to this project, so moving the checkout is safe.
for path_var in MODEL_PATH MODEL_ROOT TRAIN_FILE VAL_FILE DATA_DIR REWARD_FILE TOOL_CONFIG AGENT_LOOP_CONFIG HF_HOME HF_HUB_CACHE; do
  if [[ -n "${!path_var:-}" && "${!path_var}" != /* ]]; then
    printf -v "$path_var" '%s/%s' "$PWD" "${!path_var}"
  fi
done

resolve_model() {
  if [[ -n "${MODEL_PATH:-}" ]]; then return; fi
  local root candidate
  for root in "${MODEL_ROOT:-$REPO_ROOT/models}"; do
    candidate="$root/${MODEL_ID##*/}"
    if [[ -f "$candidate/config.json" ]]; then MODEL_PATH="$candidate"; return; fi
  done
  for root in "${HF_HUB_CACHE:-${HF_HOME:-${HOME}/.cache/huggingface}/hub}"; do
    for candidate in "$root/models--${MODEL_ID//\//--}/snapshots/"*; do
      if [[ -f "$candidate/config.json" ]]; then MODEL_PATH="$candidate"; return; fi
    done
  done
  if [[ "${DRY_RUN:-0}" == 1 ]]; then MODEL_PATH="$MODEL_ID"; return; fi
  echo "Local weights not found for $MODEL_ID. Put weights in models/${MODEL_ID##*/} or set MODEL_PATH to a local directory." >&2; exit 2
}
resolve_model
# Dry runs do not load models or write files. Check inputs and output conflicts before training.
if [[ "${DRY_RUN:-0}" != 1 ]]; then
  for path in "$MODEL_PATH" "$TRAIN_FILE" "$VAL_FILE"; do
    [[ -e "$path" ]] || { echo "Required path does not exist: $path" >&2; exit 2; }
  done
  if [[ -e "$HF_DIR" && -n "$(ls -A "$HF_DIR")" ]]; then
    echo "Weights already exist in $HF_DIR; choose another RUN_NAME or RUN_DIR." >&2; exit 2
  fi
  mkdir -p "$RUN_DIR"
fi
run() {
  printf 'Command:'; printf ' %q' "$@"; printf '\n'
  if [[ "${DRY_RUN:-0}" != 1 ]]; then "$@"; fi
}

# 2. Train with task-specific rewards and settings; trailing arguments override training parameters.
cd "$REPO_ROOT"
run "$PYTHON_BIN" -m verl.trainer.main_ppo \
  algorithm.adv_estimator="${ADV_ESTIMATOR}" \
  +algorithm.gdpo_reward_keys='["format_reward", "syntax_reward", "compile_reward", "runtime_reward","accuracy_reward","efficiency_reward","ast_similarity_reward"]' \
  algorithm.norm_adv_by_std_in_grpo=True \
  algorithm.use_kl_in_reward=False \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH:-4096}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH:-2048}" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-32}" \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF:-0.001}" \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.continuous_token_model_family="${CONTINUOUS_TOKEN_MODEL_FAMILY:-auto}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL:-0.72}" \
  actor_rollout_ref.rollout.free_cache_engine=True \
  +actor_rollout_ref.rollout.enable_sleep_mode=True \
  actor_rollout_ref.rollout.n="${ROLLOUT_N:-8}" \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}" \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  reward.num_workers="${REWARD_WORKERS:-16}" \
  reward.custom_reward_function.path="${REWARD_FILE:-${SCRIPT_DIR}/reward.py}" \
  reward.custom_reward_function.name=compute_score \
  +reward.custom_reward_function.reward_kwargs.timeout_seconds="${CODE_TIMEOUT:-5.0}" \
  +reward.custom_reward_function.reward_kwargs.memory_limit_mb="${CODE_MEMORY_MB:-1024}" \
  +reward.custom_reward_function.reward_kwargs.correctness_weight="${CORRECTNESS_WEIGHT:-0.60}" \
  +reward.custom_reward_function.reward_kwargs.format_weight="${FORMAT_WEIGHT:-0.05}" \
  +reward.custom_reward_function.reward_kwargs.syntax_weight="${SYNTAX_WEIGHT:-0.05}" \
  +reward.custom_reward_function.reward_kwargs.compile_weight="${COMPILE_WEIGHT:-0.05}" \
  +reward.custom_reward_function.reward_kwargs.runtime_weight="${RUNTIME_WEIGHT:-0.25}" \
  +reward.custom_reward_function.reward_kwargs.ast_similarity_weight="${AST_SIMILARITY_WEIGHT:-0.30}" \
  +reward.custom_reward_function.reward_kwargs.efficiency_weight="${EFFICIENCY_WEIGHT:-0.20}" \
  reward.reward_manager.name=naive \
  trainer.use_v1=True \
  trainer.balance_batch=True \
  trainer.critic_warmup=0 \
  trainer.logger="${LOGGER}" \
  trainer.project_name="${PROJECT_NAME:-leetcodedataset_grpo}" \
  trainer.experiment_name="${EXPERIMENT_NAME:-${RUN_NAME}}" \
  trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
  trainer.nnodes="${NNODES}" \
  trainer.val_before_train=False \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.default_local_dir="${SAVE_PATH}" \
  trainer.max_actor_ckpt_to_keep=2 \
  trainer.max_critic_ckpt_to_keep=2 \
  trainer.resume_mode=auto \
  trainer.save_best_checkpoint=True \
  trainer.best_checkpoint_metric=${BEST_CHECKPOINT_METRIC:-val-aux/leetcodedataset/score/mean@1} \
  trainer.best_checkpoint_mode=${BEST_CHECKPOINT_MODE:-max} \
  trainer.log_val_generations="${LOG_VAL_GENERATIONS:-10}" \
  "hydra.run.dir=${RUN_DIR}/hydra" "$@"

# 3. Convert the best validation checkpoint by default; use STEP=latest or STEP=100 to select another.
STEP="${STEP:-best}"
if [[ "$STEP" == best || "$STEP" == latest ]]; then
  TRACKER="$SAVE_PATH/${STEP}_checkpointed_iteration.txt"
  if [[ -f "$TRACKER" ]]; then
    STEP="$(<"$TRACKER")"
  elif [[ "${DRY_RUN:-0}" == 1 ]]; then
    STEP=0  # Placeholder used only to preview the conversion command.
  else
    echo "Missing checkpoint tracker: $TRACKER" >&2; exit 2
  fi
fi
[[ "$STEP" =~ ^[0-9]+$ ]] || { echo "Invalid STEP: $STEP" >&2; exit 2; }
run "$PYTHON_BIN" -m verl.model_merger merge --backend fsdp \
  --local_dir "$SAVE_PATH/global_step_${STEP}/actor" --target_dir "$HF_DIR"

# 4. Evaluate the Hugging Face weights produced by the conversion step.
export MODEL_PATH="$HF_DIR"
case "$BENCHMARK" in
  code)
    run bash "$TASK_DIR/leetcodedataset_train_rl/run_eval.sh" ;;
  humaneval|mbpp|livecodebench)
    run bash "$TASK_DIR/$BENCHMARK/run_eval.sh" \
      --model-path "$HF_DIR" --output-dir "$OUTPUT_DIR" \
      --gpus "$CUDA_VISIBLE_DEVICES" --tensor-parallel "${TENSOR_PARALLEL_SIZE:-1}" \
      --data-parallel "${DATA_PARALLEL_SIZE:-1}" ;;
esac

if [[ "${DRY_RUN:-0}" != 1 ]]; then echo "Finished. Results directory: $OUTPUT_DIR"; fi
