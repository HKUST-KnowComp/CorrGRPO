#!/usr/bin/env bash
# Evaluate a checkpoint with exactly the same AgentDojo loop and reward as VERL validation.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${ROOT}/../../.." && pwd)}"
PYTHON_BIN="$(command -v "${PYTHON_BIN:-python3}")"

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a merged Hugging Face model}"
VAL_FILE="${VAL_FILE:-${ROOT}/data/verl/test.parquet}"
TRAIN_FILE="${TRAIN_FILE:-${ROOT}/data/verl/train.parquet}"
GPU_IDS="${CUDA_VISIBLE_DEVICES-0}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/agentdojo/eval/$(date +%Y%m%d_%H%M%S)}"
OUTPUT_JSON=""
OUTPUT_MD=""
RESUME_PATH="${RESUME_PATH:-}"
HYDRA_OVERRIDES=()
CALLER_DIR="${PWD}"

absolute_path() {
    local path="$1"
    if [[ "${path}" == /* ]]; then
        realpath -m "${path}"
    else
        realpath -m "${CALLER_DIR}/${path}"
    fi
}

usage() {
    cat <<'EOF'
Usage: bash run_val_reward_eval.sh [options] [-- HYDRA_OVERRIDES...]

Options:
  --model-path PATH    Hugging Face model or merged checkpoint to evaluate.
  --resume-path PATH   Optional VERL global_step_* checkpoint to restore.
  --val-file PATH      Validation parquet (default: data/verl/test.parquet).
  --output-dir DIR     Directory for samples, logs, JSON, and Markdown.
  --output-json PATH   JSON result path (default: OUTPUT_DIR/metrics.json).
  --output-md PATH     Markdown result path (default: OUTPUT_DIR/metrics.md).
  --gpus IDS           CUDA device list (default: 0).
  -h, --help           Show this help.

Example:
  bash run_val_reward_eval.sh \
    --model-path /path/to/merged_hf_checkpoint \
    --gpus 4,5,6,7 \
    --output-dir runs/eval_step_100
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --model-path=*) MODEL_PATH="${1#*=}"; shift ;;
        --resume-path) RESUME_PATH="$2"; shift 2 ;;
        --resume-path=*) RESUME_PATH="${1#*=}"; shift ;;
        --val-file) VAL_FILE="$2"; shift 2 ;;
        --val-file=*) VAL_FILE="${1#*=}"; shift ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --output-dir=*) OUTPUT_DIR="${1#*=}"; shift ;;
        --output-json) OUTPUT_JSON="$2"; shift 2 ;;
        --output-json=*) OUTPUT_JSON="${1#*=}"; shift ;;
        --output-md) OUTPUT_MD="$2"; shift 2 ;;
        --output-md=*) OUTPUT_MD="${1#*=}"; shift ;;
        --gpus) GPU_IDS="$2"; shift 2 ;;
        --gpus=*) GPU_IDS="${1#*=}"; shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; HYDRA_OVERRIDES+=("$@"); break ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

MODEL_PATH="$(absolute_path "${MODEL_PATH}")"
VAL_FILE="$(absolute_path "${VAL_FILE}")"
TRAIN_FILE="$(absolute_path "${TRAIN_FILE}")"
OUTPUT_DIR="$(absolute_path "${OUTPUT_DIR}")"
[[ -n "${RESUME_PATH}" ]] && RESUME_PATH="$(absolute_path "${RESUME_PATH}")"

OUTPUT_JSON="${OUTPUT_JSON:-${OUTPUT_DIR}/metrics.json}"
OUTPUT_MD="${OUTPUT_MD:-${OUTPUT_DIR}/metrics.md}"
OUTPUT_JSON="$(absolute_path "${OUTPUT_JSON}")"
OUTPUT_MD="$(absolute_path "${OUTPUT_MD}")"
SAMPLE_DIR="${OUTPUT_DIR}/samples"
LOG_FILE="${OUTPUT_DIR}/eval.log"

IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS}"
NUM_GPUS="${#GPU_LIST[@]}"
if (( NUM_GPUS == 0 )); then
    echo "--gpus must contain at least one CUDA device." >&2
    exit 2
fi

for path in "${PYTHON_BIN}" "${MODEL_PATH}" "${TRAIN_FILE}" "${VAL_FILE}" \
    "${ROOT}/verl_training/reward.py" \
    "${ROOT}/verl_training/tool_config.yaml" \
    "${ROOT}/verl_training/agent_loop_config.yaml"; do
    if [[ ! -e "${path}" ]]; then
        echo "Required path does not exist: ${path}" >&2
        exit 2
    fi
done
if [[ -n "${RESUME_PATH}" && ! -d "${RESUME_PATH}" ]]; then
    echo "VERL checkpoint does not exist: ${RESUME_PATH}" >&2
    exit 2
fi
if [[ -d "${SAMPLE_DIR}" ]] && compgen -G "${SAMPLE_DIR}/*.jsonl" >/dev/null; then
    echo "Validation samples already exist in ${SAMPLE_DIR}. Choose a new --output-dir." >&2
    exit 2
fi

mkdir -p "${SAMPLE_DIR}" "$(dirname "${OUTPUT_JSON}")" "$(dirname "${OUTPUT_MD}")"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PYTHONPATH="${ROOT}:${ROOT}/vendor:${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export VLLM_USE_V1=1

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-12288}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
MAX_TURNS="${MAX_TURNS:-8}"
TOOL_RESPONSE_LENGTH="${TOOL_RESPONSE_LENGTH:-2048}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}"
ROLLOUT_TP="${ROLLOUT_TP:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"
REWARD_WORKERS="${REWARD_WORKERS:-16}"

RESUME_ARGS=(trainer.resume_mode=disable)
if [[ -n "${RESUME_PATH}" ]]; then
    RESUME_ARGS=(trainer.resume_mode=resume_path trainer.resume_from_path="${RESUME_PATH}")
fi

echo "Model:       ${MODEL_PATH}"
[[ -n "${RESUME_PATH}" ]] && echo "VERL ckpt:   ${RESUME_PATH}"
echo "Validation:  ${VAL_FILE}"
echo "GPUs:        ${GPU_IDS}"
echo "Output JSON: ${OUTPUT_JSON}"
echo "Output MD:   ${OUTPUT_MD}"
echo "Samples:     ${SAMPLE_DIR}"

VERL_ARGS=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="${TRAIN_FILE}"
    data.val_files="${VAL_FILE}"
    data.return_raw_chat=True
    data.train_batch_size="${NUM_GPUS}"
    data.val_batch_size="${VAL_BATCH_SIZE}"
    data.max_prompt_length="${MAX_PROMPT_LENGTH}"
    data.max_response_length="${MAX_RESPONSE_LENGTH}"
    data.filter_overlong_prompts=True
    data.truncation=error
    +data.apply_chat_template_kwargs.enable_thinking=False
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.actor.ppo_mini_batch_size="${NUM_GPUS}"
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}"
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}"
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}"
    actor_rollout_ref.rollout.n=2
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.val_kwargs.temperature=0
    actor_rollout_ref.rollout.val_kwargs.do_sample=False
    actor_rollout_ref.rollout.calculate_log_probs=False
    actor_rollout_ref.rollout.free_cache_engine=True
    +actor_rollout_ref.rollout.enable_sleep_mode=True
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.multi_turn.enable=True
    actor_rollout_ref.rollout.multi_turn.max_user_turns="${MAX_TURNS}"
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns="${MAX_TURNS}"
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length="${TOOL_RESPONSE_LENGTH}"
    actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=middle
    actor_rollout_ref.rollout.multi_turn.use_inference_chat_template=True
    actor_rollout_ref.rollout.multi_turn.tool_config_path="${ROOT}/verl_training/tool_config.yaml"
    actor_rollout_ref.rollout.multi_turn.format=hermes
    actor_rollout_ref.rollout.agent.default_agent_loop=agentdojo_agent
    actor_rollout_ref.rollout.agent.agent_loop_config_path="${ROOT}/verl_training/agent_loop_config.yaml"
    reward.num_workers="${REWARD_WORKERS}"
    reward.custom_reward_function.path="${ROOT}/verl_training/reward.py"
    reward.custom_reward_function.name=compute_score
    +reward.custom_reward_function.reward_kwargs.utility_weight=1.0
    +reward.custom_reward_function.reward_kwargs.safety_weight=1.0
    +reward.custom_reward_function.reward_kwargs.invalid_reward=-0.0
    reward.reward_manager.name=naive
    trainer.use_v1=True
    trainer.logger='["console"]'
    trainer.project_name=agentdojo_val_reward_eval
    trainer.experiment_name=validation_only
    trainer.n_gpus_per_node="${NUM_GPUS}"
    trainer.nnodes=1
    trainer.val_before_train=True
    trainer.val_only=True
    "hydra.run.dir=${OUTPUT_DIR}/hydra"
    trainer.test_freq=-1
    trainer.save_freq=-1
    trainer.total_epochs=1
    trainer.validation_data_dir="${SAMPLE_DIR}"
    trainer.default_local_dir="${OUTPUT_DIR}/unused_checkpoints"
    "${RESUME_ARGS[@]}"
    "${HYDRA_OVERRIDES[@]}"
)

cd "${REPO_ROOT}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    exec "${PYTHON_BIN}" -m verl.trainer.main_ppo --cfg job "${VERL_ARGS[@]}"
fi

"${PYTHON_BIN}" -m verl.trainer.main_ppo "${VERL_ARGS[@]}" 2>&1 | tee "${LOG_FILE}"

"${PYTHON_BIN}" "${ROOT}/verl_training/summarize_val_reward_eval.py" \
    --input "${SAMPLE_DIR}" \
    --output-json "${OUTPUT_JSON}" \
    --output-md "${OUTPUT_MD}" \
    --model "${RESUME_PATH:-${MODEL_PATH}}"
