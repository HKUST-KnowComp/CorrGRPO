#!/usr/bin/env bash
# Evaluate one model: bash run_eval.sh --model-path /path/to/hf --gpus 0
set -euo pipefail
BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$BENCH_DIR/../_eval/run_local_model_eval.sh" --benchmark mbpp "$@"
