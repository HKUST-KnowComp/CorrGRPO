#!/usr/bin/env bash
set -euo pipefail
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"  # code_rl
PROJECT_DIR="$(cd "$BASE_DIR/../.." && pwd)"               # CorrGRPO
PYTHON_BIN="$(command -v "${PYTHON_BIN:-python3}")"
export BASE_DIR PROJECT_DIR PYTHON_BIN
export PYTHONPATH="$BASE_DIR:$BASE_DIR/livecodebench/vendor:$BASE_DIR/humaneval/vendor${PYTHONPATH:+:$PYTHONPATH}"
export LCB_CODE_GENERATION_DATASET="$BASE_DIR/livecodebench/data"
