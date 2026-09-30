# CorrGRPO

For the paper overview, selected results, installation, reward implementations, and BibTeX citation, see the [project README](../README.md).

Each task has one entrypoint that runs **training → FSDP-to-Hugging-Face conversion → evaluation**.
Run one of these commands from `corrgrpo-src`:

```bash
bash toolcall_rl/run.sh
bash code_rl/run.sh
bash agent_security_rl/run.sh
```

Edit the model, Python executable, GPUs, training settings, and `BENCHMARK` at the top of each script.
Each entrypoint contains its task-specific training configuration.
The scripts use `python` from the activated environment; override it with `PYTHON_BIN` if needed.
For a portable, pinned environment, follow [Docker setup](../docker/README.md).
For a local virtual environment:

```bash
source .venv/bin/activate
```

```bash
MODEL=qwen25-7b CUDA_VISIBLE_DEVICES=0,1,2,3 bash toolcall_rl/run.sh
MODEL=qwen25-coder-3b TOTAL_EPOCHS=5 bash code_rl/run.sh
DRY_RUN=1 bash agent_security_rl/run.sh  # Print all three stages without executing them.
```

The `MODEL` choices are listed in each script's case statement. Place weights in `models/<model-directory>/` at the repository root, or set `MODEL_PATH`
to a local directory. Relative overrides are resolved from the directory where you invoke the script.
All default data/output paths are based on the script location, so the checkout can be moved.
The default is one visible GPU (`CUDA_VISIBLE_DEVICES=0`); set a comma-separated list for more GPUs. Architecture support and available GPU memory depend on the environment.
CorrGRPO is the default algorithm; set `ADV_ESTIMATOR=grpo` for GRPO.
Trailing arguments apply only to training, for example:
`bash code_rl/run.sh trainer.total_training_steps=10`.

| Entrypoint | Default training/evaluation | Other BENCHMARK choices |
| --- | --- | --- |
| toolcall_rl/run.sh | RLLA | api-bank |
| code_rl/run.sh | LeetCode | humaneval, mbpp, livecodebench |
| agent_security_rl/run.sh | AgentDojo validation reward | agentdojo-official, injecagent, asb |

Outputs are written to the repository's `outputs/<task>/<model>_<algorithm>/` directory:
`checkpoints/` stores training checkpoints, `hf/` stores converted weights, and
`eval/<BENCHMARK>/` stores evaluation results. Override `RUN_NAME` or set `RUN_DIR`; a relative `RUN_DIR` is resolved from the invocation directory.
A nonempty `hf/` directory stops the pipeline before training to prevent accidental overwrites.
Conversion selects the best checkpoint by default; use `STEP=latest` or `STEP=100` to select another.

RLLA training and evaluation read `toolcall_rl/rlla_train_rl/data/train.parquet` and `test.parquet`.
Override `DATA_DIR`, `TRAIN_FILE`, or `VAL_FILE` to use other inputs.
LeetCode data lives in `code_rl/leetcodedataset_train_rl/`; AgentDojo data lives in
`agent_security_rl/agentdojo_train_rl/`. Shared source datasets used by other projects remain intact.

## Directory layout

Each benchmark keeps its data, evaluation implementation, and entrypoint in its own directory:

```text
toolcall_rl/
  rlla_train_rl/   # RL data preparation, data, and evaluation
  api_bank/
  run.sh
code_rl/
  leetcodedataset_train_rl/
  humaneval/       # data, vendor, benchmark.py, run_eval.sh
  mbpp/            # data, benchmark.py, evaluate.py, run_eval.sh
  livecodebench/   # data, vendor, benchmark.py, run_eval.sh
  _eval/          # Shared generation, sharding, and Docker utilities
  run.sh
agent_security_rl/
  agentdojo_train_rl/
  injecagent/      # data, src, metrics, run_eval.sh
  asb/            # data, aios, evaluation, metrics, run_eval.sh
  _eval/          # Shared local model server and result aggregation
  run.sh
```

The former aggregate directories and centralized datasets directories have been retired.
Task-level entrypoints dispatch directly to the selected benchmark's `run_eval.sh`.
Code evaluation retains Docker isolation for executing generated programs.
Different benchmarks may require different Python dependencies.
Model vocabularies and multilingual benchmark/test inputs retain their original contents.
