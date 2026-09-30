# mbpp

This directory contains the benchmark data (`data/`), prompt and output processing
(`benchmark.py`), and evaluation entrypoint (`run_eval.sh`).
Test scoring is implemented in `evaluate.py`; code execution uses the HumanEval executor.

```bash
bash code_rl/mbpp/run_eval.sh --model-path /path/to/hf --gpus 0
```

Run this command from `corrgrpo-src`. Shared generation, shard merging, and Docker
execution utilities are in `code_rl/_eval/`.
