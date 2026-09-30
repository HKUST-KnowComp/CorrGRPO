# injecagent evaluation

This directory contains the benchmark source, `data/`, evaluation entrypoint, and metric calculations.

```bash
LOCAL_MODEL=/path/to/hf PYTHON_BIN=/path/to/python bash agent_security_rl/injecagent/run_eval.sh
```

Run this command from `corrgrpo-src`. The task-level `run.sh` can also train, convert
weights, and evaluate this benchmark. Shared model serving and result aggregation
utilities are in `../_eval/`. All input data is stored locally in this benchmark directory.
