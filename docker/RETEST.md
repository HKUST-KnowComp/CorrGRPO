# Docker script retest

Date: 2026-09-29. Host: GPU2. All runtime checks used the Docker image's own
Python and packages; no host Python environment was mounted.

## Result

| Check | Result |
| --- | --- |
| Three launchers: shell, real Hydra configuration, default logger type, parquet data | Passed |
| Runtime imports, conversion/evaluation CLIs, code/RLLA/AgentDojo reward checks | Passed; 23 CPU checks total |
| Model and benchmark selections | Passed; 64 command variants, including 18 models per launcher |
| HumanEval via the actual nested Docker sandbox launcher | 164 reference solutions passed; pass@1 = 1.0 |
| Tiny RLLA training command from toolcall_rl/run.sh | One real RL step completed and saved global_step_1 |
| Conversion of that RL checkpoint | Passed with the project's FSDP model merger |
| vLLM inference from converted weights | Passed; generated four tokens |
| Source/image consistency | All three updated run.sh files and smoke_test.py match byte for byte |

## Fix applied

All three launchers had a shell-quoting error in the default LOGGER setting.
Hydra parsed it as the string "[console]" rather than the list ["console"].
The training logger does not accept "[console]" as a backend name.
The defaults now use LOGGER's value if supplied, or the valid Hydra list [console].
The CPU smoke test now checks this value's type and contents for each launcher.

Both image tags were rebuilt:
- corrgrpo:matched
- corrgrpo:latest

Image ID: sha256:7d10f4f6762b8e9c8865210933a529bf3b49ba09873bf7d7e8449ef8c4718d2c
Image size: 6,275,844,531 bytes.

## Tiny RL test scope

The real training command was taken from the toolcall launcher and executed with
a fresh random two-layer Qwen2 model, a standard Qwen2 byte-level tokenizer, two
short synthetic prompts retaining the bundled RLLA reward schema, two rollouts
per prompt, and one training step. The test used one GPU, 64 prompt tokens,
8 response tokens, a 16 MiB vLLM KV cache, and eager rollout execution.
The observed total GPU process-memory peak was 3606 MiB.

The RL step ran sampling, reward calculation, log-probability calculation,
advantage calculation, the actor update path, and checkpoint saving.
Random outputs received zero task rewards, so advantages and gradients were zero.
This establishes execution compatibility, not useful learning or benchmark quality.
The generated checkpoint was merged on CPU and loaded by vLLM for four-token
generation. The overall test exited successfully.

The earlier generic-tokenizer fixture was incompatible with Qwen tokenization
under this Transformers version. It was replaced only in the temporary test
fixture; production tokenizer behavior was not changed.

## Limits and host-specific findings

- The full three task pipelines were not trained to completion. Code RL and
  Agent Security RL received configuration/data/reward/import checks, not full
  task-specific RL optimization runs.
- Other benchmark branches received command-construction checks, not full
  benchmark evaluations or remote API calls.
- GPU2's default Docker bridge failed with "Device does not exist". Tests used
  --network none with local model/data files. Host networking was not changed.
- Ray/vLLM shutdown emitted cleanup warnings, including an engine-core shutdown
  message and a multiprocess ResourceTracker warning. Training, conversion and
  final inference still returned success, and the test container was removed.
- Source-environment packages and original training data were not modified.

## Re-run CPU checks

From a machine with the updated image:

    docker run --rm --network none corrgrpo:matched python docker/smoke_test.py

The test does not launch a training job or allocate GPU memory.
