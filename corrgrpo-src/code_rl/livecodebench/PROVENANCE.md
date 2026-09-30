# Provenance

Downloaded on 2026-08-26 on `gpu1` through `https://hf-mirror.com` because the
host had no route to `huggingface.co`.

## Dataset snapshots

- `openai/openai_humaneval`:
  `7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544`
- `livecodebench/code_generation_lite`:
  `0fe84c3912ea0c4d4a78037083943e8f0c4dd505`
- `Muennighoff/mbpp`:
  `d81b8291e5998f5726ab7f35a0a557e761532aac` (downloaded 2026-08-27
  through the existing gpu1 SOCKS proxy because both direct endpoints timed out)

Dataset files are now in each benchmark's own `data/` directory. Download caches were removed during cleanup.
The current LiveCodeBench directory contains only `test6.jsonl` (v6); earlier
shards had already been removed before this relocation. Cumulative releases
require restoring their corresponding shards.

## Official evaluator revisions

- `LiveCodeBench/LiveCodeBench`:
  `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24`
- `openai/human-eval`:
  `6d43fb980f9fee3c892a914eda09951f772ad10d`

The evaluator Git repositories under `vendor/` retain their `.git` directories. `git diff`
shows the two local integration changes:

1. LiveCodeBench accepts `LCB_CODE_GENERATION_DATASET` and reads the downloaded
   JSONL shards without network access, including v1-v6 and cumulative tags.
2. HumanEval's upstream-disabled `exec` call is restored behind the explicit
   `HUMANEVAL_ALLOW_UNSAFE_EXECUTION=1` gate. The wrapper sets it only when the
   caller passes `--allow-unsafe-execution`.
