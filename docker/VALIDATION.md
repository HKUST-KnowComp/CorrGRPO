> Historical snapshot validation. See [RETEST.md](RETEST.md) for the updated image and latest script tests.

# Container validation

Date: 2026-09-29. Host: GPU2, Linux x86-64.

## Verified image

- Tags: `corrgrpo:matched`, `corrgrpo:latest`.
- Image ID: `sha256:ae9225e16a7fd92b303db543b5ee50a8cec3ab74576218b9d6657d48f52e7df4`.
- Docker image size: 6,275,827,630 bytes.
- Bundled Python 3.12.0, PyTorch 2.10.0+cu128, vLLM 0.19.1, Transformers 5.10.4.
- Built from a copy of the user-selected tested environment. No changes to that
  source environment were needed for this packaging step.
- No host virtual environment or host site-packages were mounted for any test.
- CPU checks used a non-root numeric UID with networking disabled.

## Results

1. All 20 checks in `docker/smoke_test.py` passed inside the image: runtime imports;
   all three task launchers' syntax, three-stage command construction, real Hydra
   configurations and parquet datasets; conversion and evaluation CLIs; code
   reward checks; AgentDojo reward/tool checks; and RLLA scoring checks.
2. The final image's default user environment passed PyTorch/vLLM/scoring imports
   under a non-root numeric UID.
3. A tiny randomly initialized two-layer Qwen2 model completed a real FSDP
   forward/backward/AdamW update inside the container. Loss was 4.8757314682.
   The same container ran the project's FSDP model merger on CPU and then vLLM
   inference from the converted weights, generating four tokens.
   The observed process GPU-memory peak was 1,012 MiB on one GPU;
   PyTorch's training tensor-allocation peak was 24.15 MiB.
4. The actual HumanEval sandbox launcher evaluated the 164 bundled reference
   solutions: pass@1 = 1.0. Its network-disabled, read-only sandbox did not mount
   host Python packages. This checks the evaluation environment, not model quality.
5. Launchers passed relocated-checkout tests, including a path containing spaces,
   caller-relative model/data/output overrides, nested Docker bind-path mapping,
   and LeetCode/local-server relative-path checks.

## Limits

Full task-specific RL training, multi-GPU training, and complete benchmark suites
were not run. The optional API-Bank/ASB image dependencies resolved but that image
was not built or runtime-tested. The public vLLM-based Dockerfile passed dependency
resolution; its base-image download timed out on GPU2. The snapshot image above
is the tested distribution.

The tiny vLLM test emitted a process-group cleanup warning at exit and returned
success. These checks establish environment and entrypoint compatibility, not
training convergence or benchmark scores.
