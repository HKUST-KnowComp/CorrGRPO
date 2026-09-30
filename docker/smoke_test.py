#!/usr/bin/env python3
"""Check the portable task entrypoints without launching training workers."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "corrgrpo-src"
AGENT = SRC / "agent_security_rl/agentdojo_train_rl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true", help="Also run a small CUDA forward/backward step.")
    args = parser.parse_args()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               PYTHON_BIN=sys.executable, CUDA_VISIBLE_DEVICES="",
               PYTHONPATH=f"{ROOT}:{AGENT}:{AGENT / 'vendor'}")
    results = []

    def check(name: str, command: list[str], cwd: Path = ROOT, extra: dict | None = None) -> str:
        proc = subprocess.run(command, cwd=cwd, env=env | (extra or {}),
                              text=True, capture_output=True, timeout=120)
        results.append((name, proc.returncode == 0))
        print(f"{'PASS' if proc.returncode == 0 else 'FAIL'} {name}", flush=True)
        if proc.returncode:
            print(proc.stdout[-4000:] + proc.stderr[-4000:], flush=True)
        return proc.stdout if proc.returncode == 0 else ""

    check("runtime imports", [sys.executable, "-c",
          "import torch, vllm, flash_attn, transformers, ray, pyarrow; "
          "print(torch.__version__,vllm.__version__,transformers.__version__)"])
    with tempfile.TemporaryDirectory(prefix="corrgrpo-smoke-") as tmp:
        for task in ("toolcall_rl", "code_rl", "agent_security_rl"):
            launcher = SRC / task / "run.sh"
            check(f"{task} shell syntax", ["bash", "-n", str(launcher)])
            output = check(f"{task} command construction", ["bash", str(launcher)],
                           cwd=Path(tmp), extra={"DRY_RUN": "1"})
            commands = [shlex.split(line.removeprefix("Command:")) for line in output.splitlines()
                        if line.startswith("Command:")]
            if len(commands) != 3:
                results.append((f"{task} three pipeline stages", False))
                continue
            config_text = check(f"{task} real Hydra configuration", commands[0] + ["--cfg", "job", "--resolve"])
            config = yaml.safe_load(config_text) if config_text else {}
            logger = config.get("trainer", {}).get("logger")
            logger_ok = logger == ["console"]
            results.append((f"{task} default logger configuration", logger_ok))
            print(f"{'PASS' if logger_ok else 'FAIL'} {task} default logger configuration: {logger!r}", flush=True)
            paths = [item.split("=", 1)[1] for item in commands[0]
                     if item.startswith(("data.train_files=", "data.val_files="))]
            check(f"{task} parquet data", [sys.executable, "-c",
                  "import sys,pyarrow.parquet as p; "
                  "tables=[p.read_table(x) for x in sys.argv[1:]]; "
                  "assert len(tables)==2 and all(t.num_rows and 'prompt' in t.column_names for t in tables)", *paths])

        check("weight conversion CLI", [sys.executable, "-m", "verl.model_merger", "merge", "--help"])
        for relative in ("toolcall_rl/rlla_train_rl/rlla_test_scripts/evaluate_rlla_tool_calls.py",
                         "code_rl/leetcodedataset_train_rl/evaluate_pass_rate.py"):
            check(f"evaluation CLI: {Path(relative).name}", [sys.executable, str(SRC / relative), "--help"])
        check("code reward stages", [sys.executable, "test_reward_stages.py"],
              SRC / "code_rl/leetcodedataset_train_rl")
        check("AgentDojo reward tests", [sys.executable, "-m", "unittest", "verl_training.test_reward"], AGENT)
        check("AgentDojo tool imports", [sys.executable, "-c", "import verl_training.agentdojo_tool"])
        methods = ("test_exact_tool_call_gets_full_component_scores",
                   "test_wrong_parameter_value_is_not_all_correct",
                   "test_extra_parameter_name_is_not_all_correct",
                   "test_non_tool_sample_cannot_count_as_all_correct")
        check("RLLA scoring tests", [sys.executable, "-m", "unittest", *[
            "rlla_test_scripts.test_evaluate_rlla_tool_calls.RLLAToolCallEvaluationTest." + method
            for method in methods]], SRC / "toolcall_rl/rlla_train_rl")

    if args.gpu:
        # The user must explicitly request this stage and expose a GPU to Docker.
        gpu_code = """import torch
assert torch.cuda.is_available(), 'No CUDA device visible'
torch.cuda.set_per_process_memory_fraction(0.025)
model=torch.nn.Linear(64,64,device='cuda'); optimizer=torch.optim.AdamW(model.parameters())
x=torch.randn(4,64,device='cuda'); loss=model(x).square().mean()
loss.backward(); optimizer.step(); torch.cuda.synchronize()
assert torch.isfinite(loss)
print('CUDA optimizer step complete; tensor peak MiB:',torch.cuda.max_memory_allocated()/1024**2)
"""
        check("small CUDA optimizer step", [sys.executable, "-c", gpu_code],
              extra={"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0")})
    print(json.dumps(dict(results), indent=2))
    if not all(ok for _, ok in results):
        raise SystemExit(1)
    print("All checks passed. Full task-specific RL training has not been run.")


if __name__ == "__main__":
    main()
