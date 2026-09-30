#!/usr/bin/env python3
"""Execution reward for LeetCodeDataset and verl.

The public ``compute_score`` function matches verl's custom reward interface.
Each candidate is evaluated in a fresh, isolated-mode Python subprocess with
wall-clock, CPU, address-space, file-size, and file-descriptor limits.

This is a lightweight guard, not a security boundary equivalent to a container
or microVM. Only run rollouts produced in a trusted research environment.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import contextlib
import difflib
import io
import json
import math
import os
import re
import resource
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any


_FENCE_RE = re.compile(r"```(?:python|py)?[ \t]*\n?(.*?)```", re.IGNORECASE | re.DOTALL)


def extract_code(solution_str: str) -> tuple[str, bool]:
    """Return candidate Python code and whether it used the requested fence format."""
    text = solution_str.strip()
    fenced = _FENCE_RE.findall(text)
    if fenced:
        # Prefer a complete LeetCode class when the response contains both a
        # quoted starter block and a final implementation block.
        code = max(fenced, key=lambda item: ("class Solution" in item, len(item)))
        return code.strip(), True

    if "</think>" in text:
        text = text.rsplit("</think>", maxsplit=1)[-1].strip()
    class_start = text.find("class Solution")
    if class_start >= 0:
        text = text[class_start:]
    return text, False


def _ast_structure_tokens(code: str) -> tuple[str, ...] | None:
    """Return ordered, identifier-independent tokens encoding AST tree shape."""
    try:
        tree = ast.parse(code, filename="<ast_similarity>", mode="exec")
    except (SyntaxError, ValueError, TypeError, MemoryError):
        return None

    tokens: list[str] = []

    def visit(node: ast.AST) -> None:
        name = type(node).__name__
        tokens.append(name)
        for child in ast.iter_child_nodes(node):
            visit(child)
        tokens.append(f"/{name}")

    visit(tree)
    return tuple(tokens)


def compute_ast_similarity(candidate_code: str, reference_solution: str) -> tuple[float, float]:
    """Return structural AST similarity and whether a valid reference exists."""
    reference_text = reference_solution.strip()
    if _FENCE_RE.search(reference_text):
        reference_code, _ = extract_code(reference_text)
    else:
        # ``prepare_data.py`` stores an already-extracted pure-code field. Do
        # not run the raw-response fallback again, because that fallback
        # intentionally trims prose before ``class Solution`` and would also
        # discard legitimate imports from an already-clean reference.
        reference_code = reference_text
    reference_tokens = _ast_structure_tokens(reference_code) if reference_code else None
    if not reference_tokens:
        return 0.0, 0.0

    candidate_tokens = _ast_structure_tokens(candidate_code) if candidate_code else None
    if not candidate_tokens:
        return 0.0, 1.0

    similarity = difflib.SequenceMatcher(
        None,
        candidate_tokens,
        reference_tokens,
        autojunk=False,
    ).ratio()
    return float(max(0.0, min(1.0, similarity))), 1.0


def compute_efficiency_score(
    generated_runtime_seconds: float,
    reference_runtime_seconds: float,
    pass_score: float,
) -> tuple[float, float, float]:
    """Return clip(1 - t_g/t_r, 0, 1), applied flag, and reference flag."""
    generated_available = (
        generated_runtime_seconds > 0.0 and math.isfinite(generated_runtime_seconds)
    )
    reference_available = (
        reference_runtime_seconds > 0.0 and math.isfinite(reference_runtime_seconds)
    )
    applied = bool(pass_score == 1.0 and generated_available and reference_available)
    if not applied:
        return 0.0, 0.0, float(reference_available)
    score = max(
        0.0,
        min(1.0, 1.0 - generated_runtime_seconds / reference_runtime_seconds),
    )
    return float(score), 1.0, 1.0


def _parse_ground_truth(ground_truth: Any) -> dict[str, Any]:
    if isinstance(ground_truth, bytes):
        ground_truth = ground_truth.decode("utf-8")
    if isinstance(ground_truth, str):
        value = json.loads(ground_truth)
    elif isinstance(ground_truth, Mapping):
        value = dict(ground_truth)
    else:
        raise TypeError(f"Unsupported ground_truth type: {type(ground_truth)!r}")

    required = ("prompt", "test", "entry_point")
    missing = [key for key in required if not value.get(key)]
    if missing:
        raise ValueError(f"Verifier payload is missing fields: {missing}")
    return value


def _make_source(code: str, verifier: Mapping[str, Any]) -> str:
    return "\n\n".join(
        (
            str(verifier["prompt"]).rstrip(),
            code.rstrip(),
            str(verifier["test"]).rstrip(),
            f"check({verifier['entry_point']})",
        )
    ) + "\n"


def _run_subprocess(source: str, timeout_seconds: float, memory_limit_mb: int) -> dict[str, Any]:
    payload = json.dumps(
        {
            "source": source,
            "cpu_seconds": max(1, math.ceil(timeout_seconds)),
            "memory_limit_mb": memory_limit_mb,
        },
        ensure_ascii=False,
    )
    worker = str(Path(__file__).resolve())
    minimal_env = {
        "PYTHONHASHSEED": "0",
        "PYTHONIOENCODING": "utf-8",
        "OMP_NUM_THREADS": "1",
    }

    with tempfile.TemporaryDirectory(prefix="verl_leetcode_") as workdir:
        process = subprocess.Popen(
            [sys.executable, "-I", worker, "--worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=workdir,
            env=minimal_env,
            start_new_session=True,
        )
        try:
            stdout, _stderr = process.communicate(input=payload, timeout=timeout_seconds + 1.0)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            return {"passed": False, "status": "timeout"}

    if process.returncode != 0 or not stdout.strip():
        if process.returncode in (-signal.SIGKILL, -signal.SIGXCPU):
            status = "timeout"
        else:
            status = "worker_error"
        return {"passed": False, "status": status}

    try:
        result = json.loads(stdout.strip().splitlines()[-1])
    except json.JSONDecodeError:
        return {"passed": False, "status": "worker_error"}
    return result


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Any = None,
    timeout_seconds: float = 5.0,
    memory_limit_mb: int = 1024,
    correctness_weight: float = 0.60,
    format_weight: float = 0.05,
    syntax_weight: float = 0.10,
    compile_weight: float = 0.10,
    runtime_weight: float = 0.15,
    ast_similarity_weight: float = 0.10,
    efficiency_weight: float = 0.10,
    **_: Any,
) -> dict[str, float]:
    """Compute staged format/syntax/compile/runtime/pass rewards for verl."""
    del data_source, extra_info
    code, fenced = extract_code(solution_str)
    format_reward = float(fenced and "class Solution" in code)
    syntax_reward = 0.0
    compile_reward = 0.0
    runtime_reward = 0.0
    ast_similarity_reward = 0.0
    ast_reference_available = 0.0
    verifier: dict[str, Any] = {}
    result: dict[str, Any] = {"passed": False, "status": "empty_code"}

    # Stage 1: syntax of the extracted candidate itself. ``ast.parse`` does
    # not execute untrusted code and is distinct from compiling the complete
    # prompt + candidate + test harness below.
    if code:
        try:
            ast.parse(code, filename="<candidate>", mode="exec")
            syntax_reward = 1.0
        except (SyntaxError, ValueError, TypeError, MemoryError):
            result = {"passed": False, "status": "syntax_error"}

    # Stage 2: compile the exact source that will be run. This catches cases
    # that parse alone misses, such as a future import in an invalid position.
    try:
        verifier = _parse_ground_truth(ground_truth)
        ast_similarity_reward, ast_reference_available = compute_ast_similarity(
            code,
            str(verifier.get("reference_solution", "")),
        )
        if syntax_reward == 1.0:
            source = _make_source(code, verifier)
            compile(source, "<candidate_with_tests>", "exec")
            compile_reward = 1.0
    except Exception:
        result = {"passed": False, "status": "compile_error"}

    # Stage 3: execute only code that passed syntax and full-source compile.
    # A failed assertion is a normal test failure, not a runtime exception.
    if compile_reward == 1.0:
        try:
            result = _run_subprocess(source, float(timeout_seconds), int(memory_limit_mb))
        except Exception:
            result = {"passed": False, "status": "reward_error"}

    accuracy_reward = float(bool(result.get("passed")))
    status = str(result.get("status", "worker_error"))
    runtime_reward = float(status in ("passed", "assertion_failed"))
    timeout = float(status == "timeout")
    execution_error = float(status not in ("passed", "assertion_failed"))
    runtime_error = float(
        compile_reward == 1.0 and runtime_reward == 0.0 and timeout == 0.0
    )
    generated_runtime_seconds = float(result.get("execution_time_seconds", 0.0))
    try:
        reference_runtime_seconds = float(verifier.get("reference_runtime_seconds", 0.0))
    except (TypeError, ValueError):
        reference_runtime_seconds = 0.0
    efficiency_reward, efficiency_applied, efficiency_reference_available = (
        compute_efficiency_score(
            generated_runtime_seconds,
            reference_runtime_seconds,
            accuracy_reward,
        )
    )

    raw_weights = (
        float(format_weight),
        float(syntax_weight),
        float(compile_weight),
        float(runtime_weight),
        float(ast_similarity_weight),
        float(efficiency_weight),
        float(correctness_weight),
    )
    if any(weight < 0 for weight in raw_weights) or sum(raw_weights) <= 0:
        raise ValueError("reward weights must be non-negative and have a positive sum")
    weights = (
        raw_weights[0],
        raw_weights[1],
        raw_weights[2],
        raw_weights[3],
        raw_weights[4] * ast_reference_available,
        raw_weights[5] * efficiency_applied,
        raw_weights[6],
    )
    weighted_score = (
        weights[0] * format_reward
        + weights[1] * syntax_reward
        + weights[2] * compile_reward
        + weights[3] * runtime_reward
        + weights[4] * ast_similarity_reward
        + weights[5] * efficiency_reward
        + weights[6] * accuracy_reward
    )
    active_weight_sum = sum(weights)
    score = weighted_score
    # score = weighted_score / active_weight_sum if active_weight_sum > 0.0 else 0.0

    return {
        # used for grpo
        "score": float(score),
        
        # used for grpo_cov_coeff
        "format_reward": format_reward,
        "syntax_reward": syntax_reward,
        "compile_reward": compile_reward,
        "runtime_reward": runtime_reward,
        "accuracy_reward": accuracy_reward, # pass rate
        
        "efficiency_reward": efficiency_reward,
        "ast_similarity_reward": ast_similarity_reward,

        "format_score": format_reward,
        "syntax_score": syntax_reward,
        "compile_score": compile_reward,
        "runtime_success_score": runtime_reward,
        "ast_similarity_score": ast_similarity_reward,
        "efficiency_score": efficiency_reward,
        "pass_score": accuracy_reward,

        
        "ast_reference_available": ast_reference_available,
        "efficiency_applied": efficiency_applied,
        "efficiency_reference_available": efficiency_reference_available,
        "generated_runtime_seconds": generated_runtime_seconds,
        "reference_runtime_seconds": reference_runtime_seconds,
        "timeout": timeout,
        "runtime_error": runtime_error,
        "execution_error": execution_error,
        "num_tests": float(verifier.get("num_tests", 0)),
    }


def _blocked(*_args: Any, **_kwargs: Any) -> Any:
    raise PermissionError("operation disabled in the LeetCode reward worker")


def _apply_resource_limits(cpu_seconds: int, memory_limit_mb: int) -> None:
    memory_bytes = max(128, memory_limit_mb) * 1024 * 1024

    def set_limit(kind: int, soft: int, hard: int) -> None:
        try:
            resource.setrlimit(kind, (soft, hard))
        except (OSError, ValueError):
            # RLIMIT_AS is unsupported on some macOS/Python combinations. The
            # production target is Linux, but keep the smoke test portable.
            pass

    set_limit(resource.RLIMIT_CPU, cpu_seconds, cpu_seconds + 1)
    set_limit(resource.RLIMIT_AS, memory_bytes, memory_bytes)
    set_limit(resource.RLIMIT_FSIZE, 1024 * 1024, 1024 * 1024)
    set_limit(resource.RLIMIT_NOFILE, 32, 32)
    set_limit(resource.RLIMIT_CORE, 0, 0)


def _apply_reliability_guard() -> None:
    os.environ.clear()
    os.environ["OMP_NUM_THREADS"] = "1"

    builtins.open = _blocked
    io.open = _blocked

    for name in (
        "system",
        "popen",
        "spawnl",
        "spawnle",
        "spawnlp",
        "spawnlpe",
        "spawnv",
        "spawnve",
        "spawnvp",
        "spawnvpe",
        "fork",
        "forkpty",
        "kill",
        "killpg",
        "remove",
        "removedirs",
        "rmdir",
        "rename",
        "renames",
        "replace",
        "truncate",
        "unlink",
        "chmod",
        "chown",
        "chroot",
        "open",
        "_exit",
    ):
        if hasattr(os, name):
            setattr(os, name, _blocked)

    for name in ("rmtree", "move", "copy", "copy2", "copytree", "chown"):
        if hasattr(shutil, name):
            setattr(shutil, name, _blocked)

    subprocess.Popen = _blocked  # type: ignore[assignment]
    subprocess.run = _blocked  # type: ignore[assignment]
    subprocess.call = _blocked  # type: ignore[assignment]
    subprocess.check_call = _blocked  # type: ignore[assignment]
    subprocess.check_output = _blocked  # type: ignore[assignment]
    socket.socket = _blocked  # type: ignore[assignment]
    socket.create_connection = _blocked  # type: ignore[assignment]

    for module_name in ("ctypes", "multiprocessing", "requests", "urllib.request", "http.client"):
        sys.modules[module_name] = None


def _worker_main() -> int:
    try:
        payload = json.load(sys.stdin)
        source = str(payload["source"])
        cpu_seconds = int(payload["cpu_seconds"])
        memory_limit_mb = int(payload["memory_limit_mb"])
    except Exception:
        print(json.dumps({"passed": False, "status": "bad_payload"}))
        return 2

    _apply_resource_limits(cpu_seconds, memory_limit_mb)
    _apply_reliability_guard()
    captured_out = io.StringIO()
    captured_err = io.StringIO()

    try:
        compiled = compile(source, "<candidate>", "exec")
    except SyntaxError:
        result = {"passed": False, "status": "syntax_error", "execution_time_seconds": 0.0}
    except BaseException:
        result = {"passed": False, "status": "compile_error", "execution_time_seconds": 0.0}
    else:
        started = time.perf_counter()
        try:
            namespace: dict[str, Any] = {"__name__": "__candidate__"}
            with contextlib.redirect_stdout(captured_out), contextlib.redirect_stderr(captured_err):
                exec(compiled, namespace, namespace)
            result = {"passed": True, "status": "passed"}
        except AssertionError:
            result = {"passed": False, "status": "assertion_failed"}
        except MemoryError:
            result = {"passed": False, "status": "memory_error"}
        except BaseException:
            result = {"passed": False, "status": "runtime_error"}
        result["execution_time_seconds"] = time.perf_counter() - started

    print(json.dumps(result, separators=(",", ":")))
    return 0


def _smoke_test(dataset_path: Path, count: int) -> int:
    with dataset_path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    count = max(1, min(count, len(rows)))
    if count == 1:
        indices = [0]
    else:
        indices = sorted({round(i * (len(rows) - 1) / (count - 1)) for i in range(count)})

    results = []
    failures = 0
    for index in indices:
        row = rows[index]
        verifier = json.dumps(
            {
                "task_id": row["task_id"],
                "prompt": row["prompt"],
                "test": row["test"],
                "entry_point": row["entry_point"],
                "num_tests": len(row["input_output"]),
                "reference_solution": extract_code(row["response"])[0],
            }
        )
        canonical = compute_score("leetcodedataset", row["response"], verifier)
        failures += canonical["accuracy_reward"] != 1.0
        results.append({"index": index, "task_id": row["task_id"], "canonical": canonical})

    first = rows[indices[0]]
    first_verifier = json.dumps(
        {
            "task_id": first["task_id"],
            "prompt": first["prompt"],
            "test": first["test"],
            "entry_point": first["entry_point"],
            "num_tests": len(first["input_output"]),
            "reference_solution": extract_code(first["response"])[0],
        }
    )
    wrong = compute_score(
        "leetcodedataset",
        "```python\nclass Solution:\n    pass\n```",
        first_verifier,
    )
    failures += wrong["accuracy_reward"] != 0.0
    print(json.dumps({"canonical_samples": results, "wrong_sample": wrong, "failures": failures}, indent=2))
    return int(failures > 0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--smoke-test", type=Path)
    parser.add_argument("--smoke-test-count", type=int, default=1)
    args = parser.parse_args()
    if args.worker:
        return _worker_main()
    if args.smoke_test:
        return _smoke_test(args.smoke_test, args.smoke_test_count)
    parser.error("pass --smoke-test DATASET.jsonl")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
