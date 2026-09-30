#!/usr/bin/env python3
"""Generate solutions for LeetCodeDataset test and report execution pass rates.

The evaluator uses vLLM for batched generation and reuses ``reward.py`` for
the exact official test execution.  It writes one JSON object per problem and
a separate ``*.metrics.json`` summary.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from reward import compute_score, extract_code


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = SCRIPT_DIR / "raw" / "LeetCodeDataset-test.jsonl"
DEFAULT_REFERENCE_RUNTIMES = SCRIPT_DIR / "data" / "reference_runtimes.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a Hugging Face model on the LeetCodeDataset test split."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--reference-runtimes",
        type=Path,
        default=DEFAULT_REFERENCE_RUNTIMES,
    )
    parser.add_argument("--model", help="Merged Hugging Face model/tokenizer directory.")
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=1, help="Completions per problem.")
    parser.add_argument("--pass-k", default="1", help="Comma-separated k values, e.g. 1,4,8.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument(
        "--data-parallel-size",
        type=int,
        default=1,
        help="Number of replicated vLLM engines used to split evaluation requests.",
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--score-workers", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
    parser.add_argument("--memory-limit-mb", type=int, default=1024)
    parser.add_argument("--limit", type=int, help="Evaluate only the first N problems.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument(
        "--reference-solutions",
        action="store_true",
        help="Score dataset reference responses instead of loading a model (smoke test only).",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.reference_solutions and not args.model:
        parser.error("--model is required unless --reference-solutions is set")
    if args.num_samples < 1:
        parser.error("--num-samples must be >= 1")
    if (
        args.batch_size < 1
        or args.score_workers < 1
        or args.tensor_parallel_size < 1
        or args.data_parallel_size < 1
    ):
        parser.error(
            "--batch-size, --score-workers, --tensor-parallel-size, and "
            "--data-parallel-size must be >= 1"
        )
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    if args.temperature < 0:
        parser.error("--temperature must be >= 0")
    if args.num_samples > 1 and args.temperature == 0:
        parser.error("set --temperature > 0 when --num-samples > 1")
    return args


def load_rows(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    if not rows:
        raise ValueError(f"No records found in {path}")

    required = {"task_id", "query", "prompt", "test", "entry_point", "input_output"}
    for index, row in enumerate(rows):
        missing = sorted(required.difference(row))
        if missing:
            raise ValueError(f"Dataset row {index} is missing fields: {missing}")
    return rows


def attach_reference_runtimes(
    rows: list[dict[str, Any]], dataset_path: Path, runtime_path: Path
) -> None:
    if not runtime_path.exists():
        return
    split = "test" if "test" in dataset_path.name.lower() else "train"
    runtimes: dict[tuple[str, str], float] = {}
    with runtime_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            runtime = record.get("reference_runtime_seconds")
            if runtime is not None and float(runtime) > 0.0:
                runtimes[(str(record["split"]), str(record["task_id"]))] = float(runtime)
    for row in rows:
        row["_reference_runtime_seconds"] = runtimes.get(
            (split, str(row["task_id"]))
        )


def parse_k_values(raw: str, num_samples: int) -> list[int]:
    try:
        values = sorted({int(item.strip()) for item in raw.split(",") if item.strip()})
    except ValueError as exc:
        raise ValueError(f"Invalid --pass-k value: {raw!r}") from exc
    if not values or any(value < 1 for value in values):
        raise ValueError("--pass-k must contain positive integers")
    too_large = [value for value in values if value > num_samples]
    if too_large:
        raise ValueError(
            f"pass@k requires k <= --num-samples ({num_samples}); invalid values: {too_large}"
        )
    return values


def make_verifier(row: dict[str, Any]) -> dict[str, Any]:
    reference_solution, _ = extract_code(str(row.get("response", "")))
    return {
        "task_id": row["task_id"],
        "prompt": row["prompt"],
        "test": row["test"],
        "entry_point": row["entry_point"],
        "num_tests": len(row["input_output"]),
        "reference_solution": reference_solution,
        "reference_runtime_seconds": row.get("_reference_runtime_seconds"),
    }


def score_one(
    row: dict[str, Any], candidate: str, timeout_seconds: float, memory_limit_mb: int
) -> dict[str, float]:
    return compute_score(
        data_source="leetcodedataset",
        solution_str=candidate,
        ground_truth=make_verifier(row),
        timeout_seconds=timeout_seconds,
        memory_limit_mb=memory_limit_mb,
    )


def render_prompts(tokenizer: Any, rows: list[dict[str, Any]]) -> list[str]:
    prompts = []
    for row in rows:
        messages = [{"role": "user", "content": row["query"]}]
        if getattr(tokenizer, "chat_template", None):
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = row["query"]
        prompts.append(prompt)
    return prompts


def load_tokenizer(args: argparse.Namespace) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required for model evaluation") from exc

    return AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )


def build_generator(args: argparse.Namespace, rank: int = 0) -> tuple[Any, Any]:
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError("vLLM is required for model evaluation") from exc

    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=args.enable_prefix_caching,
        seed=args.seed + rank,
    )
    sampling_params = SamplingParams(
        n=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed + rank,
    )
    return llm, sampling_params


def generate_batches(
    llm: Any,
    sampling_params: Any,
    prompts: list[str],
    batch_size: int,
) -> list[list[str]]:
    candidates: list[list[str]] = []
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        generated = llm.generate(batch_prompts, sampling_params, use_tqdm=True)
        candidates.extend([[sample.text for sample in item.outputs] for item in generated])
    return candidates


def generation_config(args: argparse.Namespace) -> dict[str, Any]:
    names = (
        "model",
        "num_samples",
        "temperature",
        "top_p",
        "max_tokens",
        "seed",
        "batch_size",
        "tensor_parallel_size",
        "dtype",
        "max_model_len",
        "gpu_memory_utilization",
        "trust_remote_code",
        "enforce_eager",
        "enable_prefix_caching",
    )
    return {name: getattr(args, name) for name in names}


def visible_gpu_groups(data_parallel_size: int, tensor_parallel_size: int) -> list[list[str]]:
    total_gpus = data_parallel_size * tensor_parallel_size
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    devices = [item.strip() for item in visible.split(",") if item.strip()]
    if not devices:
        devices = [str(index) for index in range(total_gpus)]
    if len(devices) < total_gpus:
        raise ValueError(
            f"DP={data_parallel_size} and TP={tensor_parallel_size} require {total_gpus} "
            f"visible GPUs, but CUDA_VISIBLE_DEVICES provides {len(devices)}: {visible!r}"
        )
    devices = devices[:total_gpus]
    return [
        devices[rank * tensor_parallel_size : (rank + 1) * tensor_parallel_size]
        for rank in range(data_parallel_size)
    ]


def split_indices(num_items: int, num_groups: int) -> list[list[int]]:
    floor, remainder = divmod(num_items, num_groups)
    groups = []
    start = 0
    for rank in range(num_groups):
        size = floor + int(rank < remainder)
        groups.append(list(range(start, start + size)))
        start += size
    return groups


def data_parallel_worker(
    rank: int,
    gpu_devices: list[str],
    indices: list[int],
    prompts: list[str],
    config: dict[str, Any],
    result_path: str,
) -> None:
    # This target runs under the multiprocessing "spawn" context. Restrict
    # devices before importing vLLM so each independent replica owns exactly
    # one TP group and dense models work without vLLM's coordinated MoE DP.
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_devices)
    worker_args = argparse.Namespace(**config)
    print(
        f"DP rank {rank}: GPUs={os.environ['CUDA_VISIBLE_DEVICES']}, prompts={len(prompts)}",
        flush=True,
    )
    llm, sampling_params = build_generator(worker_args, rank=rank)
    candidates = generate_batches(llm, sampling_params, prompts, worker_args.batch_size)
    payload = [
        {"index": index, "candidates": row_candidates}
        for index, row_candidates in zip(indices, candidates, strict=True)
    ]
    Path(result_path).write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )


def generate_all_candidates(
    rows: list[dict[str, Any]], args: argparse.Namespace
) -> list[list[str]]:
    tokenizer = load_tokenizer(args)
    prompts = render_prompts(tokenizer, rows)
    if args.data_parallel_size == 1:
        llm, sampling_params = build_generator(args)
        return generate_batches(llm, sampling_params, prompts, args.batch_size)

    if args.data_parallel_size > len(prompts):
        raise ValueError(
            f"--data-parallel-size ({args.data_parallel_size}) cannot exceed the "
            f"number of evaluated problems ({len(prompts)})"
        )

    import multiprocessing

    gpu_groups = visible_gpu_groups(args.data_parallel_size, args.tensor_parallel_size)
    index_groups = split_indices(len(prompts), args.data_parallel_size)
    config = generation_config(args)
    context = multiprocessing.get_context("spawn")
    ordered: list[list[str] | None] = [None] * len(prompts)

    with tempfile.TemporaryDirectory(prefix="leetcodedataset_dp_") as workdir:
        processes = []
        result_paths = []
        for rank, (gpu_devices, indices) in enumerate(zip(gpu_groups, index_groups, strict=True)):
            result_path = str(Path(workdir) / f"rank_{rank}.json")
            process = context.Process(
                target=data_parallel_worker,
                args=(
                    rank,
                    gpu_devices,
                    indices,
                    [prompts[index] for index in indices],
                    config,
                    result_path,
                ),
            )
            process.start()
            processes.append(process)
            result_paths.append(result_path)

        try:
            for process in processes:
                process.join()
        except BaseException:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join()
            raise

        failures = [
            f"rank={rank}, pid={process.pid}, exitcode={process.exitcode}"
            for rank, process in enumerate(processes)
            if process.exitcode != 0
        ]
        if failures:
            raise RuntimeError("Data-parallel generation failed: " + "; ".join(failures))

        for result_path in result_paths:
            records = json.loads(Path(result_path).read_text(encoding="utf-8"))
            for record in records:
                ordered[int(record["index"])] = list(record["candidates"])

    if any(candidates is None for candidates in ordered):
        raise RuntimeError("Data-parallel generation returned incomplete results")
    return [candidates for candidates in ordered if candidates is not None]


def estimate_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    """Unbiased pass@k estimator used by HumanEval."""
    if num_correct <= 0:
        return 0.0
    if num_samples - num_correct < k:
        return 1.0
    failure_probability = 1.0
    for index in range(k):
        failure_probability *= (num_samples - num_correct - index) / (num_samples - index)
    return 1.0 - failure_probability


def metrics_path(output_file: Path) -> Path:
    if output_file.suffix:
        return output_file.with_suffix(".metrics.json")
    return Path(str(output_file) + ".metrics.json")


def main() -> int:
    args = parse_args()
    rows = load_rows(args.dataset, args.limit)
    attach_reference_runtimes(rows, args.dataset, args.reference_runtimes)
    k_values = parse_k_values(args.pass_k, args.num_samples)

    output_file = args.output_file.resolve()
    summary_file = metrics_path(output_file)
    if not args.overwrite:
        existing = [path for path in (output_file, summary_file) if path.exists()]
        if existing:
            raise FileExistsError(
                "Output already exists; pass --overwrite to replace it: "
                + ", ".join(str(path) for path in existing)
            )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    partial_file = Path(str(output_file) + ".partial")

    if args.reference_solutions:
        all_candidates = [[row["response"]] * args.num_samples for row in rows]
    else:
        all_candidates = generate_all_candidates(rows, args)
    if any(len(items) != args.num_samples for items in all_candidates):
        raise RuntimeError("vLLM returned an unexpected number of samples")

    task_passes: list[list[bool]] = []
    timeout_count = 0
    execution_error_count = 0
    runtime_error_count = 0
    format_pass_count = 0
    syntax_pass_count = 0
    compile_pass_count = 0
    runtime_success_count = 0
    ast_reference_count = 0
    ast_similarity_sum = 0.0
    efficiency_reference_count = 0
    efficiency_applied_count = 0
    efficiency_score_sum = 0.0
    reward_score_sum = 0.0

    try:
        with partial_file.open("w", encoding="utf-8") as output_handle:
            for start in range(0, len(rows), args.batch_size):
                batch = rows[start : start + args.batch_size]
                candidates = all_candidates[start : start + args.batch_size]

                score_jobs = [
                    (row, candidate)
                    for row, row_candidates in zip(batch, candidates, strict=True)
                    for candidate in row_candidates
                ]
                with ThreadPoolExecutor(max_workers=args.score_workers) as pool:
                    flat_rewards = list(
                        pool.map(
                            lambda job: score_one(
                                job[0], job[1], args.timeout_seconds, args.memory_limit_mb
                            ),
                            score_jobs,
                        )
                    )

                offset = 0
                for local_index, (row, row_candidates) in enumerate(
                    zip(batch, candidates, strict=True)
                ):
                    rewards = flat_rewards[offset : offset + args.num_samples]
                    offset += args.num_samples
                    passed = [reward["accuracy_reward"] == 1.0 for reward in rewards]
                    task_passes.append(passed)
                    timeout_count += sum(int(reward["timeout"]) for reward in rewards)
                    execution_error_count += sum(
                        int(reward["execution_error"]) for reward in rewards
                    )
                    runtime_error_count += sum(
                        int(reward["runtime_error"]) for reward in rewards
                    )
                    format_pass_count += sum(
                        int(reward["format_score"] == 1.0) for reward in rewards
                    )
                    syntax_pass_count += sum(
                        int(reward["syntax_score"] == 1.0) for reward in rewards
                    )
                    compile_pass_count += sum(
                        int(reward["compile_score"] == 1.0) for reward in rewards
                    )
                    runtime_success_count += sum(
                        int(reward["runtime_success_score"] == 1.0)
                        for reward in rewards
                    )
                    ast_reference_count += sum(
                        int(reward["ast_reference_available"] == 1.0)
                        for reward in rewards
                    )
                    ast_similarity_sum += sum(
                        float(reward["ast_similarity_score"]) for reward in rewards
                    )
                    efficiency_reference_count += sum(
                        int(reward["efficiency_reference_available"] == 1.0)
                        for reward in rewards
                    )
                    efficiency_applied_count += sum(
                        int(reward["efficiency_applied"] == 1.0) for reward in rewards
                    )
                    efficiency_score_sum += sum(
                        float(reward["efficiency_score"])
                        for reward in rewards
                        if reward["efficiency_applied"] == 1.0
                    )
                    reward_score_sum += sum(float(reward["score"]) for reward in rewards)

                    record = {
                        "dataset_index": start + local_index,
                        "task_id": row["task_id"],
                        "question_id": row.get("question_id"),
                        "difficulty": row.get("difficulty"),
                        "num_tests": len(row["input_output"]),
                        "outputs": row_candidates,
                        "passed": passed,
                        "num_correct": sum(passed),
                        "rewards": rewards,
                    }
                    output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                output_handle.flush()
                print(f"Scored {min(start + len(batch), len(rows))}/{len(rows)} problems", flush=True)

        os.replace(partial_file, output_file)
    except BaseException:
        print(f"Partial results retained at: {partial_file}", file=sys.stderr)
        raise

    num_tasks = len(task_passes)
    total_samples = num_tasks * args.num_samples
    total_correct = sum(sum(passed) for passed in task_passes)
    pass_at_k = {
        f"pass@{k}": sum(
            estimate_pass_at_k(args.num_samples, sum(passed), k) for passed in task_passes
        )
        / num_tasks
        for k in k_values
    }
    summary: dict[str, Any] = {
        "dataset": str(args.dataset.resolve()),
        "model": "dataset_reference_solutions" if args.reference_solutions else args.model,
        "num_tasks": num_tasks,
        "num_samples_per_task": args.num_samples,
        "total_samples": total_samples,
        "total_correct": total_correct,
        **pass_at_k,
        "first_sample_pass_rate": sum(passed[0] for passed in task_passes) / num_tasks,
        "pass_rate": total_correct / total_samples,
        "sample_pass_rate": total_correct / total_samples,
        "any_sample_task_pass_rate": sum(any(passed) for passed in task_passes) / num_tasks,
        "format_pass_rate": format_pass_count / total_samples,
        "syntax_pass_rate": syntax_pass_count / total_samples,
        "compile_pass_rate": compile_pass_count / total_samples,
        "runtime_success_rate": runtime_success_count / total_samples,
        "ast_similarity_score": (
            ast_similarity_sum / ast_reference_count if ast_reference_count else 0.0
        ),
        "ast_reference_available_rate": ast_reference_count / total_samples,
        "efficiency_score": (
            efficiency_score_sum / efficiency_applied_count
            if efficiency_applied_count
            else 0.0
        ),
        "efficiency_applied_rate": efficiency_applied_count / total_samples,
        "efficiency_reference_available_rate": efficiency_reference_count / total_samples,
        "average_reward_score": reward_score_sum / total_samples,
        "component_scores": {
            "format_score": format_pass_count / total_samples,
            "syntax_score": syntax_pass_count / total_samples,
            "compile_score": compile_pass_count / total_samples,
            "runtime_success_score": runtime_success_count / total_samples,
            "ast_similarity_score": (
                ast_similarity_sum / ast_reference_count if ast_reference_count else 0.0
            ),
            "efficiency_score": (
                efficiency_score_sum / efficiency_applied_count
                if efficiency_applied_count
                else 0.0
            ),
            "pass_score": total_correct / total_samples,
        },
        "timeout_count": timeout_count,
        "runtime_error_count": runtime_error_count,
        "execution_error_count": execution_error_count,
        "generation": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "tensor_parallel_size": args.tensor_parallel_size,
            "data_parallel_size": args.data_parallel_size,
            "total_gpus": args.tensor_parallel_size * args.data_parallel_size,
        },
        "output_file": str(output_file),
    }
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Per-problem results: {output_file}")
    print(f"Metrics: {summary_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
