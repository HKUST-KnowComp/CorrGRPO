#!/usr/bin/env python3
"""Generate HumanEval, MBPP, and LiveCodeBench answers with local vLLM."""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from _eval.work_item import WorkItem
from humaneval.benchmark import load_humaneval, normalize_humaneval
from mbpp.benchmark import load_mbpp, normalize_mbpp
from livecodebench.benchmark import load_lcb, normalize_lcb


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_checkpoint(path: Path, samples: int) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            codes = record.get("codes", [])
            if record.get("status") == "ok" and len(codes) == samples and all(codes):
                completed[str(record["id"])] = record
    return completed


def write_final_outputs(
    benchmark: str,
    items: list[WorkItem],
    records: dict[str, dict[str, Any]],
    prefix: Path,
) -> Path:
    missing = [item.item_id for item in items if item.item_id not in records]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} generations; first: {missing[:5]}")
    if benchmark in {"humaneval", "mbpp"}:
        path = prefix.with_suffix(".samples.jsonl")
        with path.open("w", encoding="utf-8") as handle:
            for item in items:
                for code in records[item.item_id]["codes"]:
                    json.dump(
                        {
                            "task_id": (
                                int(item.item_id) if benchmark == "mbpp" else item.item_id
                            ),
                            "completion": code,
                        },
                        handle,
                        ensure_ascii=False,
                    )
                    handle.write("\n")
    else:
        path = prefix.with_suffix(".custom_outputs.json")
        payload = [
            {"question_id": item.item_id, "code_list": records[item.item_id]["codes"]}
            for item in items
        ]
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    return path


def generate_benchmark(
    *,
    name: str,
    version: str,
    mbpp_subset: str,
    items: list[WorkItem],
    llm: LLM,
    sampling: SamplingParams,
    output_dir: Path,
    batch_size: int,
    samples: int,
    tensor_parallel_size: int,
    data_parallel_size: int,
    shard_index: int,
    num_shards: int,
) -> Path:
    suffix = (
        "humaneval"
        if name == "humaneval"
        else (f"lcb_{version}" if name == "livecodebench" else f"mbpp_{mbpp_subset}")
    )
    prefix = output_dir / suffix
    checkpoint = prefix.with_suffix(".generations.jsonl")
    completed = load_checkpoint(checkpoint, samples)
    pending = [item for item in items if item.item_id not in completed]
    print(
        f"benchmark={name} version={version} total={len(items)} "
        f"completed={len(completed)} pending={len(pending)}",
        flush=True,
    )

    with checkpoint.open("a", encoding="utf-8") as handle:
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            request_outputs = llm.generate([item.prompt for item in batch], sampling)
            for item, request_output in zip(batch, request_outputs, strict=True):
                raw_outputs = [candidate.text for candidate in request_output.outputs]
                try:
                    if name == "humaneval":
                        codes = [
                            normalize_humaneval(raw, item.source_prompt, item.entry_point)
                            for raw in raw_outputs
                        ]
                    elif name == "livecodebench":
                        codes = [normalize_lcb(raw) for raw in raw_outputs]
                    else:
                        codes = [normalize_mbpp(raw) for raw in raw_outputs]
                    record: dict[str, Any] = {
                        "id": item.item_id,
                        "status": "ok",
                        "raw_outputs": raw_outputs,
                        "codes": codes,
                        "prompt_tokens": len(request_output.prompt_token_ids),
                        "completion_tokens": [len(out.token_ids) for out in request_output.outputs],
                        "finish_reasons": [out.finish_reason for out in request_output.outputs],
                        "completed_at": utc_now(),
                    }
                    completed[item.item_id] = record
                except Exception as exc:
                    record = {
                        "id": item.item_id,
                        "status": "error",
                        "raw_outputs": raw_outputs,
                        "error": f"{type(exc).__name__}: {exc}",
                        "completed_at": utc_now(),
                    }
                json.dump(record, handle, ensure_ascii=False)
                handle.write("\n")
                handle.flush()
            print(
                f"[{min(start + len(batch), len(pending))}/{len(pending)} pending] "
                f"checkpoint={checkpoint}",
                flush=True,
            )

    final_path = write_final_outputs(name, items, completed, prefix)
    prompt_tokens = sum(int(record.get("prompt_tokens", 0)) for record in completed.values())
    completion_tokens = sum(
        sum(int(value) for value in record.get("completion_tokens", []))
        for record in completed.values()
    )
    summary = {
        "benchmark": name,
        "version": version if name == "livecodebench" else None,
        "subset": mbpp_subset if name == "mbpp" else None,
        "count": len(items),
        "samples_per_problem": samples,
        "tensor_parallel_size": tensor_parallel_size,
        "data_parallel_size": data_parallel_size,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "checkpoint": str(checkpoint),
        "final_output": str(final_path),
        "completed_at": utc_now(),
    }
    with prefix.with_suffix(".summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return final_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        choices=["all", "humaneval", "livecodebench", "mbpp"],
        default="all",
        help="all runs HumanEval, MBPP test, and LiveCodeBench.",
    )
    parser.add_argument("--version", default="v6")
    parser.add_argument(
        "--mbpp-subset", choices=["test", "full", "sanitized"], default="test"
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--data-parallel-size", type=int, default=1)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--samples-per-problem", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    if not re.fullmatch(r"(?:release_)?v[1-6]", args.version):
        parser.error("version must be v1..v6 or release_v1..release_v6")
    if args.temperature == 0 and args.samples_per_problem != 1:
        parser.error("temperature=0 requires samples-per-problem=1")
    if min(
        args.tensor_parallel_size,
        args.data_parallel_size,
        args.num_shards,
        args.max_tokens,
        args.batch_size,
        args.samples_per_problem,
    ) < 1:
        parser.error("parallel sizes, token count, batch size, and sample count must be positive")
    if args.data_parallel_size != 1:
        parser.error(
            "generate_local_eval.py runs one replica; use run_local_model_eval.sh "
            "--data-parallel N to launch N isolated workers"
        )
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("shard-index must be in [0, num-shards)")
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("gpu-memory-utilization must be between 0 and 1")

    base = args.base_dir.resolve()
    model_path = args.model_path.resolve()
    if not model_path.is_dir():
        parser.error(f"model path not found: {model_path}")
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices:
        visible_count = len([value for value in visible_devices.split(",") if value.strip()])
        required_count = args.tensor_parallel_size
        if visible_count < required_count:
            parser.error(
                f"{required_count} GPUs are required by data-parallel-size="
                f"{args.data_parallel_size} and tensor-parallel-size="
                f"{args.tensor_parallel_size}, but CUDA_VISIBLE_DEVICES exposes "
                f"only {visible_count}"
            )
    os.environ.setdefault(
        "LCB_CODE_GENERATION_DATASET",
        str(base / "livecodebench" / "data"),
    )
    output_dir = (args.output_dir or base.parent.parent / "outputs" / "code" / model_path.name / "eval").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    benchmark_items: list[tuple[str, list[WorkItem]]] = []
    if args.benchmark in {"all", "humaneval"}:
        benchmark_items.append(("humaneval", load_humaneval(base, tokenizer)))
    if args.benchmark in {"all", "livecodebench"}:
        benchmark_items.append(("livecodebench", load_lcb(args.version, base)))
    if args.benchmark in {"all", "mbpp"}:
        benchmark_items.append(("mbpp", load_mbpp(args.mbpp_subset, base, tokenizer)))
    if args.limit is not None:
        benchmark_items = [(name, items[: args.limit]) for name, items in benchmark_items]
    benchmark_items = [
        (name, items[args.shard_index :: args.num_shards])
        for name, items in benchmark_items
    ]

    print(
        f"loading model={model_path} tensor_parallel_size={args.tensor_parallel_size} "
        f"data_parallel_size={args.data_parallel_size} "
        f"shard={args.shard_index + 1}/{args.num_shards} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}",
        flush=True,
    )
    llm = LLM(
        model=str(model_path),
        tokenizer=str(model_path),
        tensor_parallel_size=args.tensor_parallel_size,
        data_parallel_size=1,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        enable_prefix_caching=True,
        enforce_eager=True,
        disable_custom_all_reduce=True,
        seed=args.seed,
    )
    sampling = SamplingParams(
        n=args.samples_per_problem,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )
    for name, items in benchmark_items:
        generate_benchmark(
            name=name,
            version=args.version,
            mbpp_subset=args.mbpp_subset,
            items=items,
            llm=llm,
            sampling=sampling,
            output_dir=output_dir,
            batch_size=args.batch_size,
            samples=args.samples_per_problem,
            tensor_parallel_size=args.tensor_parallel_size,
            data_parallel_size=args.data_parallel_size,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
        )


if __name__ == "__main__":
    main()
