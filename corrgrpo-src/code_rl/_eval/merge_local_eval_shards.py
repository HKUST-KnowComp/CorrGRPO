#!/usr/bin/env python3
"""Merge isolated vLLM data-parallel worker outputs into evaluator inputs."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def natural_id_key(value: str) -> tuple[Any, ...]:
    return tuple(
        int(piece) if piece.isdigit() else piece
        for piece in re.split(r"(\d+)", value)
    )


def selected_benchmarks(name: str) -> list[str]:
    return ["humaneval", "livecodebench", "mbpp"] if name == "all" else [name]


def prefix_for(name: str, version: str, mbpp_subset: str) -> str:
    if name == "humaneval":
        return "humaneval"
    if name == "livecodebench":
        return f"lcb_{version}"
    return f"mbpp_{mbpp_subset}"


def final_ids(name: str, path: Path) -> set[str]:
    if name == "livecodebench":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {str(row["question_id"]) for row in payload}
    return {str(row["task_id"]) for row in read_jsonl(path)}


def merge_benchmark(
    *,
    name: str,
    version: str,
    mbpp_subset: str,
    output_dir: Path,
    shard_root: Path,
    num_shards: int,
    samples: int,
    tensor_parallel_size: int,
) -> None:
    prefix_name = prefix_for(name, version, mbpp_subset)
    suffix = ".custom_outputs.json" if name == "livecodebench" else ".samples.jsonl"
    allowed_ids: set[str] = set()
    expected_count = 0
    shard_summaries: list[str] = []

    for rank in range(num_shards):
        shard_dir = shard_root / f"shard_{rank}"
        final_path = shard_dir / f"{prefix_name}{suffix}"
        summary_path = shard_dir / f"{prefix_name}.summary.json"
        if not final_path.is_file() or not summary_path.is_file():
            raise FileNotFoundError(
                f"worker {rank} did not produce {final_path.name} and its summary"
            )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if int(summary.get("shard_index", -1)) != rank:
            raise RuntimeError(f"unexpected shard metadata in {summary_path}")
        ids = final_ids(name, final_path)
        duplicate_ids = allowed_ids.intersection(ids)
        if duplicate_ids:
            raise RuntimeError(
                f"duplicate task IDs across shards; first: {sorted(duplicate_ids)[:5]}"
            )
        allowed_ids.update(ids)
        expected_count += int(summary["count"])
        shard_summaries.append(str(summary_path))

    if len(allowed_ids) != expected_count:
        raise RuntimeError(
            f"{name}: shard summaries expect {expected_count} tasks, "
            f"but final outputs contain {len(allowed_ids)} unique IDs"
        )

    records: dict[str, dict[str, Any]] = {}
    for rank in range(num_shards):
        checkpoint = shard_root / f"shard_{rank}" / f"{prefix_name}.generations.jsonl"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"missing worker checkpoint: {checkpoint}")
        for record in read_jsonl(checkpoint):
            item_id = str(record["id"])
            codes = record.get("codes", [])
            if (
                item_id in allowed_ids
                and record.get("status") == "ok"
                and len(codes) == samples
                and all(codes)
            ):
                records[item_id] = record

    missing = allowed_ids.difference(records)
    if missing:
        raise RuntimeError(f"{name}: missing valid records; first: {sorted(missing)[:5]}")

    ordered_ids = sorted(allowed_ids, key=natural_id_key)
    checkpoint_lines = "".join(
        json.dumps(records[item_id], ensure_ascii=False) + "\n"
        for item_id in ordered_ids
    )
    checkpoint_path = output_dir / f"{prefix_name}.generations.jsonl"
    atomic_write_text(checkpoint_path, checkpoint_lines)

    if name == "livecodebench":
        final_payload = [
            {"question_id": item_id, "code_list": records[item_id]["codes"]}
            for item_id in ordered_ids
        ]
        final_text = json.dumps(final_payload, ensure_ascii=False, indent=2) + "\n"
    else:
        rows = []
        for item_id in ordered_ids:
            task_id: str | int = int(item_id) if name == "mbpp" else item_id
            rows.extend(
                {"task_id": task_id, "completion": code}
                for code in records[item_id]["codes"]
            )
        final_text = "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in rows
        )
    final_path = output_dir / f"{prefix_name}{suffix}"
    atomic_write_text(final_path, final_text)

    prompt_tokens = sum(int(record.get("prompt_tokens", 0)) for record in records.values())
    completion_tokens = sum(
        sum(int(value) for value in record.get("completion_tokens", []))
        for record in records.values()
    )
    summary = {
        "benchmark": name,
        "version": version if name == "livecodebench" else None,
        "subset": mbpp_subset if name == "mbpp" else None,
        "count": len(ordered_ids),
        "samples_per_problem": samples,
        "tensor_parallel_size": tensor_parallel_size,
        "data_parallel_size": num_shards,
        "parallel_mode": "isolated_process_shards",
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "checkpoint": str(checkpoint_path),
        "final_output": str(final_path),
        "shard_summaries": shard_summaries,
        "completed_at": utc_now(),
    }
    summary_path = output_dir / f"{prefix_name}.summary.json"
    atomic_write_text(
        summary_path, json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        f"merged benchmark={name} tasks={len(ordered_ids)} "
        f"shards={num_shards} output={final_path}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        choices=["all", "humaneval", "livecodebench", "mbpp"],
        default="all",
    )
    parser.add_argument("--version", default="v6")
    parser.add_argument(
        "--mbpp-subset", choices=["test", "full", "sanitized"], default="test"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--samples-per-problem", type=int, default=1)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    args = parser.parse_args()

    if args.num_shards < 2:
        parser.error("num-shards must be at least 2")
    output_dir = args.output_dir.resolve()
    shard_root = args.shard_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in selected_benchmarks(args.benchmark):
        merge_benchmark(
            name=name,
            version=args.version,
            mbpp_subset=args.mbpp_subset,
            output_dir=output_dir,
            shard_root=shard_root,
            num_shards=args.num_shards,
            samples=args.samples_per_problem,
            tensor_parallel_size=args.tensor_parallel_size,
        )


if __name__ == "__main__":
    main()
