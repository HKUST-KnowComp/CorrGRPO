#!/usr/bin/env python3
"""Run every available reference solution four times and store mean runtimes."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from reward import compute_score, extract_code


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=SCRIPT_DIR / "raw")
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "data" / "reference_runtimes.jsonl",
    )
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
    parser.add_argument("--memory-limit-mb", type=int, default=1024)
    parser.add_argument("--limit", type=int, help="Calibrate only the first N rows.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.runs < 1 or args.workers < 1:
        parser.error("--runs and --workers must be >= 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    return args


def load_rows(raw_dir: Path) -> list[tuple[str, dict[str, Any]]]:
    rows = []
    for split in ("train", "test"):
        path = raw_dir / f"LeetCodeDataset-{split}.jsonl"
        with path.open("r", encoding="utf-8") as handle:
            rows.extend((split, json.loads(line)) for line in handle if line.strip())
    return rows


def calibrate_one(
    item: tuple[str, dict[str, Any]],
    runs: int,
    timeout_seconds: float,
    memory_limit_mb: int,
) -> dict[str, Any]:
    split, row = item
    reference_solution, _ = extract_code(str(row.get("response", "")))
    base = {
        "split": split,
        "task_id": row["task_id"],
        "question_id": row.get("question_id"),
        "runs_requested": runs,
    }
    if not reference_solution:
        return {
            **base,
            "status": "missing_reference",
            "passed_runs": 0,
            "timeout_runs": 0,
            "runtime_error_runs": 0,
            "run_times_seconds": [],
            "reference_runtime_seconds": None,
        }

    verifier = {
        "task_id": row["task_id"],
        "prompt": row["prompt"],
        "test": row["test"],
        "entry_point": row["entry_point"],
        "num_tests": len(row["input_output"]),
        "reference_solution": reference_solution,
    }
    results = [
        compute_score(
            "leetcodedataset",
            row["response"],
            verifier,
            timeout_seconds=timeout_seconds,
            memory_limit_mb=memory_limit_mb,
            efficiency_weight=0.0,
        )
        for _ in range(runs)
    ]
    run_times = [
        float(result["generated_runtime_seconds"])
        for result in results
        if result["pass_score"] == 1.0 and result["generated_runtime_seconds"] > 0.0
    ]
    passed_runs = sum(int(result["pass_score"] == 1.0) for result in results)
    all_passed = passed_runs == runs and len(run_times) == runs
    return {
        **base,
        "status": "ok" if all_passed else "reference_failed",
        "passed_runs": passed_runs,
        "timeout_runs": sum(int(result["timeout"]) for result in results),
        "runtime_error_runs": sum(int(result["runtime_error"]) for result in results),
        "run_times_seconds": run_times,
        "reference_runtime_seconds": statistics.fmean(run_times) if all_passed else None,
    }


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    partial = Path(str(output) + ".partial")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}; pass --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.raw_dir)
    if args.limit is not None:
        rows = rows[: args.limit]

    worker = lambda item: calibrate_one(
        item,
        args.runs,
        args.timeout_seconds,
        args.memory_limit_mb,
    )
    counts = {"ok": 0, "missing_reference": 0, "reference_failed": 0}
    with partial.open("w", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for index, result in enumerate(pool.map(worker, rows), start=1):
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                counts[result["status"]] += 1
                if index % 50 == 0 or index == len(rows):
                    handle.flush()
                    print(f"Calibrated {index}/{len(rows)}: {counts}", flush=True)

    os.replace(partial, output)
    print(
        json.dumps(
            {
                "output": str(output),
                "rows": len(rows),
                "runs_per_reference": args.runs,
                "workers": args.workers,
                "timeout_seconds": args.timeout_seconds,
                **counts,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
