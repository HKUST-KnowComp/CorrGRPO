#!/usr/bin/env python3
"""Evaluate one completion per MBPP task using HumanEval's guarded executor."""

from __future__ import annotations

import argparse
import json
import os
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from human_eval.execution import check_correctness


def load_rows(base: Path, subset: str) -> list[dict[str, Any]]:
    root = base / "mbpp" / "data" / "data"
    if subset in {"test", "full"}:
        rows = [
            json.loads(line)
            for line in (root / "mbpp.jsonl").read_text().splitlines()
            if line.strip()
        ]
        if subset == "test":
            rows = [row for row in rows if 11 <= int(row["task_id"]) <= 510]
        return rows
    return json.loads((root / "sanitized-mbpp.json").read_text())


def make_problem(row: dict[str, Any], subset: str, include_challenge: bool) -> dict[str, Any]:
    if subset in {"test", "full"}:
        setup = str(row.get("test_setup_code", "")).strip()
        tests = list(row["test_list"])
        if include_challenge:
            tests.extend(row.get("challenge_test_list", []))
    else:
        setup = "\n".join(str(value) for value in row.get("test_imports", []))
        tests = list(row["test_list"])

    assertions = "\n".join(textwrap.indent(str(test), "    ") for test in tests)
    test_code = (
        "def __mbpp_candidate_sentinel():\n"
        "    pass\n\n"
        "def check(candidate):\n"
        f"{assertions}\n"
    )
    return {
        "task_id": int(row["task_id"]),
        "prompt": (setup + "\n\n") if setup else "",
        "test": test_code,
        "entry_point": "__mbpp_candidate_sentinel",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("samples_file", type=Path)
    parser.add_argument(
        "--subset", choices=["test", "full", "sanitized"], default="test"
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--include-challenge-tests", action="store_true")
    parser.add_argument("--limit", type=int, help="Smoke-test only; evaluate the first N tasks.")
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    if os.environ.get("MBPP_ALLOW_UNSAFE_EXECUTION") != "1":
        parser.error("MBPP_ALLOW_UNSAFE_EXECUTION=1 is required inside a suitable sandbox")
    if args.workers < 1 or args.timeout <= 0:
        parser.error("workers and timeout must be positive")

    rows = load_rows(args.base_dir.resolve(), args.subset)
    if args.limit is not None:
        rows = rows[: args.limit]
    problems = {
        int(row["task_id"]): make_problem(row, args.subset, args.include_challenge_tests)
        for row in rows
    }
    samples: dict[int, str] = {}
    with args.samples_file.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            sample = json.loads(line)
            task_id = int(sample["task_id"])
            if task_id in samples:
                raise ValueError(f"duplicate task_id in samples: {task_id}")
            samples[task_id] = str(sample["completion"])
    if set(samples) != set(problems):
        missing = sorted(set(problems).difference(samples))
        extra = sorted(set(samples).difference(problems))
        raise ValueError(f"sample/task mismatch: missing={missing[:5]} extra={extra[:5]}")

    results: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                check_correctness,
                problems[task_id],
                completion.rstrip() + "\n",
                args.timeout,
                0,
            ): task_id
            for task_id, completion in samples.items()
        }
        done = 0
        for future in as_completed(futures):
            task_id = futures[future]
            results[task_id] = future.result()
            done += 1
            if done % 25 == 0 or done == len(futures):
                print(f"[{done}/{len(futures)}] evaluated", flush=True)

    output_path = args.samples_file.with_name(args.samples_file.name + "_results.jsonl")
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            result = results[int(row["task_id"])]
            json.dump(result, handle, ensure_ascii=False)
            handle.write("\n")
    passed = sum(bool(result["passed"]) for result in results.values())
    summary = {
        "benchmark": "mbpp",
        "subset": args.subset,
        "count": len(results),
        "passed": passed,
        "pass@1": passed / len(results),
        "include_challenge_tests": args.include_challenge_tests,
        "partial_limit": args.limit,
        "results_file": str(output_path),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    summary_path = args.samples_file.with_name(f"mbpp_{args.subset}.eval_summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
