#!/usr/bin/env python3
"""Collect numeric benchmark metrics into one stable JSON result file."""

from __future__ import annotations

import argparse
import json
import numbers
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_humaneval(output_dir: Path) -> Optional[dict[str, Any]]:
    path = output_dir / "humaneval.samples.jsonl_results.jsonl"
    if not path.is_file():
        return None
    grouped: dict[str, list[bool]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            grouped[str(row["task_id"])].append(bool(row["passed"]))
    pass_at_1 = sum(sum(values) / len(values) for values in grouped.values()) / len(grouped)
    return {
        "count": len(grouped),
        "completions": sum(len(values) for values in grouped.values()),
        "passed_completions": sum(sum(values) for values in grouped.values()),
        "pass@1": pass_at_1,
        "results_file": str(path),
    }


def collect_mbpp(output_dir: Path, subset: str) -> Optional[dict[str, Any]]:
    path = output_dir / f"mbpp_{subset}.eval_summary.json"
    if not path.is_file():
        return None
    raw = read_json(path)
    return {
        "subset": subset,
        "count": int(raw["count"]),
        "passed": int(raw["passed"]),
        "pass@1": float(raw["pass@1"]),
        "include_challenge_tests": bool(raw["include_challenge_tests"]),
        "results_file": str(
            output_dir / f"mbpp_{subset}.samples.jsonl_results.jsonl"
        ),
    }


def collect_lcb(output_dir: Path, version: str) -> Optional[dict[str, Any]]:
    path = output_dir / f"lcb_{version}.custom_outputs_codegeneration_output_eval.json"
    if not path.is_file():
        return None
    raw = read_json(path)
    metric_block = raw[0]
    numeric_metrics = {
        key: float(value)
        for key, value in metric_block.items()
        if isinstance(value, numbers.Real) and not isinstance(value, bool)
    }
    pass_details = metric_block.get("detail", {}).get("pass@1", {})
    result: dict[str, Any] = {
        "version": version,
        "count": len(pass_details),
        "metrics_file": str(path),
        **numeric_metrics,
    }
    if pass_details:
        result["passed"] = int(sum(float(value) for value in pass_details.values()))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        choices=["all", "humaneval", "mbpp", "livecodebench"],
        default="all",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--version", default="v6")
    parser.add_argument(
        "--mbpp-subset", choices=["test", "full", "sanitized"], default="test"
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    collectors = {
        "humaneval": lambda: collect_humaneval(output_dir),
        "mbpp": lambda: collect_mbpp(output_dir, args.mbpp_subset),
        "livecodebench": lambda: collect_lcb(output_dir, args.version),
    }
    required = list(collectors) if args.benchmark == "all" else [args.benchmark]
    benchmarks: dict[str, Any] = {}
    for name, collect in collectors.items():
        result = collect()
        if result is not None:
            benchmarks[name] = result
        elif name in required:
            raise FileNotFoundError(f"Required evaluation result is missing: {name}")

    payload = {
        "model": output_dir.name,
        "benchmarks": benchmarks,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    output_path = output_dir / "eval_results.json"
    temporary_path = output_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary_path.replace(output_path)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Numeric evaluation results saved to: {output_path}")


if __name__ == "__main__":
    main()
