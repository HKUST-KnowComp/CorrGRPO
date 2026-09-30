#!/usr/bin/env python3
"""Convert the official LeetCodeDataset JSONL files into verl Parquet files."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from datasets import Dataset


TRAIN_FILENAME = "LeetCodeDataset-train.jsonl"
TEST_FILENAME = "LeetCodeDataset-test.jsonl"
SOURCE_REPO = "newfacade/LeetCodeDataset"
_FENCE_RE = re.compile(r"```(?:python|py)?[ \t]*\n?(.*?)```", re.IGNORECASE | re.DOTALL)


def extract_reference_solution(response: str) -> str:
    """Extract the final Python implementation from a dataset response."""
    fenced = _FENCE_RE.findall(response.strip())
    if fenced:
        return max(fenced, key=lambda item: ("class Solution" in item, len(item))).strip()
    class_start = response.find("class Solution")
    if class_start >= 0:
        return response[class_start:].strip()
    return response.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "raw",
        help="Directory containing the two official JSONL files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="Destination for SFT and GRPO Parquet files.",
    )
    parser.add_argument(
        "--reference-runtimes",
        type=Path,
        help="JSONL produced by calibrate_reference_runtime.py (defaults under output-dir).",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            required = {
                "task_id",
                "question_id",
                "difficulty",
                "tags",
                "query",
                "response",
                "prompt",
                "test",
                "entry_point",
                "input_output",
            }
            missing = sorted(required.difference(row))
            if missing:
                raise ValueError(f"Missing fields at {path}:{line_number}: {missing}")
            rows.append(row)
    if not rows:
        raise ValueError(f"No records found in {path}")
    return rows


def load_reference_runtimes(path: Path) -> dict[tuple[str, str], float]:
    if not path.exists():
        return {}
    values: dict[tuple[str, str], float] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            runtime = record.get("reference_runtime_seconds")
            if runtime is not None and float(runtime) > 0.0:
                values[(str(record["split"]), str(record["task_id"]))] = float(runtime)
    return values


def make_sft_row(
    row: dict[str, Any], split: str, reference_runtime_seconds: float | None
) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": row["query"].strip()},
            {"role": "assistant", "content": row["response"].strip()},
        ],
        "data_source": "leetcodedataset",
        "task_id": row["task_id"],
        "question_id": row["question_id"],
        "difficulty": row["difficulty"],
        "tags": row["tags"],
        "split": split,
        "reference_runtime_seconds": reference_runtime_seconds,
    }


def make_grpo_row(
    row: dict[str, Any], split: str, reference_runtime_seconds: float | None
) -> dict[str, Any]:
    # Keep the verifier payload as a JSON string. This gives Parquet a stable
    # schema and avoids Arrow coercing nested values into numpy objects.
    verifier = {
        "task_id": row["task_id"],
        "prompt": row["prompt"],
        "test": row["test"],
        "entry_point": row["entry_point"],
        "num_tests": len(row["input_output"]),
        "reference_solution": extract_reference_solution(row["response"]),
        "reference_runtime_seconds": reference_runtime_seconds,
    }
    return {
        "data_source": "leetcodedataset",
        "prompt": [{"role": "user", "content": row["query"].strip()}],
        "ability": "code",
        "reward_model": {
            "style": "rule",
            "ground_truth": json.dumps(verifier, ensure_ascii=False, separators=(",", ":")),
        },
        "extra_info": {
            "split": split,
            "task_id": row["task_id"],
            "question_id": row["question_id"],
            "difficulty": row["difficulty"],
            "tags": row["tags"],
            "num_tests": len(row["input_output"]),
            "reference_runtime_seconds": reference_runtime_seconds,
        },
    }


def write_parquet(rows: Iterable[dict[str, Any]], path: Path, overwrite: bool) -> int:
    materialized = list(rows)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(materialized).to_parquet(str(path))
    return len(materialized)


def main() -> None:
    args = parse_args()
    raw_train = args.raw_dir / TRAIN_FILENAME
    raw_test = args.raw_dir / TEST_FILENAME
    for path in (raw_train, raw_test):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}. Download it from https://huggingface.co/datasets/{SOURCE_REPO}/tree/main"
            )

    train_rows = read_jsonl(raw_train)
    test_rows = read_jsonl(raw_test)
    easy_rows = [row for row in train_rows if row["difficulty"].lower() == "easy"]
    runtime_path = args.reference_runtimes or args.output_dir / "reference_runtimes.jsonl"
    reference_runtimes = load_reference_runtimes(runtime_path)

    def runtime_for(row: dict[str, Any], split: str) -> float | None:
        return reference_runtimes.get((split, str(row["task_id"])))

    counts = {
        "sft/train.parquet": write_parquet(
            (make_sft_row(row, "train", runtime_for(row, "train")) for row in train_rows),
            args.output_dir / "sft" / "train.parquet",
            args.overwrite,
        ),
        "sft/train_easy.parquet": write_parquet(
            (make_sft_row(row, "train", runtime_for(row, "train")) for row in easy_rows),
            args.output_dir / "sft" / "train_easy.parquet",
            args.overwrite,
        ),
        "sft/test.parquet": write_parquet(
            (make_sft_row(row, "test", runtime_for(row, "test")) for row in test_rows),
            args.output_dir / "sft" / "test.parquet",
            args.overwrite,
        ),
        "grpo/train.parquet": write_parquet(
            (make_grpo_row(row, "train", runtime_for(row, "train")) for row in train_rows),
            args.output_dir / "grpo" / "train.parquet",
            args.overwrite,
        ),
        "grpo/train_easy.parquet": write_parquet(
            (make_grpo_row(row, "train", runtime_for(row, "train")) for row in easy_rows),
            args.output_dir / "grpo" / "train_easy.parquet",
            args.overwrite,
        ),
        "grpo/test.parquet": write_parquet(
            (make_grpo_row(row, "test", runtime_for(row, "test")) for row in test_rows),
            args.output_dir / "grpo" / "test.parquet",
            args.overwrite,
        ),
    }

    manifest = {
        "source": f"https://huggingface.co/datasets/{SOURCE_REPO}",
        "raw_files": {
            TRAIN_FILENAME: {"rows": len(train_rows), "sha256": sha256(raw_train)},
            TEST_FILENAME: {"rows": len(test_rows), "sha256": sha256(raw_test)},
        },
        "derived_files": counts,
        "reference_runtime_calibration": {
            "path": str(runtime_path.resolve()),
            "available_rows": len(reference_runtimes),
        },
        "notes": {
            "sft": "messages contains user=query and assistant=response",
            "grpo": (
                "reward_model.ground_truth contains prompt, test, entry_point, "
                "and reference_solution extracted from response"
                ", plus the four-run mean reference_runtime_seconds when available"
            ),
            "easy_subset": "training records whose difficulty is Easy",
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
